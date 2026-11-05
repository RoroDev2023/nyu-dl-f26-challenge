"""Training tricks used by train.py: EMA, mixup/CutMix, knowledge distillation, SAM."""
import copy

import numpy as np
import torch
import torch.nn.functional as F


class EMA:
    """Exponential moving average of the weights: a slowly moving copy of the model.
    The decay starts low so the early, random weights are forgotten quickly."""
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        ema_state = self.module.state_dict()
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema_state[name].lerp_(value, 1 - decay)
            else:                                   # BatchNorm's batch counter
                ema_state[name].copy_(value)

    def state_dict(self):
        return {"model": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state):
        self.module.load_state_dict(state["model"])
        self.updates = state["updates"]


def mix_batch(x, y, mixup_alpha, cutmix_alpha, mix_prob):
    """Mix a batch with a shuffled copy of itself. Returns x, y_a, y_b, lam, where the
    target is lam * y_a + (1 - lam) * y_b. Picks mixup or CutMix 50/50 if both are on."""
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return x, y, y, 1.0
    if np.random.rand() >= mix_prob:
        return x, y, y, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
    lam = np.random.beta(*[cutmix_alpha if use_cutmix else mixup_alpha] * 2)
    perm = torch.randperm(len(x), device=x.device)

    if not use_cutmix:
        return lam * x + (1 - lam) * x[perm], y, y[perm], float(lam)

    # CutMix: paste a box from the shuffled batch, its area is about (1 - lam)
    h, w = x.shape[2:]
    cut = np.sqrt(1 - lam)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip([cy - int(h * cut) // 2, cy + int(h * cut) // 2], 0, h)
    x1, x2 = np.clip([cx - int(w * cut) // 2, cx + int(w * cut) // 2], 0, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)   # the box may be clipped at the border
    return x, y, y[perm], float(lam)


def mixed_ce(logits, y_a, y_b, lam, label_smoothing=0.0):
    loss = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
    if lam < 1:
        loss = lam * loss + (1 - lam) * F.cross_entropy(logits, y_b,
                                                        label_smoothing=label_smoothing)
    return loss


def standardize(logits):
    """Z-score each image's logit vector (Sun et al. 2024, logit-standardization-KD):
    (z - mean) / (std + 1e-7), torch.std as in the reference code."""
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (1e-7 + std)


@torch.no_grad()
def teacher_probs(teachers, x, temp, stand=False):
    """Ensemble teacher: the average of the teachers' softened predictions.
    stand: z-score each teacher's logits first, then soften, then average."""
    probs = []
    for t in teachers:
        z = t(x).float()
        probs.append(F.softmax((standardize(z) if stand else z) / temp, dim=1))
    return torch.stack(probs).mean(0)


def kd_loss(logits, probs, temp, stand=False):
    """KL(teacher || student) at temperature temp. The temp^2 keeps its gradients on the
    same scale as the cross entropy (Hinton et al. 2015). stand: z-score the student's
    logits too (the teacher side is done in teacher_probs)."""
    z = standardize(logits) if stand else logits
    log_student = F.log_softmax(z / temp, dim=1)
    return F.kl_div(log_student, probs, reduction="batchmean") * temp ** 2


@torch.no_grad()
def sam_perturb(params, rho, adaptive, scale=1.0, eta=0.01, is_weight=None):
    """SAM: move the weights to the nearby worst case, w + e, using the gradients of the
    first pass. Returns the original weights for sam_restore.
    scale undoes GradScaler: with fp16 the gradients are still multiplied by it here.

    SAM:  e = rho * g / ||g||
    ASAM: e = rho * T^2 g / ||T g||, T = |w| + eta for params named "*weight*", T = 1 for
          biases. This is the authors' code (SamsungLabs/ASAM, asam.py ascent_step).
    is_weight: one bool per param (ASAM only); default treats every param as a weight."""
    keep = [i for i, p in enumerate(params) if p.grad is not None]
    params = [params[i] for i in keep]
    flags = [True] * len(params) if is_weight is None else [is_weight[i] for i in keep]
    grads = [p.grad / scale for p in params]
    if adaptive:
        t = [p.abs() + eta if w else torch.ones_like(p) for p, w in zip(params, flags)]
        tg = [ti * g for ti, g in zip(t, grads)]
        steps = [ti * x for ti, x in zip(t, tg)]       # T^2 g
    else:
        tg = steps = grads
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in tg]))

    backup = [p.clone() for p in params]
    if torch.isfinite(norm):      # an fp16 overflow gives inf: then skip the perturbation
        for p, s in zip(params, steps):
            p.add_(s * rho / (norm + 1e-12))
    return params, backup


@torch.no_grad()
def sam_restore(params, backup):
    for p, saved in zip(params, backup):
        p.copy_(saved)
