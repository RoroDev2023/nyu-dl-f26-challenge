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


@torch.no_grad()
def teacher_probs(teachers, x, temp):
    """Ensemble teacher: the average of the teachers' softened predictions."""
    probs = [F.softmax(t(x).float() / temp, dim=1) for t in teachers]
    return torch.stack(probs).mean(0)


def kd_loss(logits, probs, temp):
    """KL(teacher || student) at temperature temp. The temp^2 keeps its gradients on the
    same scale as the cross entropy (Hinton et al. 2015)."""
    log_student = F.log_softmax(logits / temp, dim=1)
    return F.kl_div(log_student, probs, reduction="batchmean") * temp ** 2


@torch.no_grad()
def sam_perturb(params, rho, adaptive, scale=1.0):
    """SAM: move the weights to the nearby worst case, w + e, using the gradients of the
    first pass. Returns the original weights for sam_restore.
    scale undoes GradScaler: with fp16 the gradients are still multiplied by it here.

    SAM:  e = rho * g / ||g||
    ASAM: e = rho * w^2 * g / ||w * g||  (the ball is scaled by the size of each weight)"""
    params = [p for p in params if p.grad is not None]
    grads = [p.grad / scale for p in params]
    scaled = [p.abs() * g for p, g in zip(params, grads)] if adaptive else grads
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in scaled]))

    backup = [p.clone() for p in params]
    if torch.isfinite(norm):      # an fp16 overflow gives inf: then skip the perturbation
        for p, g in zip(params, scaled):
            p.add_((p.abs() * g if adaptive else g) * rho / (norm + 1e-12))
    return params, backup


@torch.no_grad()
def sam_restore(params, backup):
    for p, saved in zip(params, backup):
        p.copy_(saved)
