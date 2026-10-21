"""Normalized forward KL / hard CE; one entropy term remains in native PPO."""
import torch


def kd_weight(t, initial=10.0, plateau=0.1, t1=66000, t2=100000):
    if not 0 < t1 < t2 or not 0 <= plateau <= initial:
        raise ValueError("Invalid three-stage annealing schedule")
    if t < 0:
        raise ValueError("Negative environment step")
    if t < t1:
        return initial - (initial - plateau) * t / t1
    return plateau if t < t2 else 0.0


def distillation_loss(student_probs, teacher_probs, valid, mode):
    """Inputs [B,T,N,6], valid [B,T,N]; normalize only over supervised entries."""
    if mode not in {"soft", "hard"}:
        raise ValueError(mode)
    if student_probs.shape != teacher_probs.shape or valid.shape != student_probs.shape[:-1]:
        raise ValueError("Teacher/student/mask alignment mismatch")
    p = teacher_probs.detach()
    selected = valid.bool()
    if not selected.any():
        return student_probs.sum() * 0.0
    p, q = p[selected], student_probs[selected]
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5):
        raise ValueError("Invalid teacher probabilities")
    log_q = q.clamp_min(1e-10).log()
    if mode == "hard":
        return -log_q.gather(-1, p.argmax(-1, keepdim=True)).mean()
    # xlogy handles 0*log(0), without adding mass or applying softmax twice.
    return (torch.special.xlogy(p, p) - p * log_q).sum(-1).mean()
