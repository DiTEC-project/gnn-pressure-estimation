import numpy as np
import torch
from typing import Any
def get_grad_norm(model):
    """
    Adapted from AutoClip: Adaptive Gradient Clipping
    Prem Seetharaman, Gordon Wichern, Bryan Pardo, Jonathan Le Roux. "AutoClip: Adaptive Gradient
    Clipping for Source Separation Networks." 2020 IEEE 30th International Workshop on Machine Learning
    for Signal Processing (MLSP). IEEE, 2020.

    https://github.com/pseeth/autoclip
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class GradientClipping:
    def __init__(self, percentile: float =10) -> None:
        self.grad_history = []
        self.percentile = percentile

    def reset(self):
        self.grad_history.clear()

    def cache_gradient_and_compute_norm(self, model: torch.nn.Module) -> float:
        self.grad_history.append(get_grad_norm(model))
        clip_value = np.percentile(self.grad_history, self.percentile)
        return clip_value