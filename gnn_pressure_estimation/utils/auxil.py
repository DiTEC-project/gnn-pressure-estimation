#
# Created on Tue Sep 26 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Collect random functions
# ------------------------------
#

import numpy as np
from copy import deepcopy
import torch
from functools import partial
import torch.nn.functional as F
from typing import Union, Any
import wandb


def scale(data: Any, norm_type: str = "minmax", mean: Any = None, std: Any = None, min: Any = None, max: Any = None, eps: float = 1e-8) -> Any:
    """scale function supports normalization

    Args:
        data (Any): unnormed data
        norm_type (str, optional): minmax or znorm . Defaults to "minmax".
        mean (Any, optional): mean values. Defaults to None.
        std (Any, optional): std values. Defaults to None.
        min (Any, optional): min values. Defaults to None.
        max (Any, optional): max values. Defaults to None.
        eps (float, optional): a very small number to avoid zero division. Defaults to 1e-8.

    Returns:
        Any: normed data
    """
    assert norm_type in ["minmax", "znorm"]
    if norm_type == "minmax":
        assert min and max, "min and max values are missing"
        return (data - min) / (max - min)
    elif norm_type == "znorm":
        assert mean and std, "mean and std values are missing"
        return (data - mean) / (std + eps)


def descale(scaled_data: Any, norm_type: str = "minmax", mean: Any = None, std: Any = None, min: Any = None, max: Any = None) -> Any:
    """Descale function supports denormalization

    Args:
        scaled_data (Any): normed data
        norm_type (str, optional): znorm or minmax. Defaults to "minmax".
        mean (Any, optional): mean values. Defaults to None.
        std (Any, optional): std values. Defaults to None.
        min (Any, optional): min values. Defaults to None.
        max (Any, optional): max values. Defaults to None.

    Returns:
        Any: unnormed data
    """
    if norm_type == "minmax":
        assert min and max, "min and max values are missing"
        data = (scaled_data * (max - min)) + min
    elif norm_type == "znorm":
        assert mean and std, "mean and std values are missing"
        data = (scaled_data * std) + mean
    else:
        data = scaled_data
    return data


def pretty_print(my_dict: dict[str, Any], indent: int = 4) -> None:
    import json

    def custom_serializer(obj):
        """Handle unserializable objects like datetime, set, etc."""
        if isinstance(obj, set):
            return list(obj)  # convert sets to lists
        return str(obj)  # fallback: Convert unknown objects to strings

    print(json.dumps(my_dict, indent=indent, sort_keys=True, default=custom_serializer))
    # for k, v in my_dict.items():
    #     if isinstance(v, dict):
    #         pretty_print(my_dict=v, indent=indent*2)
    #     else:
    #         print(f"{k:<{indent}}:\t{str(v):}\n")


def nx_to_pyg(data: Any, graph: "nx.graph") -> "torch_geometric.data.Data":
    """convert nx graph and data into pyg Data format

    Args:
        data (Any):
        graph (nx.graph): nx graph containing topology only

    Returns:
        torch_geometric.data.Data: pyg data format
    """
    g_data = deepcopy(graph)
    y = data
    g_data.y = torch.Tensor(np.reshape(y, [-1, 1]))
    g_data.x = torch.Tensor(np.reshape(y, [-1, 1]))
    return g_data


def calculate_nse(y_pred, y_true, exponent=2):
    raveled_y_pred = torch.ravel(y_pred)
    raveled_y_true = torch.ravel(y_true)
    return 1.0 - torch.div(
        torch.sum(torch.pow(raveled_y_pred - raveled_y_true, exponent)),
        torch.sum(torch.pow(raveled_y_true - torch.mean(raveled_y_true), exponent)) + 1e-12,
    )


def calculate_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def calculate_rel_error(y_pred, y_true):
    err = torch.abs(torch.subtract(y_true, y_pred))
    mask = torch.abs(y_true) > 0.01
    rel_err = torch.abs(torch.divide(err[mask], y_true[mask]))
    return torch.mean(rel_err)


def calculate_accuracy(y_pred, y_true, threshold=0.2):
    mae = torch.abs(torch.subtract(y_true, y_pred))
    acc = (mae <= (y_true * threshold)).float()
    return torch.mean(acc)


def calculate_correlation_coefficient(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
    # cov    = torch.mul(y_pred-y_pred.mean(), y_true-y_true.mean()).mean()
    # std   = torch.sqrt(torch.mul(torch.square(y_pred-y_pred.mean()), torch.square(y_true-y_true.mean()))).mean()

    return torch.clamp(cost, -1.0, 1.0)


def calculate_r2(y_pred, y_true):
    r = calculate_correlation_coefficient(y_pred, y_true)
    return r**2


def mask_nodes(num_nodes: int, masking_rate: float, required_idx: list[int]) -> np.ndarray:
    """function supports to build a mask array

    Args:
        num_nodes (int): number of available nodes
        masking_rate (float): masking ratio
        required_idx (list[int]): indices required to be masked. Set none if unused

    Returns:
        np.ndarray: binary mask array
    """
    mask_length = int(num_nodes * masking_rate) - len(required_idx)
    assert mask_length > 0
    selected_nodes = list(set(range(num_nodes)).difference(required_idx))
    idx = np.random.choice(selected_nodes, mask_length, replace=False)
    mask = np.zeros(num_nodes)
    mask[idx] = 1
    mask[required_idx] = 1
    assert len(mask[mask == 1]) == int(num_nodes * masking_rate)
    mask = mask.astype(bool)
    return mask  # .reshape(-1, 1)


def generate_batch_mask(num_nodes: int, mask_rate: float, required_idx: list[int]) -> np.ndarray:
    """generate a batch of mask arrays

    Args:
       num_nodes (int): number of available nodes
        masking_rate (float): masking ratio
        required_idx (list[int]): indices required to be masked. Set none if unused

    Returns:
        np.ndarray: _description_
    """

    def decorator(i):
        return mask_nodes(i, mask_rate, required_idx)

    test = np.hstack(list(map(decorator, num_nodes)))
    return test


def get_metric_fn_collection(prefix: str) -> dict:
    """util creating metric funtions

    Args:
        prefix (str): set a prefix name for tracking these experiment

    Returns:
        dict: contains functional name and callable functions
    """
    metric_fn_dict = {
        f"{prefix}_error": calculate_rel_error,
        f"{prefix}_0.1": partial(calculate_accuracy, threshold=0.1),
        f"{prefix}_corr": calculate_correlation_coefficient,
        f"{prefix}_r2": calculate_r2,
        f"{prefix}_mae": F.l1_loss,
        f"{prefix}_rmse": calculate_rmse,
        f"{prefix}_mynse": partial(calculate_nse, exponent=2),
    }
    return metric_fn_dict


def load_checkpoint(path: str, model: torch.nn.Module) -> tuple[torch.nn.Module, dict]:
    """support load model and relevant data

    Args:
        path (str): checkpoint file
        model (torch.nn.Module): model architecture to load weights into

    Returns:
        tuple[torch.nn.Module, dict]: tuple of loaded model and relevant data as dict
    """
    assert path[-4:] == ".pth"
    assert model is not None
    cp_dict = torch.load(path)
    model.load_state_dict(cp_dict["model_state_dict"])
    return model, cp_dict


def save_checkpoint(path: str, **kwargs) -> str:
    """support save checkpoint. User can leverage kwargs to store model and relevant data

    Args:
        path (str): saved path

    Returns:
        str: saved path
    """
    torch.save(kwargs, path)
    return path


def print_metrics(epoch: int, tr_loss: float, val_loss: float, tr_metric_dict: dict, val_metric_dict: dict):
    """support beautifying string format

    Args:
        epoch (int): current epoch/ trial
        tr_loss (float): training loss
        val_loss (float): validation loss
        tr_metric_dict (dict): training metric dict including name, values
        val_metric_dict (dict): validation metric dict including name, values
    """
    metric_log = ""

    for k, v in tr_metric_dict.items():
        metric_log += f"{k}: {v:.4f}, "

    for k, v in val_metric_dict.items():
        metric_log += f"{k}: {v:.4f}, "

    print(f"Epoch: {epoch:03d}, train loss: {tr_loss:.4f}, val_loss: {val_loss:.4f}, {metric_log}")


def print_multitest_metrics(
    trials: int,
    mean_test_loss: float,
    std_test_loss: float,
    mean_test_sensor_loss: float,
    std_test_sensor_loss: float,
    out_test_metric_dict: dict,
    out_test_sensor_metric_dict: dict,
):
    """support beautifying string format for multi-trial evaluation

    Args:
        trials (int): number of trials
        mean_test_loss (float): mean of test losses
        std_test_loss (float): std of test losses
        mean_test_sensor_loss (float): mean of test sensor losses
        std_test_sensor_loss (float): std of test sensor losses
        out_test_metric_dict (dict): metric dict contains mean, std and raw measurment values in trials runs
        out_test_sensor_metric_dict (dict): metric dict contains mean, std and raw measurment sensor values in trials runs
    """
    metric_log = ""
    for k, v in out_test_metric_dict.items():
        if "mean" in k:
            name = k[:-5]
            std = out_test_metric_dict[f"{name}_std"]
            metric_log += f"{name}: {v:.4f} +/- {std:.4f}, "
    for k, v in out_test_sensor_metric_dict.items():
        if "mean" in k:
            name = k[:-5]
            std = out_test_sensor_metric_dict[f"{name}_std"]
            metric_log += f"{name}: {v:.4f} +/- {std:.4f}, "

    print(f"\nThis TEST experiment reports the average result of {trials} runs.")
    print(
        f"test_loss: {mean_test_loss:.4f} +/- {std_test_loss:.4f}, test_loss_sensor: {mean_test_sensor_loss:.4f} +/- {std_test_sensor_loss:.4f}, {metric_log}"
    )


def get_gradient_norm(model: torch.nn.Module, norm_type=2) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """support get gradient norms from a model. Tracking modules whose names has 'block', 'mlp' or 'res' words

    Args:
        model (torch.nn.Module): any torch module
        norm_type (int, optional): calculated type of norm. Defaults to 2.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], list[str]]: returned sum norms of all blocks, list of each norms, list of each name
    """
    block_norms = []
    block_names = []
    for name, param in model.named_parameters():
        if "block" in name or "mlp" in name or "res" in name:
            block_norms.append(torch.norm(param.grad.detach(), p=norm_type))
            block_names.append(name)

    total_norm = torch.norm(torch.stack(block_norms), norm_type)
    return total_norm, block_norms, block_names


def log_metrics_on_wandb(epoch: int, commit: bool = True, is_epoch_a_trial=False, **kwargs):
    """support function allowing to push log to wandb server

    Args:
        epoch (int): deterministic epoch
        commit (bool, optional): if it is one of non-last incremental logs, set it to True. Defaults to True.
    """
    for k, v in kwargs.items():
        if isinstance(v, dict):
            wandb.log(v, commit=False)
        else:
            wandb.log({k: v}, commit=False)
    if is_epoch_a_trial:
        wandb.log({"trial": epoch}, commit=commit)
    else:
        wandb.log({"epoch": epoch}, commit=commit)
