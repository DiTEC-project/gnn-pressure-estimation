#
# Created on Mon Oct 16 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: test a trained model and track experiments using wandb
# ------------------------------
#
import sys
import os
import time
import argparse
import wandb
from datetime import datetime
import torch
import torch.nn.functional as F
from utils.DataLoader import WDNDataset, NoisyWDNDataset, get_stacked_set2
from utils.auxil import *
from utils.timer import Timer
from ConfigModels import config_gatres_small, select_model
import epynet
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Callable, Any
from collections import defaultdict


def get_sensors(test_input_path: str, feature: str, include_reservoir: bool) -> tuple[list, list]:
    """get sensors from secret files. It won't work if you don't have a secret file

    Args:
        test_input_path (str): test input water network INP file path
        feature (str): feature indicating what type of sensors
        include_reservoir (bool): flag indicates whether including reservoir nodes

    Raises:
        NotImplementedError: some sensors from other WDNs should be listed/ imported from this file

    Returns:
        tuple[list,list]: returned list of sensor indices and list of sensor nodal names. In case that sensors are unavailable, return two empty lists
    """
    file_name = os.path.basename(test_input_path)
    if "oosterbeek" in file_name.lower():
        try:
            from utils.mysecrets import secrets

            wn_test = epynet.Network(test_input_path)
            required_idx = []
            if feature in ["pressure", "head"]:
                required_uids = secrets["OOSTERBEEK_NODE_SENSORS"]
                required_idx = [node.index for node in wn_test.nodes if node.uid in required_uids]
            else:
                required_uids = secrets["OOSTERBEEK_LINK_SENSORS"]
                if not include_reservoir:
                    required_uids = required_uids[:-3]
                required_idx = [link.index for link in wn_test.links if link.uid in required_uids]

            assert required_idx, "The WDN has no required uids. Please correct the input file path."
            return required_idx, required_uids
        except ImportError:
            print("Warning! Secrets are not found! Sensors are unavailable! The results will be similar to all cases! Return w/o inference!")
            return [], []

    else:
        # raise NotImplementedError('put sensor ids and its uids here')
        print("ERROR! Sensors or secrets are not found!Return w/o train!Return w/o inference!")
        return [], []


def get_default_datasets(args: argparse.Namespace, mean_dmd=0.1, std_dmd=1.0) -> tuple[WDNDataset, WDNDataset]:
    """support function to get default dataset preparing for an evaluation

    Args:
        args (argparse.Namespace): default arguments
        mean_dmd (float, optional): mean of noisy injected into demands. Defaults to 0.1.
        std_dmd (_type_, optional): std of noisy injected into demands. Defaults to 1.

    Returns:
        tuple[WDNDataset, WDNDataset]: a Training dataset for gathering mean, std, min max and a Testing dataset for evaluation
    """
    edge_attrs = args.use_data_edge_attrs.split(",") if args.use_data_edge_attrs is not None else None

    train_ds = WDNDataset(
        zip_file_paths=args.dataset_paths,
        input_paths=args.input_paths,
        feature=args.feature,
        from_set="train",
        num_records=args.num_trains,
        removal=args.train_val_removal,
        do_scale=True,
        mean=None,
        std=None,
        min=None,
        max=None,
        lazy_convert_pygdata=False,
        edge_attrs=edge_attrs,  # ['diameter','length'],
        edge_mean=None,
        edge_std=None,
        edge_min=None,
        edge_max=None,
        norm_type=args.norm_type,
    )

    if args.test_single_snapshot:
        if args.test_type in ["noisy11", "noisyNN"]:
            test_ds = [
                NoisyWDNDataset(
                    input_paths=args.input_paths,  # <- it assumely has 1 single snapshot
                    feature=args.feature,
                    num_records=args.num_tests,
                    removal=args.test_removal,
                    do_scale=True,
                    mean=train_ds.mean,
                    std=train_ds.std,
                    min=train_ds.min,
                    max=train_ds.max,
                    lazy_convert_pygdata=False,
                    edge_attrs=edge_attrs,
                    edge_mean=train_ds.edge_mean,
                    edge_std=train_ds.edge_std,
                    edge_min=train_ds.edge_min,
                    edge_max=train_ds.edge_max,
                    norm_type=args.norm_type,
                    mean_dmd=mean_dmd,
                    std_dmd=std_dmd,
                )
                for _ in range(args.num_test_trials)
            ]
        else:  # clean
            test_ds = NoisyWDNDataset(
                input_paths=args.input_paths,  # <- it assumely has 1 single snapshot
                feature=args.feature,
                num_records=args.num_tests,
                removal=args.test_removal,
                do_scale=True,
                mean=train_ds.mean,
                std=train_ds.std,
                min=train_ds.min,
                max=train_ds.max,
                lazy_convert_pygdata=False,
                edge_attrs=edge_attrs,
                edge_mean=train_ds.edge_mean,
                edge_std=train_ds.edge_std,
                edge_min=train_ds.edge_min,
                edge_max=train_ds.edge_max,
                norm_type=args.norm_type,
                mean_dmd=0.0,
                std_dmd=0.0,
            )
    else:
        if args.test_type in ["noisy11", "noisyNN"]:
            test_ds = [
                NoisyWDNDataset(
                    input_paths=[args.test_input_path],
                    feature=args.feature,
                    num_records=args.num_tests,
                    removal=args.test_removal,
                    do_scale=True,
                    mean=train_ds.mean,
                    std=train_ds.std,
                    min=train_ds.min,
                    max=train_ds.max,
                    lazy_convert_pygdata=False,
                    edge_attrs=edge_attrs,
                    edge_mean=train_ds.edge_mean,
                    edge_std=train_ds.edge_std,
                    edge_min=train_ds.edge_min,
                    edge_max=train_ds.edge_max,
                    norm_type=args.norm_type,
                    mean_dmd=mean_dmd,
                    std_dmd=std_dmd,
                )
                for _ in range(args.num_test_trials)
            ]
        else:  # clean
            if args.test_from_set != "all":
                if args.test_from_set == "inp":
                    test_ds = NoisyWDNDataset(
                        input_paths=[args.test_input_path],
                        feature=args.feature,
                        num_records=args.num_tests,
                        removal=args.test_removal,
                        do_scale=True,
                        mean=train_ds.mean,
                        std=train_ds.std,
                        min=train_ds.min,
                        max=train_ds.max,
                        lazy_convert_pygdata=False,
                        edge_attrs=edge_attrs,
                        edge_mean=train_ds.edge_mean,
                        edge_std=train_ds.edge_std,
                        edge_min=train_ds.edge_min,
                        edge_max=train_ds.edge_max,
                        norm_type=args.norm_type,
                        mean_dmd=0.0,
                        std_dmd=0.0,
                    )
                else:
                    test_ds = WDNDataset(
                        zip_file_paths=[args.test_data_path],
                        input_paths=[args.test_input_path],
                        feature=args.feature,
                        from_set=args.test_from_set,
                        num_records=args.num_tests,
                        removal=args.test_removal,
                        do_scale=True,
                        mean=train_ds.mean,
                        std=train_ds.std,
                        min=train_ds.min,
                        max=train_ds.max,
                        lazy_convert_pygdata=False,
                        edge_attrs=edge_attrs,
                        edge_mean=train_ds.edge_mean,
                        edge_std=train_ds.edge_std,
                        edge_min=train_ds.edge_min,
                        edge_max=train_ds.edge_max,
                        norm_type=args.norm_type,
                    )

            else:
                test_ds = get_stacked_set2(
                    zip_file_path=args.test_data_path,  # fullnode
                    input_path=args.test_input_path,
                    feature=args.feature,
                    num_tests=args.num_tests,
                    edge_attrs=edge_attrs,
                    train_mean=train_ds.mean,
                    train_std=train_ds.std,
                    train_max=train_ds.max,
                    train_min=train_ds.min,
                    train_edge_mean=train_ds.edge_mean,
                    train_edge_std=train_ds.edge_std,
                    train_edge_max=train_ds.edge_max,
                    train_edge_min=train_ds.edge_min,
                    norm_type=args.norm_type,
                    removal=args.test_removal,
                )
    return train_ds, test_ds


def test_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    do_test_on_sensors: bool,
    mask_rate: float,
    device: str,
    mean: Any,
    std: Any,
    min_val: Any,
    max_val: Any,
    norm_type: str,
    criterion: Callable,
    metric_fn_dict: dict,
    edge_attrs: list[str],
    use_data_batch: bool,
    test_input_path: str,
    feature: str,
    gpu_warmup_times: int,
    include_reservoir: bool = False,
    use_same_mask: bool = False,
) -> tuple[float, dict]:
    """test one epoch/ trial

    Args:
        model (torch.nn.Module): testing model
        loader (DataLoader): loader containing testing samples
        do_test_on_sensors (bool): flag indicates whether test is on sensors only
        mask_rate (float): masking ratio
        device (str): device hardware
        mean (Any): mean values from training dataset
        std (Any): std values from training dataset
        min_val (Any): min values
        max_val (Any): max values from training dataset
        norm_type (str): norm type. minmax, znorm or unused
        criterion (Callable): loss criterion callable function
        metric_fn_dict (dict): dict contain name of metric and callable functions
        edge_attrs (list[str]): list of edge attrs used in testing
        use_data_batch (bool): flag indicates whether we use data batch size
        test_input_path (str): testing water network INP file path
        feature (str): support pressure and head
        gpu_warmup_times (int): times used to warm up gpus for time-related measurments
        include_reservoir (bool, optional): flag indicates whether involving reservoir. Defaults to False.
        use_same_mask (bool, optional): flag indicates whether using same mask through this testing batch. Defaults to False.

    Returns:
        tuple[float, dict]: returned loss and metrics as dict
    """
    model.eval()
    if do_test_on_sensors:
        required_idx, required_uid = get_sensors(test_input_path=test_input_path, feature=feature, include_reservoir=include_reservoir)
        print(f"CHECKING SENSORS: {required_idx}")
        postfix = "_sensor"
    else:
        required_idx = []
        postfix = ""

    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    all_timer = Timer()
    created_all_mask = False
    all_mask = None

    with torch.no_grad():
        for data in loader:
            data.x = data.x
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)

            data_edge_attr = data.edge_attr.to(device) if edge_attrs else None
            data_batch = data.batch.to(device) if use_data_batch else None
            num_nodes = torch.unique(data.batch, return_counts=True)[1]

            data_x1 = torch.clone(data.x).to(device)

            if (use_same_mask and not created_all_mask) or not use_same_mask:
                created_all_mask = True
                all_mask = generate_batch_mask(
                    num_nodes=num_nodes,
                    mask_rate=mask_rate,
                    required_idx=required_idx,
                )

            data_x1[all_mask] = 0

            wrapped_model = all_timer.auto_measure(model, num_graphs_per_batch=data.num_graphs, gpu_warmup_times=gpu_warmup_times)
            # out = model(data_x1, data.edge_index, data_batch , data_edge_attr)
            out = wrapped_model(data_x1, data.edge_index, data_batch, data_edge_attr)

            y_pred = out[all_mask]  # y_pred.masked_select(mask)
            y_true = data.y[all_mask]  # y_true.masked_select(mask)

            y_pred_rescaled = descale(scaled_data=y_pred, norm_type=norm_type, mean=mean, std=std, max=max_val, min=min_val)
            y_true_rescaled = descale(scaled_data=y_true, norm_type=norm_type, mean=mean, std=std, max=max_val, min=min_val)

            val_loss = criterion(y_pred, y_true)
            total_loss += float(val_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                total_metric_dict[k] += computed_metric * data.num_graphs

    len_dataset = len(loader.dataset)
    test_loss = total_loss / len_dataset

    test_metric_dict = {k: total_metric_dict[k] / len_dataset for k in total_metric_dict.keys()}

    prefix = list(metric_fn_dict.keys())[0].split("_")[0]
    test_metric_dict[prefix + "_time"] = all_timer.compute_time(len_dataset)
    test_metric_dict[prefix + "_throughput"] = all_timer.compute_throughput(len_dataset)

    test_metric_dict = {k + postfix: v for k, v in test_metric_dict.items()}

    return test_loss, test_metric_dict


def test_clean(
    model: torch.nn.Module,
    test_ds: WDNDataset,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    min_val: Any,
    max_val: Any,
    norm_type: str,
    criterion: Callable,
    metric_fn_dict: dict,
    edge_attrs: list[str],
) -> tuple[list, dict, list, dict]:
    test_losses = []
    test_metrics_dict = defaultdict(list)

    test_sensor_losses = []
    test_sensor_metrics_dict = defaultdict(list)

    test_batch_size = args.batch_size
    repeat_test_time = args.num_test_trials
    loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
    for i in tqdm(range(repeat_test_time)):
        trial = i
        test_loss, test_metric_dict, test_sensor_loss, test_sensor_metric_dict = test_and_collect_once(
            model=model,
            loader=loader,
            trial=trial,
            args=args,
            device=device,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            norm_type=norm_type,
            criterion=criterion,
            metric_fn_dict=metric_fn_dict,
            edge_attrs=edge_attrs,
        )

        test_losses.append(test_loss)
        for k in test_metric_dict.keys():
            test_metrics_dict[k].append(test_metric_dict[k])

        test_sensor_losses.append(test_sensor_loss)
        for k in test_sensor_metric_dict.keys():
            test_sensor_metrics_dict[k].append(test_sensor_metric_dict[k])

    return test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict


def test_noisyNN(
    model: torch.nn.Module,
    test_ds: WDNDataset,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    min_val: Any,
    max_val: Any,
    norm_type: str,
    criterion: Callable,
    metric_fn_dict: dict,
    edge_attrs: list[str],
) -> tuple[list, dict, list, dict]:
    test_losses = []
    test_metrics_dict = defaultdict(list)

    test_sensor_losses = []
    test_sensor_metrics_dict = defaultdict(list)

    test_batch_size = args.batch_size
    repeat_test_time = args.num_test_trials

    total_trials = repeat_test_time * len(test_ds)
    i = 0
    assert isinstance(test_ds, list)
    with tqdm(total=total_trials) as pbar:
        for sub_ds in test_ds:
            sub_loader = DataLoader(sub_ds, batch_size=test_batch_size, shuffle=False)
            for trial in range(repeat_test_time):
                test_loss, test_metric_dict, test_sensor_loss, test_sensor_metric_dict = test_and_collect_once(
                    model=model,
                    loader=sub_loader,
                    trial=trial,
                    args=args,
                    device=device,
                    mean=mean,
                    std=std,
                    min_val=min_val,
                    max_val=max_val,
                    norm_type=norm_type,
                    criterion=criterion,
                    metric_fn_dict=metric_fn_dict,
                    edge_attrs=edge_attrs,
                )

                test_losses.append(test_loss)
                for k in test_metric_dict.keys():
                    test_metrics_dict[k].append(test_metric_dict[k])

                test_sensor_losses.append(test_sensor_loss)
                for k in test_sensor_metric_dict.keys():
                    test_sensor_metrics_dict[k].append(test_sensor_metric_dict[k])
                pbar.update(1)
                i += 1
            del sub_loader

    return test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict


def test_noisy11(
    model: torch.nn.Module,
    test_ds: WDNDataset,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    min_val: Any,
    max_val: Any,
    norm_type: str,
    criterion: Callable,
    metric_fn_dict: dict,
    edge_attrs: list[str],
) -> tuple[list, dict, list, dict]:
    test_losses = []
    test_metrics_dict = defaultdict(list)

    test_sensor_losses = []
    test_sensor_metrics_dict = defaultdict(list)

    test_batch_size = args.batch_size

    total_trials = len(test_ds)
    i = 0
    assert isinstance(test_ds, list)
    with tqdm(total=total_trials) as pbar:
        for sub_ds in test_ds:
            sub_loader = DataLoader(sub_ds, batch_size=test_batch_size, shuffle=False)
            trial = i
            test_loss, test_metric_dict, test_sensor_loss, test_sensor_metric_dict = test_and_collect_once(
                model=model,
                loader=sub_loader,
                trial=trial,
                args=args,
                device=device,
                mean=mean,
                std=std,
                min_val=min_val,
                max_val=max_val,
                norm_type=norm_type,
                criterion=criterion,
                metric_fn_dict=metric_fn_dict,
                edge_attrs=edge_attrs,
            )

            test_losses.append(test_loss)
            for k in test_metric_dict.keys():
                test_metrics_dict[k].append(test_metric_dict[k])

            test_sensor_losses.append(test_sensor_loss)
            for k in test_sensor_metric_dict.keys():
                test_sensor_metrics_dict[k].append(test_sensor_metric_dict[k])
            pbar.update(1)
            i += 1
            del sub_loader

    return test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict


def test_and_collect_once(
    model: torch.nn.Module,
    loader: DataLoader,
    trial: int,
    args: argparse.Namespace,
    device: str,
    mean: Any,
    std: Any,
    min_val: Any,
    max_val: Any,
    norm_type: str,
    criterion: Callable,
    metric_fn_dict: dict,
    edge_attrs: list[str],
) -> tuple[float, dict, float, dict]:
    # for all nodes
    test_loss, test_metric_dict = test_one_epoch(
        model=model,
        loader=loader,
        do_test_on_sensors=False,
        mask_rate=args.mask_rate,
        device=device,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        norm_type=norm_type,
        criterion=criterion,
        metric_fn_dict=metric_fn_dict,
        edge_attrs=edge_attrs,
        use_data_batch=args.use_data_batch,
        test_input_path=args.test_input_path,
        feature=args.feature,
        gpu_warmup_times=args.gpu_warmup_times if trial == 0 else 0,
        use_same_mask=args.use_same_mask,
        include_reservoir=False,
    )

    # for estimating and comparing pressure at nodes equipped sensors
    test_sensor_loss, test_sensor_metric_dict = test_one_epoch(
        model=model,
        loader=loader,
        do_test_on_sensors=True,  # <= on sensors only
        mask_rate=args.mask_rate,
        device=device,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        norm_type=norm_type,
        criterion=criterion,
        metric_fn_dict=metric_fn_dict,
        edge_attrs=edge_attrs,
        use_data_batch=args.use_data_batch,
        test_input_path=args.test_input_path,
        feature=args.feature,
        gpu_warmup_times=0,
        use_same_mask=args.use_same_mask,
        include_reservoir=False,
    )

    if args.log_method == "wandb":
        log_metrics_on_wandb(
            epoch=trial,
            is_epoch_a_trial=True,
            commit=True,
            test_loss=test_loss,
            test_sensor_loss=test_sensor_loss,
            test_metric_dict=test_metric_dict,
            test_sensor_metric_dict=test_sensor_metric_dict,
        )

    return test_loss, test_metric_dict, test_sensor_loss, test_sensor_metric_dict


def internal_test(
    args: argparse.Namespace, model: torch.nn.Module, train_ds: WDNDataset, test_ds: WDNDataset, do_load: bool = True
) -> tuple[dict, dict, dict]:
    """perform a full test

    Args:
        args (argparse.Namespace): default arguments
        model (torch.nn.Module): testing models
        train_ds (WDNDataset): training dataset only for gather mean, std, min, max
        test_ds (WDNDataset): testing dataset
        do_load (bool, optional): allows to load weights into model. Defaults to True.

    Raises:
        FileNotFoundError: model path or secret path may be not found
        KeyError: a criterion may be not supported
        NotImplementedError: testing strategy may be not supported

    Returns:
        tuple[dict,dict,dict]: dict of loss, dict of metric on all unknown nodes, dict of metric on nodes that are sensors only in N runs
    """
    assert args.test_type in ["clean", "noisy11", "noisyNN"]
    edge_attrs = args.use_data_edge_attrs.split(",") if args.use_data_edge_attrs is not None else None

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    assert model is not None
    args.model_name = model.name if model.name is not None else type(model).__name__

    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    if do_load:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"{args.model_path} file is not found")
        param_dict = torch.load(args.model_path)
        model.load_state_dict(param_dict["model_state_dict"])

    model = model.to(device)

    print("#" * 80)
    postfix = datetime.today().strftime("%Y%m%d_%H%M")
    # start a new wandb run to track this script
    if args.log_method == "wandb":
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name,
            name=f"{args.model_name}_{args.variant}_{postfix}" if args.variant else f"{args.model_name}_{postfix}",
            # track hyperparameters and run metadata
            config=dict(vars(args)),
        )
    print("args list:")
    for k, v in vars(args).items():
        print(f"{k} = {v}")

    print("#" * 80)

    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))

    if args.criterion is not None and args.criterion == "sce":

        def sce_loss(x, y, alpha=3):
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
            return loss.mean()

        criterion = sce_loss
    elif args.criterion == "mse":
        criterion = torch.nn.MSELoss(reduction="mean").to(device)
    elif args.criterion == "mae":
        criterion = torch.nn.L1Loss(reduction="mean").to(device)
    else:
        raise KeyError(f"criterion {args.criterion} is not supported")

    test_metric_fn_dict = get_metric_fn_collection(prefix="test")
    mean = train_ds.mean
    std = train_ds.std
    max_val = train_ds.max
    min_val = train_ds.min
    norm_type = args.norm_type

    start_time = time.time()
    dt1 = datetime.fromtimestamp(start_time)
    print("Start time:", dt1)
    print("*" * 80)
    ##########################################

    test_type = args.test_type
    if test_type == "clean":
        test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict = test_clean(
            model=model,
            test_ds=test_ds,
            args=args,
            device=device,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            norm_type=norm_type,
            criterion=criterion,
            metric_fn_dict=test_metric_fn_dict,
            edge_attrs=edge_attrs,
        )
    elif test_type == "noisyNN":
        test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict = test_noisyNN(
            model=model,
            test_ds=test_ds,
            args=args,
            device=device,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            norm_type=norm_type,
            criterion=criterion,
            metric_fn_dict=test_metric_fn_dict,
            edge_attrs=edge_attrs,
        )
    elif test_type == "noisy11":
        test_losses, test_metrics_dict, test_sensor_losses, test_sensor_metrics_dict = test_noisy11(
            model=model,
            test_ds=test_ds,
            args=args,
            device=device,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            norm_type=norm_type,
            criterion=criterion,
            metric_fn_dict=test_metric_fn_dict,
            edge_attrs=edge_attrs,
        )
    else:
        raise NotImplementedError(f"test type {test_type} is not implemented")

    trials = len(test_losses)

    mean_test_loss, std_test_loss = np.mean(test_losses), np.std(test_losses) + 1e-6
    mean_test_sensor_loss, std_test_sensor_loss = np.mean(test_sensor_losses), np.std(test_sensor_losses) + 1e-6

    out_test_metric_dict = {}
    out_test_sensor_metric_dict = {}
    for k in test_metrics_dict.keys():
        x = torch.tensor(test_metrics_dict[k])
        out_test_metric_dict[f"{k}_mean"], out_test_metric_dict[f"{k}_std"] = torch.mean(x), torch.std(x) + 1e-6
    for k in test_sensor_metrics_dict.keys():
        x = torch.tensor(test_sensor_metrics_dict[k])
        out_test_sensor_metric_dict[f"{k}_mean"], out_test_sensor_metric_dict[f"{k}_std"] = torch.mean(x), torch.std(x) + 1e-6

    print_multitest_metrics(
        trials=trials,
        mean_test_loss=mean_test_loss,
        std_test_loss=std_test_loss,
        mean_test_sensor_loss=mean_test_sensor_loss,
        std_test_sensor_loss=std_test_sensor_loss,
        out_test_metric_dict=out_test_metric_dict,
        out_test_sensor_metric_dict=out_test_sensor_metric_dict,
    )

    out_test_loss_dict = {
        "test_loss_mean": mean_test_loss,
        "test_loss_std": std_test_loss,
        "test_loss_sensor_mean": mean_test_sensor_loss,
        "test_loss_sensor_std": std_test_sensor_loss,
    }

    if args.log_method == "wandb":
        log_metrics_on_wandb(
            epoch=trials,
            is_epoch_a_trial=True,
            commit=True,
            out_test_loss_dict=out_test_loss_dict,
            out_test_metric_dict=out_test_metric_dict,
            out_test_sensor_metric_dict=out_test_sensor_metric_dict,
        )

    ##########################################
    end_time = time.time()
    dt2 = datetime.fromtimestamp(end_time)
    print("*" * 80)
    print("End time:", dt2)
    print("Testing time:", dt2 - dt1)
    wandb.finish()

    return out_test_loss_dict, out_test_metric_dict, out_test_sensor_metric_dict


def test(args: argparse.Namespace, model: torch.nn.Module = None, do_load=True) -> tuple[dict, dict, dict]:
    """get the default datasets and perform a full test on testing dataset

    Args:
        args (argparse.Namespace): default arguments
        model (torch.nn.Module, optional): testing model. Defaults to None.
        do_load (bool, optional): flag allows to load weights into model. Defaults to True.

    Returns:
        tuple[dict,dict,dict]: dict of loss, dict of metric on all unknown nodes, dict of metric on nodes that are sensors only in N runs
    """
    train_ds, test_ds = get_default_datasets(args)
    return internal_test(args=args, model=model, train_ds=train_ds, test_ds=test_ds, do_load=do_load)


def convert_train_2_test_arguments(train_args: argparse.Namespace) -> argparse.Namespace:
    """support converting train arguments to test arguments

    Args:
        train_args (argparse.Namespace): training arguments

    Returns:
        argparse.Namespace: testing arguments
    """
    test_args = get_arguments([])
    exception_keys = ["log_method", "project_name"]
    for k, v in vars(train_args).items():
        if k in test_args and k not in exception_keys:
            setattr(test_args, k, v)
    return test_args


def get_arguments(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gatres_small",
        type=str,
        choices=["gatres_small", "gatres_large", "gin", "graphconvwat", "chebnet", "mgcn", "gcn2", "gat"],
        help="support model selection only.",
    )
    parser.add_argument("--model_path", default="", type=str, help="path to load the trained weights")
    parser.add_argument("--mask_rate", default=0.95, type=float, help="Please give a value for mask_rate")
    parser.add_argument(
        "--dataset_paths",
        default=[
            r"",
        ],
        type=str,
        nargs="*",
        action="store",
        help="list of dataset paths used for training and validation (order-sensitive)",
    )

    parser.add_argument(
        "--input_paths",
        default=[r""],
        type=str,
        nargs="*",
        action="store",
        help="list of WDN input paths used for training and validation (order-sensitive)",
    )

    parser.add_argument("--test_data_path", default=r"", type=str, help="timed dataset path for testing")  # 24hour
    parser.add_argument("--test_input_path", default=r"", type=str, help="timed input path for testing")
    parser.add_argument(
        "--test_removal",
        default="keep_junction",
        type=str,
        choices=["keep_all", "keep_list", "keep_junction", "reservoir", "tank"],
        help="Node removal strategy to remove different nodal types in the water network. If you don't know, use keep_junction",
    )
    parser.add_argument("--feature", default="pressure", choices=["pressure", "head"], type=str, help="feature input")
    parser.add_argument("--variant", default=datetime.now().strftime("%Y%M%d"), type=str, help="Please give a value for model's variant")
    parser.add_argument("--model_name", default=None, type=str, help="Name of model ")
    parser.add_argument("--criterion", default="mse", type=str, help="criterion loss. Support mse|sce|mae")
    parser.add_argument("--num_trains", default=None, type=int, help="Number of train records. Set None to use all")
    parser.add_argument("--batch_size", default=80, type=int, help="batch size")
    parser.add_argument("--use_data_batch", default=False, type=bool, help="pass pyg data batch as parameter")
    parser.add_argument("--use_data_edge_attrs", default=None, type=str, help="pass pyg data edge attributes. Support: diameter| length| None")
    parser.add_argument(
        "--train_val_removal",
        default="keep_all",
        type=str,
        help="simple-keep_all, tough-keep_list. Node removal strategy. Support: keep_list| reservoir| tank| keep_junction| keep_all",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="Training device. If gpu is unavailable, device is set to cpu. Support: cuda| cpu",
    )
    parser.add_argument("--norm_type", default="znorm", type=str, help="normalization type. Support znorm| minmax| unused")
    ######TRACKING EXPERIMENTS SETTINGS######
    parser.add_argument("--log_method", default=None, type=str, help="log method! support wandb|aim|None")
    parser.add_argument("--project_name", default="fun_evaluation_test", type=str, help="name of tracking project")
    #########################################
    parser.add_argument(
        "--num_test_trials",
        default=10,
        type=int,
        help="Repeat the inference on test set N times with diff masks. The report will include mean and std in N times",
    )
    parser.add_argument(
        "--test_type",
        default="clean",
        type=str,
        choices=["clean", "noisy11", "noisyNN"],
        help="test strategy. clean- test with a clean scenario and repeat N times | noisy11- test N times with a pair of noise scene and random mask | noisyNN- test N noise scenes in which each is tested with N masks, so total is NxN cases ",
    )
    parser.add_argument("--gpu_warmup_times", default=10, type=int, help="Perform warmup inference N times before measuring latency and throughput")
    parser.add_argument(
        "--use_same_mask",
        default=False,
        type=bool,
        help="Flag indicates whether a single mask is applied to all snapshots in a test scenario. Default is False",
    )
    parser.add_argument(
        "--test_single_snapshot",
        default=False,
        type=bool,
        help="Flag indicates whether testing on a single snapshot. If yes, we take dataset_paths[0] as the testing input path because it assumely has only one snapshot ",
    )
    parser.add_argument("--test_norm_type", default="znorm", type=str, help="normalization type. Support znorm| minmax| unused")
    parser.add_argument(
        "--test_from_set",
        default="all",
        type=str,
        choices=["all", "train", "valid", "test", "inp"],
        help="ONLY FOR CLEAN TEST. Test on which set. Default is all",
    )
    parser.add_argument("--num_tests", default=None, type=int, help="ONLY FOR CLEAN TEST. Number of test records. Set None to use all. ")

    args = parser.parse_args(args=raw_args)
    return args


if __name__ == "__main__":
    # if int(args.num_test_trials) <= 10:
    #    seed = 1234
    #    torch.manual_seed(seed)
    #    np.random.seed(seed)

    ##########EXAMPLE TO TEST GATRES IN A CLEAN TEST##################
    # margs, model = config_gatres_small(get_arguments([]),'gatressmall_test100noisyNNnew')
    # test(margs,model=model)
    # wandb.finish()
    #################################################

    args = get_arguments(sys.argv[1:])
    args, model = select_model(args, None, reset_model_path=True)
    test(args, model=model)
