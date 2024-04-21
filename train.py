#
# Created on Mon Oct 16 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Train a GNN model and track experiments via wandb
# ------------------------------
#

import sys
import os
import torch
import argparse
import time
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from copy import deepcopy
import wandb
from datetime import datetime
import math
from GraphModels import GraphConvWat, ChebNet, GATResMeanConv, m_GCN, GCN2, GAT, GIN
import gc
from utils.early_stopping import EarlyStopping
from utils.gradient_clipping import GradientClipping
from utils.auxil import *
from utils.DataLoader import WDNDataset,  get_stacked_set
from typing import Callable
import evaluation
from ConfigModels import select_model, config_gatres_small
torch.cuda.empty_cache()
gc.collect()


def get_default_datasets(args: argparse.Namespace)-> tuple[WDNDataset, WDNDataset, WDNDataset]:
    """get default datasets for training

    Args:
        args (argparse.Namespace): default arguments

    Returns:
        tuple[WDNDataset, WDNDataset, WDNDataset]: training dataset, valid dataset, testing dataset
    """
    edge_attrs = args.use_data_edge_attrs.split(',') if args.use_data_edge_attrs is not None else None
    
    train_ds = WDNDataset(zip_file_paths=args.dataset_paths,
                            input_paths=args.input_paths,
                            feature=args.feature,
                            from_set='train',
                            num_records=args.num_trains,
                            removal=args.train_val_removal,
                            do_scale=True,
                            mean=None,
                            std=None,
                            min=None,
                            max=None,
                            lazy_convert_pygdata=False,
                            edge_attrs=edge_attrs,#['diameter','length'],
                            edge_mean=None,
                            edge_std=None,
                            edge_min=None,
                            edge_max=None,
                            norm_type=args.norm_type,
                            )
    
    val_ds =  WDNDataset(zip_file_paths=args.dataset_paths,
                            input_paths=args.input_paths,
                            feature=args.feature,
                            from_set='valid',
                            num_records=None,
                            removal=args.train_val_removal,
                            do_scale=True,
                            mean=train_ds.mean,
                            std=train_ds.std,
                            min=train_ds.min,
                            max=train_ds.max,
                            lazy_convert_pygdata=False,
                            edge_attrs=edge_attrs,#['diameter','length'],
                            edge_mean=train_ds.edge_mean,
                            edge_std=train_ds.edge_std,
                            edge_min=train_ds.edge_min,
                            edge_max=train_ds.edge_max,
                            norm_type=args.norm_type,
                        )
        
    test_ds = get_stacked_set(zip_file_path=args.test_data_path,#fullnode
                                input_path=args.test_input_path,
                                feature=args.feature,
                                edge_attrs=edge_attrs,
                                train_mean=train_ds.mean,
                                train_std=train_ds.std,
                                train_max=train_ds.max,
                                train_min=train_ds.min,
                                train_edge_mean=train_ds.edge_mean,
                                train_edge_std=train_ds.edge_std, 
                                train_edge_max=train_ds.edge_max,
                                train_edge_min=train_ds.edge_min,
                                norm_type=args.norm_type,)
    return train_ds, val_ds, test_ds

def train_one_epoch(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loader: DataLoader,
                    mask_rate: float,
                    device:str,
                    mean:Any,
                    std:Any,
                    min_val:Any,
                    max_val:Any,
                    norm_type:str,
                    criterion:Callable,
                    metric_fn_dict:dict,
                    edge_attrs:list,
                    use_data_batch:bool,
                    use_gradient_clipping:bool,
                    grad_clipper: GradientClipping=None,) -> tuple[float,dict,dict,Any]:
    """train an epoch

    Args:
        model (torch.nn.Module): training model 
        optimizer (torch.optim.Optimizer): torch optimizer
        loader (DataLoader): training loader
        mask_rate (float): masking ratio
        device (str): hardware device name
        mean (Any): mean values of training dataset
        std (Any): std values of training dataset
        min_val (Any): min values of training dataset
        max_val (Any): max values of training dataset
        norm_type (str): normalization type. minmax or znorm or unused
        criterion (Callable): loss function
        metric_fn_dict (dict): dict of metric callable functions
        edge_attrs (list): list of edge attributes if used.
        use_data_batch (bool): flag indicates whether using data batch size. Default is False
        use_gradient_clipping (bool): flag indicates whether using gradient clipping. Default is False
        grad_clipper (GradientClipping, optional): gradient clipper object to compute clipping amount. Defaults to None.

    Returns:
        tuple[float,dict,dict,Any]: _description_
    """
    model.train()
    
    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    record_metric_dict = {}
    
    for data in loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.

        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)

        
        data_edge_attr = data.edge_attr.to(device) if edge_attrs else None
        data_batch = data.batch.to(device) if use_data_batch else None
        
        #data.batch has shape [batch_size * max_of_num_nodes_across_graphs] 
        #num_nodes has shape [batch_size] 
        num_nodes = torch.unique(data.batch, return_counts=True)[1] 
        batch_mask = generate_batch_mask(num_nodes=num_nodes, mask_rate=mask_rate, required_idx=[])

        data.x[batch_mask] = 0
        out = model(data.x, data.edge_index, data_batch, data_edge_attr)  # CHANGE THIS FOR COORD


        y_pred = out[batch_mask]
        y_true = data.y[batch_mask]


        y_pred_rescaled = descale(scaled_data=y_pred,norm_type=norm_type,mean=mean,std=std,max=max_val,min=min_val)
        y_true_rescaled = descale(scaled_data=y_true,norm_type=norm_type,mean=mean,std=std,max=max_val,min=min_val)

        tr_loss = criterion(y_pred, y_true)

        tr_loss.backward()  # Derive gradients.
        if use_gradient_clipping and grad_clipper is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipper.cache_gradient_and_compute_norm(model))
        optimizer.step()  # Update parameters based on gradients.

        total_loss += float(tr_loss) * data.num_graphs
        for k, fn in metric_fn_dict.items():
            computed_metric = fn(y_pred_rescaled, y_true_rescaled)
            if computed_metric.size():
                  if k not in record_metric_dict:
                      record_metric_dict[k] = []
                  record_metric_dict[k].append(computed_metric)
            else:
              total_metric_dict[k] += computed_metric * data.num_graphs

    
    metric_dict = {k : total_metric_dict[k] / len(loader.dataset) for k in total_metric_dict.keys() }
    
    return total_loss / len(loader.dataset), metric_dict, record_metric_dict, out

def test_one_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   mask_rate: float,
                   device:str,
                   mean:Any,
                   std:Any,
                   min_val:Any,
                   max_val:Any,
                   norm_type:str, 
                   criterion:Callable, 
                   metric_fn_dict:dict, 
                   edge_attrs:list, 
                   use_data_batch:bool) -> tuple[float, dict, dict]:
    
    """test function

    Args:
        model (torch.nn.Module): training model 
        loader (DataLoader): training loader
        mask_rate (float): masking ratio
        device (str): hardware device name
        mean (Any): mean values of training dataset
        std (Any): std values of training dataset
        min_val (Any): min values of training dataset
        max_val (Any): max values of training dataset
        norm_type (str): normalization type. minmax or znorm or unused
        criterion (Callable): loss function
        metric_fn_dict (dict): dict of metric callable functions
        edge_attrs (list):  list of edge attributes if used
        use_data_batch (bool): flag indicates whether using data batch size. Default is False

    Returns:
        tuple[float, dict, dict]: loss, dict of all unknown estimated measurements, dict of sensor estimated measurements.
    """
    model.eval()

    total_loss = 0
    total_metric_dict = {k: 0 for k in metric_fn_dict.keys()}
    record_metric_dict = {}

    with torch.no_grad():
        for data in loader:
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            data.edge_index = data.edge_index.to(device)
                
            
            data_edge_attr = data.edge_attr.to(device) if edge_attrs else None
            data_batch = data.batch.to(device) if use_data_batch else None
            num_nodes = torch.unique(data.batch, return_counts=True)[1]
            batch_mask = generate_batch_mask(num_nodes=num_nodes, mask_rate=mask_rate,required_idx=[])

            data.x[batch_mask] = 0
            out = model(data.x, data.edge_index, data_batch, data_edge_attr)  # CHANGE THIS FOR COORD

            y_pred = out[batch_mask]  # y_pred.masked_select(mask)
            y_true = data.y[batch_mask]  # y_true.masked_select(mask)

            y_pred_rescaled = descale(scaled_data=y_pred,norm_type=norm_type,mean=mean,std=std,max=max_val,min=min_val)
            y_true_rescaled = descale(scaled_data=y_true,norm_type=norm_type,mean=mean,std=std,max=max_val,min=min_val)


            val_loss = criterion(y_pred, y_true)

            total_loss += float(val_loss) * data.num_graphs
            for k, fn in metric_fn_dict.items():
                computed_metric = fn(y_pred_rescaled, y_true_rescaled)
                if computed_metric.size():
                  if k not in record_metric_dict:
                      record_metric_dict[k] = []
                  record_metric_dict[k].append(computed_metric)
                else:
                  total_metric_dict[k] += computed_metric * data.num_graphs

        
        metric_dict = {k : total_metric_dict[k] / len(loader.dataset) for k in total_metric_dict.keys() }
        return total_loss / len(loader.dataset), metric_dict, record_metric_dict

def internal_train(args: argparse.Namespace,
                   model: torch.nn.Module, 
                   train_ds: WDNDataset, 
                   val_ds: WDNDataset,
                   test_ds: WDNDataset, 
                   do_load: bool=True) -> tuple[float, dict, dict] :
    """perform a full train

    Args:
        args (argparse.Namespace): default arguments
        model (torch.nn.Module): training model
        train_ds (WDNDataset): training dataset
        val_ds (WDNDataset): validation dataset
        test_ds (WDNDataset): testing dataset
        do_load (bool, optional): allow to load trained weights into model for contineous training. Defaults to True.

    Raises:
        FileNotFoundError: model path may be unfounded
        KeyError: criterion may not supported

    """
    edge_attrs = args.use_data_edge_attrs.split(',') if args.use_data_edge_attrs is not None else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device =  args.device 

    assert model is not None
    args.model_name =  model.name if model.name is not None else type(model).__name__
    
    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    if do_load:
        if not hasattr(args,'model_path'):
            print(f'model_path is not found! Please distinguish it from save_path folder')
        elif not os.path.exists(args.model_path):
            raise FileNotFoundError(f'{args.model_path} file is not found')
        else:
            model,_ = load_checkpoint(args.model_path, model)

    model = model.to(device)

    print('#'*80)
    postfix = datetime.today().strftime('%Y%m%d_%H%M')
    # start a new wandb run to track this script
    if args.log_method == 'wandb':
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name, 
            name = f'{args.model_name}_{args.variant}_{postfix}' if args.variant else f'{args.model_name}_{postfix}',
            # track hyperparameters and run metadata
            config=dict(vars(args))
        )
    print('args list:')
    for k,v in vars(args).items():
        print(f'{k} = {v}')
    
    print('#'*80)
    
    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    
    early_stop = EarlyStopping(mode="min",
                                min_delta=args.min_delta,
                                patience=args.patience) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.scheduler_patience)
    else:
        scheduler = None

    if args.criterion is not None and args.criterion == 'sce':
        def sce_loss(x, y, alpha = 3):
            x = F.normalize(x,p=2,dim=-1)
            y = F.normalize(y,p=2,dim=-1)
            loss = (1.0 - (x*y).sum(dim=-1)).pow_(alpha)
            return loss.mean()

        criterion = sce_loss
    elif args.criterion == 'mse':
        criterion = torch.nn.MSELoss(reduction="mean").to(device)
    elif args.criterion == 'mae':
        criterion = torch.nn.L1Loss(reduction="mean").to(device)
    else:
        raise KeyError(f'criterion {args.criterion} is not supported')

    
    
    best_loss = np.inf
    best_epoch = 0
    best_metric_dict =  best_record_metric_dict  = {}
    start_time = time.time()
    dt1 = datetime.fromtimestamp(start_time)
    print('Start time:', dt1)
    print("*" * 80)
    os.makedirs(args.save_path,exist_ok=True)
    #model_artifact = wandb.Artifact(args.model_name, type='model')

    train_metric_fn_dict= get_metric_fn_collection(prefix='train')
    val_metric_fn_dict=  get_metric_fn_collection(prefix='val')

    mean = train_ds.mean
    std = train_ds.std
    max_val = train_ds.max
    min_val = train_ds.min

    grad_clipper = GradientClipping(percentile=args.percentile) if args.use_gradient_clipping else None

    for epoch in range(1, args.epochs + 1):

        tr_loss, tr_metric_dict, tr_record_metric_dict, out = train_one_epoch(model=model, optimizer=optimizer , loader=train_loader, #num_nodes=num_nodes,
                                                    mask_rate=args.mask_rate, device=device, mean=mean, std=std, max_val=max_val, min_val=min_val, norm_type=args.norm_type, criterion=criterion, metric_fn_dict=train_metric_fn_dict, edge_attrs=edge_attrs, use_data_batch=args.use_data_batch, 
                                                    use_gradient_clipping= args.use_gradient_clipping, grad_clipper=grad_clipper,
                                                    )
        
        val_loss, val_metric_dict, val_record_metric_dict = test_one_epoch(model=model,loader=val_loader, #num_nodes=num_nodes,
                                                        mask_rate=args.mask_rate, device=device, mean=mean, std=std, max_val=max_val, min_val=min_val, norm_type=args.norm_type,  criterion=criterion, metric_fn_dict=val_metric_fn_dict, edge_attrs=edge_attrs, use_data_batch=args.use_data_batch)
       
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_metric_dict = val_metric_dict
            best_record_metric_dict = val_record_metric_dict
            best_epoch = epoch

            # # save training_checkpoint
            save_checkpoint(
                path= os.path.join(args.save_path,f"best_{args.model_name}_{args.variant}.pth"),
                model_state_dict=model.state_dict(), 
                optimizer_state_dict=optimizer.state_dict() if optimizer else None, 
                epoch=best_epoch, 
                loss=best_loss, 
                val_metric_dict = best_metric_dict, 
                val_record_metric_dict=best_record_metric_dict, 
                mean=train_ds.mean,
                std=train_ds.std,
                min=train_ds.min,
                max=train_ds.max,
                edge_attrs=edge_attrs,
                edge_mean=train_ds.edge_mean,
                edge_std=train_ds.edge_std,
                edge_min=train_ds.edge_min,
                edge_max=train_ds.edge_max,
                norm_type=args.norm_type, 
            )

        if epoch == 1 or (epoch % 5) == 0 :
            print_metrics(epoch=epoch,tr_loss=tr_loss, val_loss=val_loss,tr_metric_dict=tr_record_metric_dict, val_metric_dict=val_metric_dict)
            if not math.isnan(tr_loss):
                save_checkpoint(
                    path= os.path.join(args.save_path,f"last_{args.model_name}_{args.variant}.pth"),
                    model_state_dict=model.state_dict(), 
                    optimizer_state_dict=optimizer.state_dict() if optimizer else None, 
                    epoch=best_epoch, 
                    loss=best_loss, 
                    val_metric_dict = val_metric_dict, 
                    val_record_metric_dict=val_record_metric_dict, 
                    mean=train_ds.mean,
                    std=train_ds.std,
                    min=train_ds.min,
                    max=train_ds.max,
                    edge_attrs=edge_attrs,
                    edge_mean=train_ds.edge_mean,
                    edge_std=train_ds.edge_std,
                    edge_min=train_ds.edge_min,
                    edge_max=train_ds.edge_max,
                    norm_type=args.norm_type, 
                )

        if args.log_method == 'wandb':
            if args.log_gradient:
                if epoch == 1:
                    first_out = out
                    model_update = 0
                    accum_model_update = 0
                else:
                    model_update = torch.abs(out - first_out).mean()
                accum_model_update += model_update
                grad_norm, block_norms, block_names =  get_gradient_norm(model,norm_type=2)
                    
                log_metrics_on_wandb(epoch=epoch,
                                     commit=False,
                                     total_grad_norm=grad_norm,
                                     model_update=model_update,
                                     accum_model_update=accum_model_update,
                                     )
                
            log_metrics_on_wandb(epoch=epoch,
                                 commit=True,
                                 train_loss=tr_loss,
                                 val_loss=val_loss,
                                 best_loss=best_loss,
                                 best_epoch=best_epoch,
                                 tr_metric_dict=tr_metric_dict,
                                 val_metric_dict=val_metric_dict,
                                 )

           
        if early_stop.step(torch.tensor(val_loss)):
            print(f"\n!! No improvement for {args.patience} epochs. Training stopped!")
            break   
                
        if scheduler is not None:
            scheduler.step(val_loss)  
    
    end_time = time.time()
    dt2 = datetime.fromtimestamp(end_time)
    
    print("*" * 80)
    print('End time:', dt2)
    print('Training time:', dt2 - dt1)

    wandb.finish()
    ###################
    #TEST HERE
    ###################
    if args.do_test:
        trained_model,_ = load_checkpoint(path=os.path.join(args.save_path,f"best_{args.model_name}_{args.variant}.pth"), model=model)
        # by default, testing is clean and unshared mask
        testing_args = evaluation.convert_train_2_test_arguments(args)
        
        return evaluation.internal_test(args=testing_args,
                            model=trained_model,
                            train_ds=train_ds,
                            test_ds=test_ds,
                            do_load=False)
    else:
        return best_loss, best_metric_dict, best_record_metric_dict 

    
def train(args: argparse.Namespace, model: torch.nn.Module =None, do_load=True):
    train_ds, val_ds, test_ds = get_default_datasets(args)

    return internal_train(args=args,
                        model=model,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        test_ds=test_ds,
                        do_load=do_load)

def get_arguments(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='gatres_small',type=str,choices=['gatres_small','gatres_large','gin','graphconvwat','chebnet','mgcn','gcn2','gat'], help="support model selection only.")
    parser.add_argument('--lr',default=0.0005,type=float, help="Learning rate. Default is 0.0005")
    parser.add_argument('--weight_decay',default=0.000006,type=float, help="weight decay. Default is 0.000006")
    parser.add_argument('--epochs',default=5,type=int, help="number of epochs to train the model")
    parser.add_argument('--mask_rate',default= 0.95,type=float, help="masking ratio. Default is 0.95")
    parser.add_argument('--dataset_paths',default=['datasets/ctown.zip'],type=str, nargs='*', action='store', help="list of dataset paths used for training and validation (order-sensitive)")
    parser.add_argument('--input_paths',default=['inputs/ctown.inp'],type=str, nargs='*', action='store', help="list of WDN input paths used for training and validation (order-sensitive)")
    parser.add_argument('--do_test',default= False,type=bool, help="after training, we evaluate the model on clean or noisy tests. However, we should evaluate a different pipeline. As such, this flag is set to False by default.")
    parser.add_argument('--test_data_path',default= r"G:\.shortcut-targets-by-id\1uoKIPvTJgynIObYCS0Sktu0qX-U6wiSg\DiTEC\Data\Datasets[ size _ datatypes _ simtype _ notes _ name _ ddmmyyyy ]\0k288_hp_EPYNET_fullnodetime_Val-GEN-09 Oosterbeek_31032023.zip",type=str, help="timed dataset path for testing")
    parser.add_argument('--test_input_path',default= r"G:\.shortcut-targets-by-id\1uoKIPvTJgynIObYCS0Sktu0qX-U6wiSg\DiTEC\Data\Datasets[ size _ datatypes _ simtype _ notes _ name _ ddmmyyyy ]\WDN input files\Val-GEN-09Oosterbeek_20233101.inp",type=str, help="timed input path for testing")
    parser.add_argument('--feature',default= "pressure", choices=["pressure", "head"], type=str, help="feature input")
    parser.add_argument('--variant',default= '',type=str, help="Please give a value for model's variant")
    parser.add_argument('--model_name',default=None,type=str, help="Name of model. Keep its empty to use the name of class by default")
    parser.add_argument('--criterion',default='mse', choices=["mse", "mae", 'sce'],type=str, help="criterion loss. Support mse|sce|mae")
    parser.add_argument('--num_trains',default=4, type=int, help="Number of train records. Set None to use all")
    parser.add_argument('--batch_size',default=2, type=int, help="batch size")
    parser.add_argument('--use_data_batch',default=False, type=bool, help="pass pyg data batch as parameter")
    parser.add_argument('--use_data_edge_attrs',default=None, type=str, help="pass pyg data edge attributes. Support: diameter| length| None")
    parser.add_argument('--patience',default=100, type=int, help="Early stopping patience in these epochs. If val_loss unchanges, the training is stopped")
    parser.add_argument('--min_delta',default=1e-4, type=float, help="delta between last_loss and best_loss")
    parser.add_argument('--train_val_removal',default='keep_list', choices=["keep_all", "keep_list", "keep_junction", "reservoir", "tank"], type=str, help="simple-keep_all, tough-keep_list. Node removal strategy to remove different nodal types in the water network. Support: keep_list| reservoir| tank| keep_junction| keep_all")
    parser.add_argument('--device',default='cuda', type=str, choices=['cuda','cpu'], help="Training device. If gpu is unavailable, device is set to cpu. Support: cuda| cpu")
    parser.add_argument('--use_gradient_clipping',default=False, type=bool, help="Flag indicates gradient clipping is used in training")
    parser.add_argument('--percentile',default=10., type=float, help="percentile from historical gradients used for gradient clipping. Only used when use_gradient_clipping is True")
    parser.add_argument('--scheduler',default=None, type=str, choices=['ReduceLROnPlateau',None], help="scheduler name. Support ReduceLROnPlateau. Set None if unused")#
    parser.add_argument('--scheduler_patience',default=2, type=int, help="scheduler patience. Should be less than patience of early stopping")#
    parser.add_argument('--norm_type',default='znorm', choices=["znorm", "minmax", "unused"], type=str, help="normalization type. Support znorm| minmax|unused or None")
    ######TRACKING EXPERIMENTS SETTINGS######
    parser.add_argument('--log_method',default=None, choices=["wandb", None], type=str, help="log method! support wandb|None")
    parser.add_argument('--log_gradient',default=True, type=bool, help="flag indicates keeping track of gradient flow")
    parser.add_argument('--project_name',default='test_project', type=str, help="name of tracking project")
    parser.add_argument('--save_path',default='experiments_logs/test_args/fun_test', type=str, help="Path to store model weights")
    #########################################
    parser.add_argument('--num_test_trials',default=10, type=str, help="Repeat the inference on test set N times with diff masks. The report will include mean and std in N times")
    
    args = parser.parse_args(args=raw_args)
    return args

if __name__ == "__main__":
    ##########EXAMPLE TO RUN GATRES##################
    #args = get_arguments(sys.argv)
    #args, model = config_gatres_small(args,'GATRes_Small_znorm_15b_32c')
    #train(args,model=model,do_load=False)
    #################################################


    #get default argument and parse from terminal
    args = get_arguments(sys.argv[1:])
    #select based on 
    args, model = select_model(args,None)
    #train
    train(args,model=model,do_load=False)

