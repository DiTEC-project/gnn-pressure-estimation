#
# Created on Fri Oct 13 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Set a default argument configs in argument w.r.t particular model
# ------------------------------
#

import argparse
import torch
from GraphModels import GraphConvWat, ChebNet, GATResMeanConv, m_GCN, GCN2, GAT, GIN#, GATResRewrittenConv, GATResLinearConv

def config_gatres_large(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\GATResMeanConvLarge_znorm_25b_128c_20235401_20230601_1754\best_GATResMeanConvLarge_znorm_25b_128c_20235401.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'

    return args, GATResMeanConv(name='GATRes_Large_znorm_25b_128c' if test_model_variant_name is None else test_model_variant_name, num_blocks=25, nc=128)

def config_gatres_small(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\gatres_znorm\best_GATResMeanConv_znorm_20235922.pth" 
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, GATResMeanConv(name='GATResMeanConv_small_znorm_15b_32c' if test_model_variant_name is None else test_model_variant_name, num_blocks=15, nc=32)

def config_mgcn(args: argparse.Namespace,  test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\m_GCNwosigmoid-n_aggr45-nhops1-nmlp2_mae_minmax_wosigmoid_20230615_1846\best_m_GCNwosigmoid-n_aggr45-nhops1-nmlp2_mae_minmax_wosigmoid.pth"
    args.criterion = 'mae'
    args.use_data_edge_attrs = 'diameter,length'
    args.norm_type = 'minmax'
    return args, m_GCN(name=test_model_variant_name,in_dim=1,out_dim=1,edge_dim=2,latent_dim=96,n_aggr=45,n_hops=1,num_layers=2,batch_size=args.batch_size,w_sigmoid=False)

def config_graphconvwat(args: argparse.Namespace,  test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path =r"experiments_logs\simple_test\GraphConvWat_minmax\best_GraphConvWat_minmax_120c_240k_Habrok_20233903.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'minmax'
    return args, GraphConvWat('GraphConvWat_minmax'  if test_model_variant_name is None else test_model_variant_name ,1,1)

def config_chebnet_graphconvwat_tuned(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\chebnet_znorm\best_chebnet_znorm_20235722.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, ChebNet( name='ChebNet_znorm'  if test_model_variant_name is None else test_model_variant_name, in_channels=1, out_channels=1, nc = 32)

def config_gat(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\GAT\best_GAT_10b_32c_2h_20231827.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, GAT(name='GAT_10b_32c_2h' if test_model_variant_name is None else test_model_variant_name,num_blocks = 10, nc = 32, in_channels=1, out_channels=1)

def config_gcn2(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\GCN2-64b-32nc\best_GCN2-64b-32nc_64b_32c_20232325.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, GCN2(name='GCN2-64b-32nc' if test_model_variant_name is None else test_model_variant_name,num_blocks = 64, nc = 32, in_channels=1, out_channels=1)

def config_gin(args: argparse.Namespace, test_model_variant_name=None)-> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path = r"experiments_logs\simple_test\GIN_bottleneck_15b_32c\best_GIN_bottleneck_b15_c32_15b_32c_20231413.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, GIN(name='GIN_bottleneck_b15_c32' if test_model_variant_name is None else test_model_variant_name, num_blocks=15, nc=32)


# def config_GATResRewrittenConv(args: argparse.Namespace, test_model_variant_name=None) -> tuple[argparse.Namespace, torch.nn.Module]:
#     args.model_path =  r".\experiments_logs\simple_test\GATResRewrittenConv_znorm_15b_32c\best_GATResRewrittenConv_znorm_15b_32c_20234517.pth"
#     args.criterion = 'mse'
#     args.use_data_edge_attrs = None
#     args.norm_type = 'znorm'
#     return args, GATResRewrittenConv(name='GATResRewrittenConv_znorm_15b_32c' if test_model_variant_name is None else test_model_variant_name, num_blocks=15, nc=32)

# def config_GATResLinearConv(args: argparse.Namespace, test_model_variant_name=None) -> tuple[argparse.Namespace, torch.nn.Module]:
#     args.model_path =r".\experiments_logs\simple_test\GATResLinearConv_znorm_15b_32c\best_GATResLinearConv_znorm_15b_32c_20235317.pth"
#     args.criterion = 'mse'
#     args.use_data_edge_attrs = None
#     args.norm_type = 'znorm'
#     return args, GATResLinearConv(name='GATResLinearConv_znorm_15b_32c' if test_model_variant_name is None else test_model_variant_name, num_blocks=15, nc=32)


def config_gatres_small_tough(args: argparse.Namespace, test_model_variant_name=None) -> tuple[argparse.Namespace, torch.nn.Module]:
    args.model_path =r"experiments_logs\simple_test\GATRes_small_tough_znorm_15b_32c\best_GATRes_small_tough_znorm_15b_32c_20233629.pth"
    args.criterion = 'mse'
    args.use_data_edge_attrs = None
    args.norm_type = 'znorm'
    return args, GATResMeanConv(name='GATResMeanConv_small_tough_znorm_15b_32c' if test_model_variant_name is None else test_model_variant_name, num_blocks=15, nc=32)


def select_model(args: argparse.Namespace, test_model_variant_name: str=None, reset_model_path: bool= False)  -> tuple[argparse.Namespace, torch.nn.Module]:
    """support selecting models using args.model_name

    Args:
        args (argparse.Namespace): default argument
        test_model_variant_name (str): set a name for the model
        reset_model_path (bool): remove default model path wrt model
    Returns:
        tuple[argparse.Namespace, torch.nn.Module]: model configs and model
    """
    if hasattr(args,'model'):
        model = args.model
    else:
        model = 'gatres_small'

    if hasattr(args,'model_path'):
        old_model_path = args.model_path

    print(f'before-args.model_path = {args.model_path}')
    assert model in ['gatres_small','gatres_large','gin','graphconvwat','chebnet','mgcn','gcn2','gat']
    if model == 'gatres_small':
        args,model= config_gatres_small(args,test_model_variant_name)
    elif model == 'gatres_large':
        args,model=  config_gatres_large(args,test_model_variant_name)
    elif model == 'gin':
        args,model=  config_gin(args,test_model_variant_name)
    elif model == 'graphconvwat':
        args,model=  config_graphconvwat(args,test_model_variant_name)
    elif model == 'chebnet':
        args,model=  config_chebnet_graphconvwat_tuned(args,test_model_variant_name)
    elif model == 'mgcn':
        args,model=  config_mgcn(args,test_model_variant_name)
    elif model == 'gcn2':
        args,model=  config_gcn2(args,test_model_variant_name)
    elif model == 'gat':
        args,model=  config_gat(args,test_model_variant_name)
    else:
        raise NotImplementedError(f'Unknown model! Got {model}!')
    
    if reset_model_path:
        args.model_path = old_model_path

    print(f'after-args.model_path = {args.model_path}')
    
    return args,model