#
# Created on Mon Sep 11 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Data loader
# ------------------------------
#


import sys
# setting path
sys.path.append('../signal-recovery')

import os
from torch_geometric.data import Dataset, Data
import zarr
from wntr.network import WaterNetworkModel
from wntr.sim.epanet import EpanetSimulator
import epynet
import torch_geometric.utils as pgu
import torch
import networkx as nx
import numpy as np
import pandas as pd
from epynet import epanet2
from copy import deepcopy
from utils.auxil import  scale, nx_to_pyg


from typing import Optional, Union,Any


def get_graph_template(new_graph: nx.Graph, edge_attrs:bool = False) -> 'torch_geometric.data.Data':
    graph_template = pgu.from_networkx(new_graph)
    if edge_attrs:
        graph_template.edge_attr = deepcopy(graph_template.weight)
        del(graph_template.weight)
        del(graph_template.type)
    del(graph_template.pos)
    del(graph_template.edge_type)

    return graph_template

def get_keep_list(wn: WaterNetworkModel, removal:str, root: Union[zarr.Group , zarr.Array], feature: str) -> list[str]:

    if removal == 'keep_list':
        if root and "ordered_name_list" in root.attrs:
            keep_list = root.attrs['ordered_name_list']
        elif root and "ordered_names_by_attr" in  root.attrs and feature in root.attrs['ordered_names_by_attr']:
            keep_list = root.attrs['ordered_names_by_attr'][feature]
        else:
            print(f'WARN! ordered_name_list or ordered_names_by_attr attribute is not found in zarr file. Use removal = \'keep_junction\'')
            keep_list = wn.junction_name_list
    elif removal == 'reservoir':
        keep_list = list(set(wn.node_name_list).difference(wn.reservoir_name_list)) if wn.reservoir_name_list else None
    elif removal == 'tank':
        keep_list = list(set(wn.node_name_list).difference(wn.tank_name_list))  if wn.tank_name_list else None
    elif removal == 'keep_junction':
        keep_list = wn.junction_name_list
    elif removal == 'keep_all':
        keep_list = None

    
    return keep_list

    

class WDNDataset(Dataset):
    def __init__(self,
                 input_paths,
                 zip_file_paths,
                 feature,
                 from_set,
                 num_records =None,
                 removal = 'keep_list',
                 do_scale=True,
                 mean=None,
                 std=None,
                 min=None,
                 max=None,
                 lazy_convert_pygdata=False,
                 edge_attrs=['diameter','length'],
                 edge_mean=None,
                 edge_std=None,
                 edge_min=None,
                 edge_max=None,
                 norm_type='znorm',
                 **kwargs):
        """The dataset class supports multiple datasets
       
        :param List<str> input_paths: list of inp files
        :param List<str> zip_file_paths: list of zip files
        :param str feature: supported features: pressure/ head/ flow
        :param str from_set: supported sets: train/ valid/ test
        :param int num_records: number of records, defaults to None means taken all
        :param str removal: supported removals: keep_list/ reservoir/ tank/ keep_junction, defaults to 'keep_list'
        :param bool do_scale: standardization flag, defaults to True
        :param mean: existing computed mean, set None to re-compute, defaults to None 
        :param std: existing computed std, set None to re-compute, defaults to None 
        :param min: existing computed min, set None to re-compute, defaults to None 
        :param max: existing computed max, set None to re-compute, defaults to None 
        :param bool lazy_convert_pygdata: if True, converting all into Pyg data; else, only convert to Pyg Data when called, defaults to True
        :param List<str> edge_attrs: list of edge attributes. Supported: diameter, length, valve_mask, None
        :param edge_mean: existing computed edge attributes mean, set None to re-compute, defaults to None 
        :param edge_std: existing computed edge attributes std, set None to re-compute, defaults to None 
        :param edge_min: existing computed edge attributes min, set None to re-compute, defaults to None 
        :param edge_max: existing computed edge attributes max, set None to re-compute, defaults to None 
        :param norm_type: normalization type supports: znorm/minmax/unused, default is znorm
        :raises KeyError: Key is not found in attrs of the zarr file
        """
        assert norm_type in ['znorm','minmax','unused']
        assert edge_attrs is None  or   set(edge_attrs).issubset( ['diameter', 'length', 'valve_mask'] ) 
        assert len(input_paths) == len(zip_file_paths)
        self._roots = []
        self._templates =[]
        self._lengths = []
        self.template_dict = {}
        
        #print(f'{self.__class__.__name__}-removal = {removal}')
        assert removal in ['keep_list','reservoir','tank','keep_junction','keep_all'], f'Removal only supports keep_list,reservoir,tank,keep_junction. Got {removal}'
        _arrays = []
        _keeplists = []
        for i, (input_path, zip_file_path) in enumerate(zip(input_paths, zip_file_paths)) :
            assert os.path.isfile(input_path) and (input_path[-4:] == '.inp' or input_path[-4:] == '.net'), f'{input_path} is not a INP/ NET file'
            
            graph_template, array, keep_list  = self.collect(input_path, zip_file_path, feature, edge_attrs, removal, from_set, num_records, **kwargs)
                
            #print(f'{self.__class__.__name__}-array shape = {array.shape}')
            self._templates.append(graph_template)
            self._lengths.append(array.shape[0]) 
            _arrays.append(array) 
            _keeplists.append(keep_list)
            
        self._keeplists = _keeplists
        self._arrays = _arrays
        self._ids = np.cumsum(self._lengths[:-1])
        self.cumsum_lengths = np.cumsum(self._lengths)
        self.feature = feature
        self.from_set=  from_set
        self.length  = sum(self._lengths)
        self.transform = None
        self.lazy_convert_pygdata = lazy_convert_pygdata
        self.norm_type = norm_type
        
        flatten_arr = np.concatenate([arr.flatten() for arr in self._arrays] )
        
        self.mean = np.mean(flatten_arr) if mean is None else mean
        self.std = np.std(flatten_arr) if std is None else std
        self.min = np.min(flatten_arr) if min is None else min
        self.max = np.max(flatten_arr) if max is None else max
        self.edge_mean = None 
        self.edge_std = None
        self.edge_min = None 
        self.edge_max = None 
        do_scale = do_scale if norm_type in ['znorm','minmax'] else False
        if do_scale:
            for i in range(len(self._arrays)):
                self._arrays[i] = scale(self._arrays[i], norm_type=norm_type, mean=self.mean, std=self.std, min=self.min, max=self.max)
            
            if edge_attrs:
                flatten_edge_arr = np.concatenate([template.edge_attr for template in self._templates],axis=0)
                self.edge_mean = np.mean(flatten_edge_arr,axis=0) if edge_mean is None else edge_mean
                self.edge_std = np.std(flatten_edge_arr,axis=0)  if edge_std is None else edge_std
                self.edge_min = np.min(flatten_edge_arr,axis=0) if edge_min is None else edge_min
                self.edge_max = np.max(flatten_edge_arr,axis=0)  if edge_max is None else edge_max
                
                for i in range(len(self._templates)):
                    self._templates[i].edge_attr = scale(self._templates[i].edge_attr, norm_type=norm_type, mean=self.edge_mean, std=self.edge_std, min=self.edge_min, max=self.edge_max)
           
                

        self._indices = range(self.length)
        self.num_arrays = len(self._arrays)

        if not self.lazy_convert_pygdata:
            tmp_array = []
            for arr_id in range(len(self._arrays)):
                for internal_id in range(len( self._arrays[arr_id])):
                    tmp_array.append(nx_to_pyg(data= self._arrays[arr_id][internal_id], graph = self._templates[arr_id]))
            
            self._arrays = tmp_array
    
        

    def size(self)->int:
        return len(self._arrays)

    def len(self) -> int:
        return self.length

    def get(self, idx):
        if  self.lazy_convert_pygdata:
            arr_id = -1
            for i,l in enumerate(self.cumsum_lengths):
                if idx < l:
                    arr_id = i
                    break
            if arr_id > 0:
                internal_id = idx- self.cumsum_lengths[arr_id-1] 
            else:
                internal_id = idx
            return nx_to_pyg(data= self._arrays[arr_id][internal_id], graph = self._templates[arr_id])
        else:
            return self._arrays[idx]
    
    def collect(self, input_path:str, zip_file_path:str, feature:str, edge_attrs:list, removal:str, from_set:str, num_records:int, **kwargs) -> tuple[Data, np.ndarray, list[str]]:
        assert os.path.isfile(zip_file_path) and zip_file_path[-4:] == '.zip', f'{zip_file_path} is not a zip file'
        assert from_set in ['train', 'valid', 'test'], f'from_set {from_set} is not supported'
        
        root = zarr.open(store=zip_file_path, mode='r')
        assert set([feature]).issubset(root.group_keys()), f'feature {feature} is unavailabel in zarr file {zip_file_path}'
        
        self._roots.append(root)
        wn =  WaterNetworkModel(input_path)
        
        if edge_attrs:
            link_weights = [wn.query_link_attribute(attribute=att) for att in edge_attrs]
            link_weights = pd.concat(link_weights,axis=1) 
            # outer joins can cause nan -> fill with 0
            link_weights = link_weights.fillna(0)
            link_weight_dict = link_weights.T.to_dict(orient='list')
            weight_shape = len(list(link_weight_dict.values())[0])
            weight_type=  link_weights.dtypes[0]
            for link_uid in wn.link_name_list:
                if link_uid not in link_weight_dict:
                    link_weight_dict[link_uid] = np.zeros(shape=weight_shape,dtype=weight_type)
            
            assert not list(set(link_weight_dict.keys()).difference(wn.link_name_list)) 
        else:
            link_weight_dict = None
        
        #graph = wn.to_graph(link_weight= link_weight_dict).to_undirected()

        graph = nx.Graph(wn.to_graph(link_weight= link_weight_dict)).to_undirected()
        
        keep_list = get_keep_list(wn=wn, removal=removal, root=root, feature=feature)
        
        
        array = np.array( root[feature][from_set])
        if num_records is not None:
            array = array[:num_records]
        
        if keep_list is not None:
            taken_indices = []
            for i, name in enumerate(wn.node_name_list):
                if name in keep_list:
                    taken_indices.append(i)
            
            array = np.take(array, taken_indices, axis=-1)

            assert array.shape[-1] >= len(keep_list) 
        
        new_graph = graph.subgraph(keep_list).copy() if keep_list is not None else graph
        

        graph_template = get_graph_template(new_graph=new_graph, edge_attrs=edge_attrs)


        
        return graph_template, array, keep_list if keep_list  else wn.node_name_list
       

class NoisyWDNDataset(WDNDataset):
    """NoisyWDNDataset allows to run an online testing using a simulation environment. We can inject noise into dynamic params using this class.

    Args:
        WDNDataset (_type_): _description_
    """

    def collect(self, input_path:str, zip_file_path:str, feature:str, edge_attrs:list, removal:str, from_set:str, num_records:int, **kwargs) -> tuple[zarr.Array, Data, np.ndarray, list[str]]:
        #addition params
        sim_engine = kwargs.pop('sim_engine','epynet')
        mean_dmd = kwargs.pop('mean_dmd',0.)
        std_dmd = kwargs.pop('sim_engine',0.)
        ################
        #root = zarr.open(store=zip_file_path, mode='r')
        #assert set([feature]).issubset(root.group_keys()), f'feature {feature} is unavailable in zarr file {zip_file_path}'
        
        wn =  WaterNetworkModel(input_path)
        
        if edge_attrs:
            link_weights = [wn.query_link_attribute(attribute=att) for att in edge_attrs]
            link_weights = pd.concat(link_weights,axis=1) 
            # outer joins can cause nan -> fill with 0
            link_weights = link_weights.fillna(0)
            link_weight_dict = link_weights.T.to_dict(orient='list')
            weight_shape = len(list(link_weight_dict.values())[0])
            weight_type=  link_weights.dtypes[0]
            for link_uid in wn.link_name_list:
                if link_uid not in link_weight_dict:
                    link_weight_dict[link_uid] = np.zeros(shape=weight_shape,dtype=weight_type)
            
            assert not list(set(link_weight_dict.keys()).difference(wn.link_name_list)) 
        else:
            link_weight_dict = None
        
        #graph = wn.to_graph(link_weight= link_weight_dict).to_undirected()
        graph = nx.Graph(wn.to_graph(link_weight= link_weight_dict)).to_undirected()

        keep_list = get_keep_list(wn=wn, removal=removal, root=None, feature=feature)

        new_graph = graph.subgraph(keep_list).copy() if keep_list is not None else graph
        
        graph_template = get_graph_template(new_graph=new_graph, edge_attrs=edge_attrs)

        self._templates.append(graph_template)


        if sim_engine == 'wntr':
            
            ###########################################################
            #ENSURE NO NEGATIVE VALUES
            for k, pattern in wn.patterns.items():
                pattern.multipliers =  np.abs(np.array(pattern.multipliers)).tolist()

            ############################################################

            #inject noise into demands
            if mean_dmd != 0. or std_dmd != 0:
                noise_vector = np.random.normal(loc=mean_dmd,scale=std_dmd,size=[len(wn.junction_name_list)])
                    
                for i,(_, junction) in enumerate(wn.junctions()):
                    tmp = junction.base_demand + noise_vector[i] * junction.base_demand 
                    if tmp >= 0:
                        junction.demand_timeseries_list[0].base_value = tmp
            
            #online simulation    
            sim = EpanetSimulator(wn)
            results =  sim.run_sim(version=2.2)
            
            if feature in ['pressure','head']:
                root = results.node
                
            else:
                root = results.link
            
            array = root[feature] 
        else:
            epynet_wn = epynet.Network(input_path)
            ###########################################################
            #ENSURE NO NEGATIVE VALUES
            for junc in epynet_wn.junctions:
                pattern_index = int(junc.get_property(epanet2.EN_PATTERN))
                if pattern_index >0:
                    p = junc.pattern
                    new_values = np.array(p.values)
                    new_values = np.abs(new_values)
                    p.values = new_values
            ############################################################
            #inject noise into demands
            if mean_dmd != 0. or std_dmd != 0:
                noise_vector = np.random.normal(loc=mean_dmd,scale=std_dmd,size=[len(wn.junction_name_list)])
                for i,junction in enumerate(epynet_wn.junctions):
                    tmp = junction.base_demand + noise_vector[i] * junction.base_demand 
                    if tmp >= 0:
                        junction.base_demand  = tmp

            #online simulation 
            epynet_wn.run()
            if feature == 'pressure':
                array = epynet_wn.nodes.pressure
            elif feature =='head':
                array = epynet_wn.nodes.head
            elif feature =='flow':
                array = epynet_wn.links.flow
            
        if keep_list is not None:
            array = array[keep_list]

        if num_records is not None:
            array = array[:num_records]
        array = array.to_numpy()

        return  graph_template, array, keep_list if keep_list  else wn.node_name_list
        

    def __init__(self,
                 input_paths,
                 feature,
                 num_records =None,
                 removal = 'keep_list',
                 do_scale=True,
                 mean=None,
                 std=None,
                 min=None,
                 max=None,
                 lazy_convert_pygdata=False,
                 edge_attrs=['diameter','length'],
                 edge_mean=None,
                 edge_std=None,
                 edge_min=None,
                 edge_max=None,
                 norm_type='znorm',
                 ###########additional###################
                 sim_engine='wntr',
                 #####uncertainty#########
                 mean_dmd=0.05,
                 std_dmd=0.2,
                 ):
        
        zip_file_paths = [None for _ in range(len(input_paths))]
        from_set = None

        super().__init__(input_paths=input_paths,
                        zip_file_paths=zip_file_paths,
                        feature=feature,
                        from_set=from_set,
                        num_records=num_records,
                        removal=removal,
                        do_scale=do_scale,
                        mean=mean,
                        std=std,
                        min=min,
                        max=max,
                        lazy_convert_pygdata=lazy_convert_pygdata,
                        edge_attrs=edge_attrs,
                        edge_mean=edge_mean,
                        edge_std=edge_std,
                        edge_min=edge_min,
                        edge_max=edge_max,
                        norm_type=norm_type,
                        sim_engine=sim_engine,
                        mean_dmd=mean_dmd,
                        std_dmd=std_dmd,)
        



def get_stacked_set(
    zip_file_path: str,
    input_path: str,
    train_mean: Any,
    train_std:Any,
    train_min:Any,
    train_max:Any,
    train_edge_mean:Any,
    train_edge_std:Any,
    train_edge_min:Any,
    train_edge_max:Any,
    norm_type:str,
    edge_attrs:list[str],
    feature: str='pressure',): 
  
    test_train_ds = WDNDataset(zip_file_paths=[zip_file_path],
                          input_paths=[input_path],
                          feature=feature,
                          from_set='train',
                          num_records=None,
                          removal='keep_list',
                          do_scale=True,
                          mean=train_mean,
                          std=train_std,
                          min=train_min,
                          max=train_max,
                          lazy_convert_pygdata=False,
                          edge_attrs=edge_attrs,
                          edge_mean=train_edge_mean,
                          edge_std=train_edge_std,
                          edge_min=train_edge_min,
                          edge_max=train_edge_max,
                          norm_type=norm_type,
                          )

    test_valid_ds = WDNDataset(zip_file_paths=[zip_file_path],
                            input_paths=[input_path],
                            feature=feature,
                            from_set='valid',
                            num_records=None,
                            removal='keep_list',
                            do_scale=True,
                            mean=train_mean,
                            std=train_std,
                            min=train_min,
                            max=train_max,
                            lazy_convert_pygdata=False,
                            edge_attrs=edge_attrs,
                            edge_mean=train_edge_mean,
                            edge_std=train_edge_std,
                            edge_min=train_edge_min,
                            edge_max=train_edge_max,
                            norm_type=norm_type,
                            )

    test_ds = WDNDataset(zip_file_paths=[zip_file_path],
                            input_paths=[input_path],
                            feature=feature,
                            from_set='test',
                            num_records=None,
                            removal='keep_list',
                            do_scale=True,
                            mean=train_mean,
                            std=train_std,
                            min=train_min,
                            max=train_max,
                            lazy_convert_pygdata=False,
                            edge_attrs=edge_attrs,
                            edge_mean=train_edge_mean,
                            edge_std=train_edge_std,
                            edge_min=train_edge_min,
                            edge_max=train_edge_max,
                            norm_type=norm_type,
                         )


    test_ds = test_ds +  test_train_ds + test_valid_ds

    return test_ds


def get_stacked_set2(
    zip_file_path: str,
    input_path: str,
    train_mean: Any,
    train_std:Any,
    train_min:Any,
    train_max:Any,
    train_edge_mean:Any,
    train_edge_std:Any,
    train_edge_min:Any,
    train_edge_max:Any,
    norm_type:str,
    edge_attrs:list[str],
    feature: str='pressure',
    num_tests: Optional[int] = None,
    removal : str ='keep_list'): 
    current_records = 0
    test_train_ds = WDNDataset(zip_file_paths=[zip_file_path],
                          input_paths=[input_path],
                          feature=feature,
                          from_set='train',
                          num_records=num_tests,
                          removal=removal,
                          do_scale=True,
                          mean=train_mean,
                          std=train_std,
                          min=train_min,
                          max=train_max,
                          lazy_convert_pygdata=False,
                          edge_attrs=edge_attrs,
                          edge_mean=train_edge_mean,
                          edge_std=train_edge_std,
                          edge_min=train_edge_min,
                          edge_max=train_edge_max,
                          norm_type=norm_type,
                          )
    current_records += len(test_train_ds)
    ret_test_ds =  test_train_ds 

    if num_tests is not None and current_records < num_tests:
        residual = num_tests - current_records 
        test_valid_ds = WDNDataset(zip_file_paths=[zip_file_path],
                                input_paths=[input_path],
                                feature=feature,
                                from_set='valid',
                                num_records=residual,
                                removal=removal,
                                do_scale=True,
                                mean=train_mean,
                                std=train_std,
                                min=train_min,
                                max=train_max,
                                lazy_convert_pygdata=False,
                                edge_attrs=edge_attrs,
                                edge_mean=train_edge_mean,
                                edge_std=train_edge_std,
                                edge_min=train_edge_min,
                                edge_max=train_edge_max,
                                norm_type=norm_type,
                                )
        current_records += len(test_valid_ds)
        ret_test_ds = ret_test_ds + test_valid_ds 

    if num_tests is not None and current_records < num_tests:
        residual = num_tests - current_records 
        test_ds = WDNDataset(zip_file_paths=[zip_file_path],
                                input_paths=[input_path],
                                feature=feature,
                                from_set='test',
                                num_records=residual,
                                removal=removal,
                                do_scale=True,
                                mean=train_mean,
                                std=train_std,
                                min=train_min,
                                max=train_max,
                                lazy_convert_pygdata=False,
                                edge_attrs=edge_attrs,
                                edge_mean=train_edge_mean,
                                edge_std=train_edge_std,
                                edge_min=train_edge_min,
                                edge_max=train_edge_max,
                                norm_type=norm_type,
                            )
        current_records += len(test_valid_ds)
        ret_test_ds = ret_test_ds + test_ds 

    if num_tests is not None:
        print(f'Test set length: actual: {current_records} / Expected : {num_tests}')

    return ret_test_ds