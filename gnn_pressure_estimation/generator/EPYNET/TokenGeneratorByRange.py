#
# Created on Wed Oct 18 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: generator helps generate random tokens
# ------------------------------
#


from configparser import ConfigParser
import numpy as np
from enum import Enum
import json
from tqdm import tqdm
from epynet import Network, ObjectCollection
from epynet import epanet2
from generator.EPYNET import epynet_utils as eutils
import os
import ray
import zarr
from collections import defaultdict
from numcodecs import Blosc
from sklearn.cluster import k_means
import networkx as nx
import wntr
from copy import deepcopy

EPSILON = 1e-12

class ParamEnum(str, Enum):
    RANDOM_TOKEN        = 'token'
    JUNC_DEMAND         = 'junc_demand'
    JUNC_ELEVATION      = 'junc_elevation'
    PUMP_STATUS         = 'pump_status'
    PUMP_SPEED          = 'pump_speed'
    PUMP_LENGTH         = 'pump_speed'
    TANK_LEVEL          = 'tank_level'
    TANK_ELEVATION      = 'tank_elevation'
    TANK_DIAMETER       = 'tank_diameter'
    VALVE_SETTING       = 'valve_setting'
    VALVE_STATUS        = 'valve_status'
    VALVE_DIAMETER      = 'valve_diameter'
    PIPE_ROUGHNESS      = 'pipe_roughness'
    PIPE_DIAMETER       = 'pipe_diameter'
    PIPE_LENGTH         = 'pipe_length'
    PIPE_MINORLOSS      = 'pipe_minor_loss'
    RESERVOIR_TOTALHEAD = 'reservoir_totalhead'


def update_object_by_json_string(json_string, object_dict, expected_shape):
    is_success = True
    overridden_values = None
    try:
        value_dict = json.loads(json_string)

        overridden_values = np.zeros(shape=expected_shape)
        
        uids = object_dict.uid
        
        in_uids_mask = np.isin(uids, value_dict, assume_unique=True, invert=True)

        tmp = np.array( list(value_dict.values()) ).T
        overridden_values[:,in_uids_mask] = tmp 
        if in_uids_mask.shape[0] != overridden_values.shape[1]:
            print(f'WARNING! in mask shape is not equal to expected shape! in mask shape = {in_uids_mask.shape}, expected shape = {expected_shape.shape}!')
            print(f'Missing values will be replaced by zeros!')
    except Exception as e:
        print(f'Error in update_demand_json - Error: {e}')
        is_success = False
        overridden_values = None

    return overridden_values, is_success

def compute_contineous_values_by_range( tokens, ratios, ori_vals=None, **kwargs):
    range_lo, range_hi = ratios[0], ratios[1]
    new_values = range_lo + tokens * (range_hi - range_lo)
    return new_values

def compute_boolean_values(tokens, ratios, **kwargs):
    open_prob = ratios[0]
    new_values = np.less(tokens, open_prob).astype(tokens.dtype)
    return new_values

def compute_contineous_values_by_ratio(ori_vals, tokens, ratios, **kwargs):
    param_minmax= [0, np.max(ori_vals)]
    ratio_lo, ratio_hi = ratios[0], ratios[1]
    new_values = ori_vals + np.sign(tokens) * (ratio_lo + (np.abs(tokens) * (ratio_hi - ratio_lo))) * ori_vals
    new_values = np.clip(new_values, param_minmax[0], param_minmax[1])
    return new_values        

def compute_contineous_diameter_by_ratio(ori_vals, tokens, ratios, **kwargs):
    param_minmax= [np.min(ori_vals), np.max(ori_vals)]
    ratio_lo, ratio_hi = ratios[0], ratios[1]
    new_values = ori_vals + np.sign(tokens) * (ratio_lo + (np.abs(tokens) * (ratio_hi - ratio_lo))) * ori_vals
    #new_values_mask = (new_values <= 0) #np.clip(new_values, param_minmax[0], param_minmax[1])
    new_values = np.where(new_values<=param_minmax[0], ori_vals, new_values)
    return new_values            

def compute_contineous_values_by_ran_cluster(ori_vals, tokens, ratios, **kwargs):
    """Require kwargs : 
        use_existing_clusters (bool): flag,\n
        num_clusters_lo (int): lowest n_clusters,\n
        num_clusters_hi (int): highest n_clusters,\n
    Optional kwargs:
        kmean_params (np.ndrray): nodal coordinates or link coordinates or any feature vectors, only used if use_existing_clusters = False\n
        sigma (float): standard deviation from local centroid values. If None, take std from ori_vals\n
        cluster_num_clusters (int): a specific number of cluster, only used if use_existing_clusters = True\n
        cluster_labels (np.ndrray): a specific labels assigned to elements, only used if use_existing_clusters = True\n
    Args:
        ori_vals (list): old/ original values
        tokens (np.ndrray): random tokens
        ratios (list): range or ratio list 

    Returns:
        np.ndrray: new updated values
    """
    
    use_existing_clusters = kwargs['use_existing_clusters']
    num_clusters_lo, num_clusters_hi = kwargs['num_clusters_lo'],kwargs['num_clusters_hi']
    chunk_size = tokens.shape[0]
    
    range_lo, range_hi = ratios[0], ratios[1]
    


    num_elements =  len(ori_vals)
    if not use_existing_clusters:
            
        kmean_params = kwargs['kmean_params']
        num_clusters = num_clusters_lo  + np.random.random([chunk_size,1]) * (num_clusters_hi - num_clusters_lo)
        if num_clusters_hi < num_elements:
            chunk_labels=[]
            for c in range(chunk_size):
                centroids, labels, _ = k_means(kmean_params,n_clusters=int(num_clusters[c]),n_init='auto') 
                chunk_labels.append(labels)
            labels = np.array(chunk_labels).reshape([chunk_size,-1])
        else:
            labels = np.arange(num_elements).reshape([1,-1]).repeat(chunk_size,axis=0)

    else:
        num_clusters    = kwargs['cluster_num_clusters']
        labels          = kwargs['cluster_labels']
    


    if num_clusters_hi < num_elements:
        local_tokens = range_lo + np.random.random(size=[chunk_size, num_clusters_hi]) * (range_hi - range_lo)
    else:
        local_tokens = range_lo + np.random.random(size=[chunk_size, num_elements]) * (range_hi - range_lo)

    sign = np.where(np.random.random(size=tokens.shape) >= 0.5, 1.0, -1.0) 

    cluster_vals = np.take_along_axis(local_tokens, labels, axis=1) 
    
    if 'sigma' in kwargs:
        sigma = kwargs['sigma'] 
    else:
        sigma = np.std(ori_vals.flatten())
        #sigma = abs(ori_vals - cluster_vals)

                    
    new_values = cluster_vals + sign * tokens * sigma
    
    new_values = np.clip(new_values, a_min= range_lo, a_max= range_hi)
    return new_values 
      
def get_node_coordinates(wn_g: nx.Graph, wn: Network, do_normalize: bool=True) -> dict:
    pos_dict = nx.get_node_attributes(wn_g, "pos")

    node_coords = [pos_dict[node.uid] for node in wn.nodes if node.uid in pos_dict]
    
    assert len(wn.nodes) == len(node_coords)
    node_coords = np.array(node_coords)
    if do_normalize:
        node_coords = node_coords / (np.linalg.norm(node_coords) + EPSILON)
    
    return dict(zip(wn.nodes.uid, node_coords.tolist()))

def get_link_coordinates(wn_g: nx.Graph, wn: Network, do_normalize: bool=True) -> dict:
    pos_dict = nx.get_node_attributes(wn_g, "pos")

    link_coords = []
    for link in wn.links:
        upstream_node = link.upstream_node.uid
        downstream_node = link.upstream_node.uid
        if upstream_node in pos_dict and downstream_node in pos_dict:
            xy = pos_dict[upstream_node]
            zt = pos_dict[downstream_node]
            link_coords.append([xy[0],xy[1],zt[0],zt[1]])

    assert len(wn.links) == len(link_coords)
    link_coords = np.array(link_coords)
    if do_normalize:
        link_coords = link_coords / (np.linalg.norm(link_coords) + EPSILON)
    
    return dict(zip(wn.links.uid, link_coords.tolist()))

def generate_params(
                    tokens,
                    ratios,
                    target_object_collection,
                    get_original_values_fn,
                    update_formula_fn,
                    component_key,
                    update_json=None,
                    **kwargs,
                    ):
    
    new_values = None
    if update_json is not None:
        new_values,_ = update_object_by_json_string(json_string= update_json,
                                    object_dict= target_object_collection,
                                    expected_shape=tokens.shape)

    if new_values is None:
        if kwargs is not None and 'coords' in kwargs:
            coord_dict = kwargs['coords']
            kwargs['kmean_params'] = [coord_dict[obj.uid] for obj in target_object_collection]

        
        

        #ori_vals = da.from_array(list(map(get_original_values_fn,target_object_collection)))
        # new_demands = ori_dmds + sign(tokens) *  (dmd_lo + abs(tokens) * (dmd_hi - dmd_lo) )
        #tokens = da.from_array(tokens,chunks=self.num_chunks)
        ori_vals = np.array(list(map(get_original_values_fn,target_object_collection)))
        if sum(ratios) == 0.: 
            new_values = ori_vals
        else:
            new_values = update_formula_fn(tokens=tokens, ratios=ratios, ori_vals=ori_vals, **kwargs)

    return new_values

@ray.remote
def ray_batch_update(chunk_size, num_features, featlen_dict, args):
    return batch_update(chunk_size, num_features, featlen_dict, args)

def batch_update(chunk_size, num_features, featlen_dict, args):
    tokens = np.random.uniform(
        low=0.0,#-1.0,
        high=1.0,
        size= (chunk_size, num_features)
    )

    config = ConfigParser()
    config.read(args.config)
    config_keys = dict(config.items()).keys()

    wn_inp_path = config.get('general','wn_inp_path')
    wn = Network(wn_inp_path)
    wn_g = wntr.network.WaterNetworkModel(wn_inp_path).get_graph()

    ragged_tokens = eutils.RaggedArrayDict.from_keylen_and_stackedarray(featlen_dict, tokens)

    node_coord_dict = get_node_coordinates(wn_g, wn, do_normalize=True)
    link_coord_dict = get_link_coordinates(wn_g, wn, do_normalize=True)
    
    update_formula_args = {
        'use_existing_clusters' : False,
        'num_clusters_lo' : 4,
        'num_clusters_hi' : 50,
        'sigma': 1.0,
    }
    node_update_formula_args = deepcopy(update_formula_args)
    link_update_formula_args = deepcopy(update_formula_args)
    node_update_formula_args['coords'] = node_coord_dict
    link_update_formula_args['coords'] = link_coord_dict


    new_tokens = defaultdict()
    if 'junction' in config_keys:
        if args.gen_demand:
            def get_origin_dmd(junc):
                return junc.basedemand* junc.pattern.values[0] if eutils.ENhasppatern(junc) else 0.
            new_tokens[ParamEnum.JUNC_DEMAND] = generate_params(
                    tokens=ragged_tokens[ParamEnum.JUNC_DEMAND],
                    ratios=[
                        config.getfloat('junction','demand_lo'),
                        config.getfloat('junction','demand_hi')
                    ],
                    target_object_collection=wn.junctions,
                    get_original_values_fn=get_origin_dmd,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.JUNC_DEMAND,
                    update_json=args.update_demand_json,
                    **node_update_formula_args
                )

        if args.gen_elevation:
            new_tokens[ParamEnum.JUNC_ELEVATION] = generate_params(
                    tokens=ragged_tokens[ParamEnum.JUNC_ELEVATION],
                    ratios=[
                        config.getfloat('junction','ele_lo'),
                        config.getfloat('junction','ele_hi')
                    ],
                    target_object_collection=wn.junctions,
                    get_original_values_fn=lambda junc: junc.elevation,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.JUNC_ELEVATION,
                    update_json=args.update_elevation_json,
                    **node_update_formula_args
                )
            

    if 'pump' in config_keys:
        if args.gen_pump_init_status:
            new_tokens[ParamEnum.PUMP_STATUS] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PUMP_STATUS],
                    ratios=[
                        config.getfloat('pump','open_prob'),
                    ],
                    target_object_collection=wn.pumps,
                    get_original_values_fn=lambda pump: pump.initstatus,
                    update_formula_fn=compute_boolean_values,
                    component_key= ParamEnum.PUMP_STATUS,
                    update_json=args.update_pump_init_status_json,
                )
           
        if args.gen_pump_speed:
            new_tokens[ParamEnum.PUMP_SPEED] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PUMP_SPEED],
                    ratios=[
                        config.getfloat('pump','speed_lo'),
                        config.getfloat('pump','speed_hi'),
                    ],
                    target_object_collection=wn.pumps,
                    get_original_values_fn=lambda pump: pump.speed,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PUMP_SPEED,
                    update_json=args.update_pump_speed_json,
                    **link_update_formula_args,
                )
            
        if args.gen_pump_length:
            new_tokens[ParamEnum.PUMP_LENGTH] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PUMP_LENGTH],
                    ratios=[
                        config.getfloat('pump','length_lo'),
                        config.getfloat('pump','length_hi'),
                    ],
                    target_object_collection=wn.pumps,
                    get_original_values_fn=lambda pump: pump.length,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PUMP_LENGTH,
                    update_json=args.update_pump_length_json,
                    **link_update_formula_args,
                )
           

    if 'tank' in config_keys:
        if args.gen_tank_level:
            new_tokens[ParamEnum.TANK_LEVEL] = generate_params(
                    tokens=ragged_tokens[ParamEnum.TANK_LEVEL],
                    ratios=[
                        config.getfloat('tank','level_lo'),
                        config.getfloat('tank','level_hi'),
                    ],
                    target_object_collection=wn.tanks,
                    get_original_values_fn=lambda tank: tank.tanklevel,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.TANK_LEVEL,
                    update_json=args.update_tank_level_json,
                    **node_update_formula_args,
                )
             
        
        if args.gen_tank_elevation:
            new_tokens[ParamEnum.TANK_ELEVATION] = generate_params(
                    tokens=ragged_tokens[ParamEnum.TANK_ELEVATION],
                    ratios=[
                        config.getfloat('tank','ele_lo'),
                        config.getfloat('tank','ele_hi'),
                    ],
                    target_object_collection=wn.tanks,
                    get_original_values_fn=lambda tank: tank.elevation,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.TANK_ELEVATION,
                    update_json=args.update_tank_elevation_json,
                    **node_update_formula_args,
                )
             
        
        if args.gen_tank_diameter:
            new_tokens[ParamEnum.TANK_DIAMETER] = generate_params(
                    tokens=ragged_tokens[ParamEnum.TANK_DIAMETER], 
                    ratios=[
                        config.getfloat('tank','dia_lo'),
                        config.getfloat('tank','dia_hi'),
                    ],
                    target_object_collection=wn.tanks,
                    get_original_values_fn=lambda tank: tank.diameter,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.TANK_DIAMETER,
                    update_json=args.update_tank_diameter_json,
                    **node_update_formula_args,
                )
             

    if 'valve' in config_keys:
        if args.gen_valve_setting:
            '''
            ratio_los = []
            ratio_his = []
            for v in wn.valves:
                key= v.valve_type.lower()
                ratio_lo,ratio_hi = config.getfloat('valve',f'{key}_ratio_lo'),config.getfloat('valve',f'{key}_ratio_hi')
                ratio_los.append(ratio_lo)
                ratio_his.append(ratio_hi)
            ratios = np.stack([ratio_los,ratio_his],axis=0)
            '''
            valve_type_ratio_dict = {}
            valve_type_uid_dict = {}
            for v in wn.valves:
                if v.valve_type not in valve_type_ratio_dict:
                    key = v.valve_type.lower()
                    ratio_lo, ratio_hi = config.getfloat('valve',f'setting_{key}_lo'),config.getfloat('valve',f'setting_{key}_hi')
                    valve_type_ratio_dict[v.valve_type] = ratio_lo, ratio_hi
                    valve_type_uid_dict[v.valve_type] = []
                valve_type_uid_dict[v.valve_type].append(v.uid)
            overridden_values = np.zeros(shape = [chunk_size, len(wn.valves)])

            for valve_type in valve_type_ratio_dict:
                ratios = valve_type_ratio_dict[valve_type]
                uids = valve_type_uid_dict[valve_type]
                target_object_collection =  ObjectCollection({k: wn.valves[k] for k in uids if k in wn.valves})
                #print(f'len target = {len(target_object_collection)} | len all = {len(wn.valves)}')
                in_uids_mask = np.isin(list(wn.valves.keys()), uids)
                #print(f'min in_uids_mask = {np.min(in_uids_mask)}')
                valve_type_new_tokens= generate_params(
                    tokens=ragged_tokens[ParamEnum.VALVE_SETTING][:,in_uids_mask], 
                    ratios=ratios,
                    target_object_collection= target_object_collection,
                    get_original_values_fn=lambda v: v.setting,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.VALVE_SETTING,
                    update_json=args.update_valve_setting_json,
                    **link_update_formula_args
                    )
                
                overridden_values[:,in_uids_mask] = valve_type_new_tokens 
                  
            new_tokens[ParamEnum.VALVE_SETTING] = overridden_values
        
        if args.gen_valve_init_status:
            new_tokens[ParamEnum.VALVE_STATUS] = generate_params(
                    tokens=ragged_tokens[ParamEnum.VALVE_STATUS], 
                    ratios=[
                        config.getfloat('valve','open_prob'),
                    ],
                    target_object_collection=wn.valves,
                    get_original_values_fn=lambda v: v.initstatus,
                    update_formula_fn=compute_boolean_values,
                    component_key= ParamEnum.VALVE_STATUS,
                    update_json=args.update_valve_init_status_json,
                )
             

        if args.gen_valve_diameter:
            new_tokens[ParamEnum.VALVE_DIAMETER] = generate_params(
                    tokens=ragged_tokens[ParamEnum.VALVE_DIAMETER], 
                    ratios=[
                        config.getfloat('valve','dia_lo'),
                        config.getfloat('valve','dia_hi'),
                    ],
                    target_object_collection=wn.valves,
                    get_original_values_fn=lambda v: v.diameter,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.VALVE_DIAMETER,
                    update_json=args.update_valve_diameter_json,
                    **link_update_formula_args
                )
            
    if 'pipe' in config_keys:
        if args.gen_roughness:
            new_tokens[ParamEnum.PIPE_ROUGHNESS] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PIPE_ROUGHNESS], 
                    ratios=[
                        config.getfloat('pipe','roughness_lo'),
                        config.getfloat('pipe','roughness_hi'),
                    ],
                    target_object_collection=wn.pipes,
                    get_original_values_fn=lambda p: p.roughness,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PIPE_ROUGHNESS,
                    update_json=args.update_pipe_roughness_json,
                    **link_update_formula_args,
                )
             
        if args.gen_diameter:
            new_tokens[ParamEnum.PIPE_DIAMETER] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PIPE_DIAMETER],
                    ratios=[
                        config.getfloat('pipe','diameter_lo'),
                        config.getfloat('pipe','diameter_hi'),
                    ],
                    target_object_collection=wn.pipes,
                    get_original_values_fn=lambda p: p.diameter,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PIPE_DIAMETER,
                    update_json=args.update_pipe_diameter_json,
                    **link_update_formula_args,
                )
             
        if args.gen_length:
            new_tokens[ParamEnum.PIPE_LENGTH] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PIPE_LENGTH], 
                    ratios=[
                        config.getfloat('pipe','length_lo'),
                        config.getfloat('pipe','length_hi'),
                    ],
                    target_object_collection=wn.pipes,
                    get_original_values_fn=lambda p: p.length,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PIPE_LENGTH,
                    update_json=args.update_pipe_length_json,
                    **link_update_formula_args
                )
             
        if args.gen_minorloss:
            new_tokens[ParamEnum.PIPE_MINORLOSS] = generate_params(
                    tokens=ragged_tokens[ParamEnum.PIPE_MINORLOSS], 
                    ratios=[
                        config.getfloat('pipe','minorloss_lo'),
                        config.getfloat('pipe','minorloss_hi'),
                    ],
                    target_object_collection=wn.pipes,
                    get_original_values_fn=lambda p: p.minorloss,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.PIPE_MINORLOSS,
                    update_json=args.update_pipe_minorloss_json,
                    **link_update_formula_args
                )
            
    if 'reservoir' in config_keys:
        if args.gen_res_total_head:
            def get_original_res_head(res):
                base_head = res.elevation
                try:
                    p_index = res.get_object_value(epanet2.EN_PATTERN)
                    head = wn.ep.ENgetpatternvalue(int(p_index),1) # get first value only
                except epanet2.ENtoolkitError:
                    head = 1.

                return base_head * head

            new_tokens[ParamEnum.RESERVOIR_TOTALHEAD] = generate_params(
                    tokens=ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD], 
                    ratios=[
                        config.getfloat('reservoir','head_lo'),
                        config.getfloat('reservoir','head_hi'),
                    ],
                    target_object_collection=wn.reservoirs,
                    get_original_values_fn=get_original_res_head,
                    update_formula_fn=compute_contineous_values_by_range,
                    component_key= ParamEnum.RESERVOIR_TOTALHEAD,
                    update_json=args.update_res_total_head_json,
                    **node_update_formula_args
                )

    #ensure the order
    concated_arrays = [new_tokens[k] for k in featlen_dict.keys()]
    return np.concatenate(concated_arrays,axis=-1)
    
class RayTokenGenerator():
    def __init__(self, store,  num_scenes, featlen_dict, num_chunks):
        self.store          = store
        self.num_scenes     = num_scenes
        self.featlen_dict   = featlen_dict
        self.num_chunks     = num_chunks
        self.num_features   = sum(self.featlen_dict.values())
    

    
    def update(self, args):
        chunk_size = args.batch_size
        num_chunks = self.num_scenes // chunk_size
        progressbar = tqdm(total=num_chunks)
        start_index = 0
        worker_ids = [ray_batch_update.remote(chunk_size, self.num_features, self.featlen_dict, args) for _ in range(num_chunks)] 
        done_ids,undone_ids = ray.wait(worker_ids)
        num_out_features = sum(self.featlen_dict.values())

        while done_ids:
            result  = ray.get(done_ids[0])
            if start_index == 0:
                z_tokens = zarr.empty([self.num_scenes, num_out_features],
                                        chunks=(chunk_size, num_out_features),
                                        dtype='f8',
                                        store= os.path.join(self.store.path,ParamEnum.RANDOM_TOKEN),
                                        overwrite=True,
                                        synchronizer=zarr.ThreadSynchronizer(),
                                        compressor=Blosc(cname='lz4',clevel=5)
                                        )
            z_tokens[start_index: start_index + chunk_size] = result
            start_index += chunk_size
            del result
            done_ids, undone_ids = ray.wait(undone_ids)
            progressbar.update(1)
        progressbar.close()
        ray.shutdown()
        print('OK')
        
    def sequential_update(self,args):
        chunk_size = args.batch_size
        num_chunks = self.num_scenes // chunk_size
        num_out_features = sum(self.featlen_dict.values())
        start_index = 0
        progressbar = tqdm(total=num_chunks)
        for _ in range(num_chunks):
            result = batch_update(chunk_size, self.num_features, self.featlen_dict, args)
            if start_index == 0:
                z_tokens = zarr.empty([self.num_scenes, num_out_features],
                                        chunks=(chunk_size, num_out_features),
                                        dtype='f8',
                                        store= os.path.join(self.store.path,ParamEnum.RANDOM_TOKEN),
                                        overwrite=True,
                                        synchronizer=zarr.ThreadSynchronizer(),
                                        compressor=Blosc(cname='lz4',clevel=5)
                                        )
            z_tokens[start_index: start_index + chunk_size] = result
            start_index += chunk_size
            del result
            progressbar.update(1)

        progressbar.close()
        print('OK') 
    
    def load_computed_params(self):
        param = zarr.open_array(
            store=os.path.join(self.store.path,ParamEnum.RANDOM_TOKEN),
            mode='r'
            )
        return param
