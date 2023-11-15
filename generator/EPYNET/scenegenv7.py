#
# Created on Wed Oct 18 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: This file supports to generate dataset based on a config file
# Version: 7.1
# Note: You may need to increase/ decrease values in the config to get stable states
# Tip: Start with gen_demand=True, set off other gen_* flags
# Tip: set debug=True for more details
# Tip: don't change static hydraulic values
# ------------------------------
#


import sys
# setting path
sys.path.append('../signal-recovery')
from TokenGeneratorByRange import *
import argparse
import os
import shutil
import datetime
import zarr
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from Executorv7 import *
import ray
import pandas as pd
from ray.exceptions import RayError

program_start = time()
parser = argparse.ArgumentParser()
#main config
parser.add_argument('--config', default = r"G:\Other computers\My Laptop\PhD\Codebase\signal-recovery\configs\v7.1\ctown_7v1__EPYNET_config.ini" , type = str, help = 'configuration path')

#initial valve/pipe states
parser.add_argument('--init_valve_state',default = 1, type=int, help= 'init status = CheckedValve(3) Active(2) Open(1) Closed(0) KeepInitStatus(None)')
parser.add_argument('--init_pipe_state',default = None, type=int, help= 'init status = CheckedValve(3) Active(2) Open(1) Closed(0) KeepInitStatus(None)')

#removal flags
parser.add_argument('--remove_pattern', default = True, type = bool, help = 'flag indicates to remove any pattern in input file')
parser.add_argument('--remove_control', default = False, type = bool, help = 'flag indicates to remove any control in input file')
parser.add_argument('--remove_rule', default = False, type = bool, help = 'flag indicates to remove any rule in input file! Note EPANET authors confuse control and rule')

#demands settings
parser.add_argument('--gen_demand', default = True, type = bool, help = 'If true, replacing nonzero base demand to 1.0 | ELSE, replacing ALL base demands to 1.0. Default is False')
parser.add_argument('--replace_nonzero_basedmd', default = False, type = bool, help = 'If true, replacing nonzero base demand to 1.0 | ELSE, replacing ALL base demands to 1.0. Default is False')
parser.add_argument('--update_demand_json', default=None, type=str, help='JSON string. Overriding demand values (Note: demand = base_dmd * multipliers) according to the JSON file. Set None if unsed. Default is None')

#elevation settings
parser.add_argument('--gen_elevation', default = False, type = bool, help = 'flag indicates to change the nodal elevation')
parser.add_argument('--ele_kmean_init', default = 'k-means++', type = str, help = 'Initialization of K-mean for elevation cluster = k-means++ | random')
parser.add_argument('--update_elevation_method', default = 'ran_cluster', type = str, help = 'update elevation if gen_elevation is True, options: ran_cluster | ran_local | ran | idw_dist | idw_ran')
parser.add_argument('--ele_std',default=1.,type=float,help='the std apart from the elevation of local region')
parser.add_argument('--update_elevation_json', default = None, type = str, help = 'JSON string. Overriding elevation values according to the JSON file. Set None if unsed. Default is None')

#pipe settings
parser.add_argument('--gen_roughness', default = True, type = bool, help = 'flag indicates to change the pipe roughness')
parser.add_argument('--gen_diameter', default = False, type = bool, help = 'flag indicates to change the pipe diameter')
parser.add_argument('--dia_kmean_init', default = 'k-means++', type = str, help = '(UNSED)Initialization of K-mean for diameter cluster = k-means++ | random')
parser.add_argument('--gen_length', default = False, type = bool, help = 'flag indicates to change the pipe roughness')
parser.add_argument('--gen_minorloss', default = False, type = bool, help = 'flag indicates to change the pipe diameter')
parser.add_argument('--update_pipe_roughness_json', default = None, type = str, help = 'JSON string. Overriding pipe roughness values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_diameter_json', default = None, type = str, help = 'JSON string. Overriding pipe_diameter values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_length_json', default = None, type = str, help = 'JSON string. Overriding pipe length values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pipe_minorloss_json', default = None, type = str, help = 'JSON string. Overriding pipe minorloss values according to the JSON file. Set None if unsed. Default is None')

#valve settings
parser.add_argument('--gen_valve_init_status', default = True, type = bool, help = 'flag indicates to change the valve init status')
parser.add_argument('--gen_valve_setting', default = True, type = bool, help = 'flag indicates to change the valve settings')
parser.add_argument('--gen_valve_diameter', default = False, type = bool, help = 'flag indicates to change the valve diameter')
parser.add_argument('--update_valve_init_status_json', default = None, type = str, help = 'JSON string. Overriding valve status values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_valve_setting_json', default = None, type = str, help = 'JSON string. Overriding valve setting values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_valve_diameter_json', default = None, type = str, help = 'JSON string. Overriding valve diameter values according to the JSON file. Set None if unsed. Default is None')

#pump settings
parser.add_argument('--gen_pump_init_status', default = False, type = bool, help = 'flag indicates to change the pump init status')
parser.add_argument('--gen_pump_speed', default = True, type = bool, help = 'flag indicates to change the pump speed')
parser.add_argument('--gen_pump_length', default = False, type = bool, help = 'flag indicates to change the pump length')
parser.add_argument('--update_pump_init_status_json', default = None, type = str, help = 'JSON string. Overriding pump init status values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pump_speed_json', default = None, type = str, help = 'JSON string. Overriding pump speed values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_pump_length_json', default = None, type = str, help = 'JSON string. Overriding pump length values according to the JSON file. Set None if unsed. Default is None')


#tank settings 
parser.add_argument('--gen_tank_level', default = True, type = bool, help = 'flag indicates to change the tank level')
parser.add_argument('--gen_tank_elevation', default = False, type = bool, help = 'flag indicates to change the tank elevation')
parser.add_argument('--gen_tank_diameter', default = False, type = bool, help = 'flag indicates to change the tank diameter')
parser.add_argument('--update_tank_level_json', default = None, type = str, help = 'JSON string. Overriding tank level values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_tank_elevation_json', default = None, type = str, help = 'JSON string. Overriding tank elevation values according to the JSON file. Set None if unsed. Default is None')
parser.add_argument('--update_tank_diameter_json', default = None, type = str, help = 'JSON string. Overriding tank diameter values according to the JSON file. Set None if unsed. Default is None')

#reservoir settings
parser.add_argument('--gen_res_total_head', default = True, type = bool, help = 'flag indicates to change the total head of reservoir')
parser.add_argument('--skip_resevoir_result', default = True, type = bool, help = 'flag indicates to skip the resevoirs result after simulation')
parser.add_argument('--update_totalhead_method', default = None, type = str, help = 'update total head of reservor only if gen_res_total_head is True, options: add_max_elevation | None')
parser.add_argument('--update_res_total_head_json', default = None, type = str, help = 'JSON string. Overriding reservoir totalHead values according to the JSON file. Set None if unsed. Default is None')

#settings
parser.add_argument('--debug', default = True, type = bool, help = 'flag allows to print some useful measurements')
parser.add_argument('--allow_error', default = False, type = bool, help = 'flag allows to bypass error scenarios (useful for debug ), defaults to False')
parser.add_argument('--convert_results_by_flow_unit', default= 'CMH', type=str, help='CMH Convert all results according to the SI flow units that includes LPS, LPM, MLD, CMH, CMD. Set None to keep original unit' )
parser.add_argument('--change_dmd_by_junc_indices_path', default=None, type = str, help = 'selected_sensitivity_by_cv_2023-02-13.pkl|Path to the indices of junctions used to change demand only. The one which is not in this list has the minimum value. Setting None if not used')#

#conditions
parser.add_argument('--accept_warning_code', default= False, type = bool, help = 'flag allows to accept warning codes (0 < code < 6)')
parser.add_argument('--pressure_lowerbound', default = None, type = float, help = 'threshold value to filter small pressure values - 5mH2O - 7.1 psi. Set None if unused')
parser.add_argument('--pressure_upperbound', default = None, type = float, help = 'threshold value to filter high pressure values - 100mH2O - 142.23 psi. Set None if unused')
parser.add_argument('--flowrate_threshold', default = None, type = float, help = 'threshold value to filter valid flowarate values')
parser.add_argument('--mean_cv_threshold', default = None, type = float, help = '5.0 threshold value to filter high cv')
parser.add_argument('--neighbor_std_threshold', default = None, type = float, help = 'threshold value to filter high neighbor std')

#general info
parser.add_argument('--batch_size', default = 5, type = int, help = 'batch size')
parser.add_argument('--executors', default = 2, type = int, help = 'number of executors')
parser.add_argument('--att', default = 'pressure,head', type = str, help = 'list of simulation attributes you want to extract. Supported attributes: demand, pressure, head, flow, velocity')
parser.add_argument('--train_ratio',default=0.6,type=float,help='the ratio of training scenarios and total')
parser.add_argument('--valid_ratio',default=0.2,type=float,help='the ratio of validation scenarios and total')
parser.add_argument('--is_single_thread',default=False,type=bool,help='run the generation with only a single thread for debugging only. Defaults is False')



args = parser.parse_args([])

config = ConfigParser()
config.read(args.config)
config_keys = dict(config.items()).keys()

wn_inp_path = config.get('general','wn_inp_path')
storage_dir = config.get('general','storage_dir')

zarr_storage_dir = os.path.join(storage_dir,'zarrays')
random_array_dir = os.path.basename(wn_inp_path)[:-4] + '_random_array_'+ datetime.datetime.now().strftime('%m_%d_%Y_%H_%M') #get input name
random_array_dir = os.path.join(storage_dir,random_array_dir)
os.makedirs(storage_dir,exist_ok=True)
os.chdir(storage_dir)
shutil.rmtree(path=storage_dir,ignore_errors=True)
os.makedirs(zarr_storage_dir,exist_ok=False)

saved_path              = storage_dir 
num_scenarios           = config.getint('general','num_scenarios') #1000
backup_num_scenarios    = num_scenarios * 10
batch_size              = args.batch_size 
num_executors           = args.executors
expected_attributes     = args.att.strip().split(',') #['pressure','head','flowrate','velocity'] 
train_ratio             = args.train_ratio
valid_ratio             = args.valid_ratio
num_batches             = backup_num_scenarios // batch_size
num_chunks              = backup_num_scenarios // batch_size
support_node_attr_keys  = ['head','pressure','demand']
support_link_attr_keys  = ['flow','velocity']#'flowrate',
support_keys = list(set(support_node_attr_keys).union(support_link_attr_keys))
for a in expected_attributes:
    if a not in support_keys:
        raise AttributeError(f'{a} is not found or not supported!')
###################################################


wn = Network(wn_inp_path)
#wn_g = wntr.network.WaterNetworkModel(wn_inp_path).get_graph()
skip_nodes = config.get('general','skip_nodes').strip().split(',')  if config.has_option('general', 'skip_nodes') else None

valve_type_dict={}
# for valve in wn.valves:
#     valve_type = str(valve.valve_type) 
#     if valve_type not in valve_type_dict:
#         valve_type_dict[valve_type] = np.array(config.get('valve',f'setting_{valve_type}').strip().split(','),dtype=float)


featlen_dict= dict()


if len(wn.junctions) >0:
    if args.gen_demand: 
        featlen_dict[ ParamEnum.JUNC_DEMAND]= len(wn.junctions)
        
    if args.gen_elevation :
        featlen_dict[ParamEnum.JUNC_ELEVATION]   = len(wn.junctions)

if len(wn.pipes) > 0:
    num_pipes = len(wn.pipes)
    if args.gen_roughness: 
        featlen_dict[ ParamEnum.PIPE_ROUGHNESS]= num_pipes
    if args.gen_diameter: 
        featlen_dict[ ParamEnum.PIPE_DIAMETER]= num_pipes
    if args.gen_length: 
        featlen_dict[ ParamEnum.PIPE_LENGTH]= num_pipes
    if args.gen_minorloss: 
        featlen_dict[ ParamEnum.PIPE_MINORLOSS]= num_pipes

if len(wn.pumps) >0:
    num_pumps = len(wn.pumps)
    if args.gen_pump_init_status:
        featlen_dict[ParamEnum.PUMP_STATUS]= num_pumps
    if args.gen_pump_speed:
        featlen_dict[ParamEnum.PUMP_SPEED]= num_pumps
    if args.gen_pump_length:
        featlen_dict[ParamEnum.PUMP_LENGTH]= num_pumps

if len(wn.tanks) >0:
    num_tanks = len(wn.tanks)
    if args.gen_tank_level:
        featlen_dict[ParamEnum.TANK_LEVEL]= num_tanks
    if args.gen_tank_elevation:
        featlen_dict[ParamEnum.TANK_ELEVATION]= num_tanks
    if args.gen_tank_diameter:
        featlen_dict[ParamEnum.TANK_DIAMETER]= num_tanks
         
if len(wn.valves)  >0:
    num_valves = len(wn.valves) 
    if args.gen_valve_init_status :
        featlen_dict[ParamEnum.VALVE_STATUS]= num_valves
    if args.gen_valve_setting :
        featlen_dict[ParamEnum.VALVE_SETTING]= num_valves
    if args.gen_valve_diameter :
        featlen_dict[ParamEnum.VALVE_DIAMETER]= num_valves
    
if args.gen_res_total_head and len(wn.reservoirs)  >0:
    featlen_dict[ParamEnum.RESERVOIR_TOTALHEAD]= len(wn.reservoirs) 

########################################################
last_results = []

print('Start simulation...')
print('saved_path = ',saved_path)
skip_nodes = skip_links = []
num_skip_nodes = num_skip_links = 0
if config.has_option('general', 'skip_nodes'):
    skip_nodes = config.get('general','skip_nodes').strip().split(',') 

if args.skip_resevoir_result:
    skip_nodes.extend(wn.reservoirs.uid.to_list())

num_skip_nodes = len(skip_nodes)

print(f'skip nodes = {skip_nodes}')
print(f'#skip_nodes = {num_skip_nodes}')

if config.has_option('general', 'skip_links'):
    skip_links = config.get('general','skip_links').strip().split(',')  
    num_skip_links = len(skip_links)
print(f'#skip_links = {num_skip_links}')

node_uids =  wn.nodes.uid
num_result_nodes = len(node_uids.loc[~node_uids.isin(skip_nodes)]) if skip_nodes else len(node_uids)
print(f'exepected #result_nodes = {num_result_nodes} | Note that if attribute is \'demand\', #results_nodes should be #junctions')

link_uids =  wn.links.uid
num_result_links =  len(link_uids.loc[~link_uids.isin(skip_links)]) if skip_links else len(link_uids)
print(f'exepected #result_links = {num_result_links}')
###########################################################

#store = zarr.DirectoryStore(zarr_storage_dir)
store   = zarr.DirectoryStore('test')
tg = RayTokenGenerator(store=store,
                    num_scenes=backup_num_scenarios,
                    featlen_dict=featlen_dict,
                    num_chunks=num_chunks)

#tg.init()
tg.sequential_update(args=args)
ragged_tokens = tg.load_computed_params()
root_group = zarr.open_group(store,
                        synchronizer=zarr.ThreadSynchronizer())


tmp_group = root_group.create_group('tmp', overwrite=True)
for att in expected_attributes:
    if att in support_node_attr_keys:
        if att == 'demand':
            uids =  wn.junctions.uid
            num_junctions = len(uids.loc[~uids.isin(skip_nodes)])  if skip_nodes else len(uids)
            tmp_group.create(att,shape=[num_scenarios,num_junctions],
                            chunks=[batch_size,num_result_nodes],
                            overwrite=True)
        else:
            tmp_group.create(att,shape=[num_scenarios,num_result_nodes],
                            chunks=[batch_size,num_result_nodes],
                            overwrite=True)
        
    elif att in support_link_attr_keys:
        tmp_group.create(att,shape=[num_scenarios,num_result_links],
                        chunks=[batch_size,num_result_links],
                        overwrite=True)






def single_thread_executor(batch_size,
                           ragged_tokens,
                           new_featlen_dict,
                           config,
                           valve_type_dict,
                           args,
                           tmp_group,
                           num_batches,
                           ):
    token_ids = []
    scene_ids = []

    for batch_id in range(num_batches):
        start_id = batch_id * batch_size
        end_id = start_id + batch_size
        batch_ragged_tokens =  ragged_tokens[start_id:end_id]
        token_ids.append(batch_ragged_tokens)
        scene_ids.append([start_id + x for x in range(batch_size)])

    sim_start = time()
    executor = WDNExecutor(
                            featlen_dict= new_featlen_dict,
                            config=config,
                            valve_type_dict= valve_type_dict,
                            args=args,
                            ) 
    
    start_index = 0
    progressbar = tqdm(total=num_batches) 
    
    ordered_names_dict = {}
    success_scenarios = 0
    while len(token_ids) > 0:
        catch_error=False
        try:
            result, ordered_name_list  = executor.simulate(token_ids.pop(), scene_ids.pop())
        except Exception as e:
            print(e)
            catch_error = True

        if not catch_error:
            success_size = 0
            for key,value in result.items():
                if key not in ordered_names_dict:
                    ordered_names_dict[key] = ordered_name_list
                if start_index + value.shape[0] < tmp_group[key].shape[0]:
                    success_size =  value.shape[0]
                    tmp_group[key][start_index:start_index + success_size] = value
                else:
                    success_size = tmp_group[key].shape[0] - start_index
                    tmp_group[key][start_index:start_index + success_size] = value[:success_size]
            
            del result
            start_index += success_size
            success_scenarios+= success_size
        progressbar.update(1)
    
    progressbar.close()
    ray.shutdown()
    elapsed_time = time() - sim_start
    print(f'\nSimulation time: { elapsed_time } seconds')

    print(f'Success/Total: {success_scenarios}/{num_scenarios} scenes') 
    return success_scenarios, ordered_names_dict

try:
    sim_start = time()
    
    '''
        #use it for debug
        success_scenarios, ordered_names_dict= single_thread_executor(
                                                    batch_size,
                                                    ragged_tokens,
                                                    featlen_dict,
                                                    config,
                                                    valve_type_dict,
                                                    args,
                                                    tmp_group,
                                                    num_batches)
    '''
    
    token_ids = []
    scene_ids = []

    for batch_id in range(num_batches):
        start_id = batch_id * batch_size
        end_id = start_id + batch_size
        batch_ragged_tokens =  ragged_tokens[start_id:end_id]
        token_ids.append(ray.put(batch_ragged_tokens))
        scene_ids.append(ray.put([start_id + x for x in range(batch_size)]))
    new_featlen_dict = featlen_dict # we don't create new features
    
    executors = [WDNRayExecutor.remote(
                                    featlen_dict= new_featlen_dict,
                                    config=config,
                                    valve_type_dict= valve_type_dict,
                                    args=args,
                                    ) for _ in range(num_executors)]
    

    start_index = 0
    progressbar = tqdm(total=num_batches) #tqdm(total=num_batches,desc="batch" ,leave=False, colour='red')
    #successbar = tqdm(total= num_scenarios,desc="scase",leave=False, colour='green')
    result_worker_dict = {e.simulate.remote(token_ids.pop(), scene_ids.pop()) : e for  e in executors  if scene_ids }
    done_ids,_ = ray.wait(list(result_worker_dict),num_returns=1)

    ordered_names_dict = {}
    success_scenarios = 0
    while done_ids and success_scenarios < num_scenarios:
        done_worker_id = done_ids[0]
        catch_error = False
        try:
            result, ordered_name_list  = ray.get(done_worker_id)
        except RayError as e:
            print(f'WARNING! Ray error {e}')
            catch_error = True
        worker  = result_worker_dict.pop(done_worker_id)
        if scene_ids:
            result_worker_dict[worker.simulate.remote(token_ids.pop(), scene_ids.pop())] = worker
        
        if not catch_error:
            success_size = 0
            #write_id = dataset[start_index:start_index + batch_size].write(result) #.result()
            for key,value in result.items():
                if key not in ordered_names_dict:
                    ordered_names_dict[key] = ordered_name_list

                #print(f'key = {key}, value shape = {value.shape}')
                if start_index + value.shape[0] < tmp_group[key].shape[0]:
                    success_size =  value.shape[0]
                    tmp_group[key][start_index:start_index + success_size] = value
                else:
                    success_size = tmp_group[key].shape[0] - start_index
                    tmp_group[key][start_index:start_index + success_size] = value[:success_size]
            
            del result
            start_index += success_size
            success_scenarios+= success_size
            #successbar.update(success_size)
        progressbar.update(1)
        done_ids,_ = ray.wait(list(result_worker_dict),num_returns=1)
    
    #successbar.close()
    progressbar.close()
    ray.shutdown()
    


    

    elapsed_time = time() - sim_start
    print(f'\nSimulation time: { elapsed_time } seconds')
    print(f'Process run on {num_batches} batches, total scenes: {backup_num_scenarios}')
    print(f'Success/Expected: {success_scenarios}/{num_scenarios} scenes')
    
    del root_group[ParamEnum.RANDOM_TOKEN]
    if success_scenarios > 0:
        for name in list(tmp_group.keys()):
            name_group= root_group.create_group(name,overwrite=True)
            if success_scenarios != num_scenarios:
                #reshape
                tmp_group[name].resize(success_scenarios,tmp_group[name].shape[-1])
                if success_scenarios < batch_size:
                    #TODO: rechunk to maximize the r/w speed
                    pass
        
        train_index = int(success_scenarios * train_ratio)
        valid_index = train_index + int(success_scenarios * valid_ratio)

        key_list = list(tmp_group.keys())

        config_dict = {sect: dict(config.items(sect)) for sect in config.sections()} 
        if skip_nodes:
            config_dict['skip_nodes'] = skip_nodes
        if skip_links:
            config_dict['skip_links'] = skip_links
        
        root_group.attrs['config'] = config_dict
        root_group.attrs['args'] =  vars(args)
        root_group.attrs['ordered_names_by_attr'] = ordered_names_dict
        
        for key in key_list:
            a = tmp_group[key]
            #print(f'\n{key}.info: {a.info}')
            train_a,valid_a,test_a = a[:train_index],a[train_index:valid_index],a[valid_index:]
            
            train_a_df = pd.DataFrame(train_a).astype(float)
            train_min = train_a.min() 
            train_max = train_a.max()
            train_mean = train_a.mean()
            train_std = train_a.std()
            

            train_mean_feat_coef = train_a_df.corr().mean().mean()  #np.corrcoef(train_a.T).mean()
            train_mean_batch_coef = train_a_df.T.corr().mean().mean() #np.corrcoef(train_a).mean()
            train_cv =  (train_a.var(axis=-1)/ train_a.mean(axis=-1)).mean()

            root_group[key].attrs['min'] = train_min
            root_group[key].attrs['max'] = train_max
            root_group[key].attrs['mean'] = train_mean
            root_group[key].attrs['std'] = train_std
            root_group[key].attrs['mcoef'] = train_mean_feat_coef
            root_group[key].attrs['bcoef'] = train_mean_batch_coef
            root_group[key].attrs['cv'] = train_cv
            
            print(f'##############################{key}###############################################')
            print(f'min         : {train_min}')
            print(f'max         : {train_max}')
            print(f'mean        : {train_mean}')
            print(f'std         : {train_std}')
            print(f'mean fcoef  : {train_mean_feat_coef}')
            print(f'mean bcoef  : {train_mean_batch_coef}')
            print(f'cv          : {train_cv}')
                                                                                                                                                                                                                                                                                                                                                                                                                                    

            key_train = os.path.join(key,'train')
            root_group.empty_like(key_train,train_a,chunks=(batch_size, a.chunks[-1]))
            root_group[key_train][:] = train_a
            print(f'\n{key_train}.info: {root_group[key_train].info}')

            key_valid = os.path.join(key,'valid')
            root_group.empty_like(key_valid,valid_a,chunks= (batch_size, a.chunks[-1]))
            root_group[key_valid][:] = valid_a
            print(f'\n{key_valid}.info: {root_group[key_valid].info}')

            key_test = os.path.join(key,'test')
            root_group.empty_like(key_test,test_a,chunks= (batch_size, a.chunks[-1]))
            root_group[key_test][:] = test_a
            print(f'\n{key_test}.info: {root_group[key_test].info}')

            #del tmp_group[key]

        del root_group['tmp'] 

        elapsed_time = time() - program_start
        print(f'\nExecution time: { elapsed_time } seconds')
        
        store2= zarr.ZipStore(saved_path+'.zip', mode='w')
        zarr.copy_store(store,store2,if_exists='replace')
        store2.close()
        print(root_group.tree())

        if args.debug:
            f, axs = plt.subplots(2,1)
            axs[0].hist(np.mean(train_a, axis=0), bins=100, alpha=1)
            axs[0].set_title('Histogram-  mean axis = 0')
            axs[1].hist(np.mean(train_a, axis=1), bins=100, alpha=1)
            axs[1].set_title('Histogram-  mean axis = 1')
            plt.show()
except Exception as e:
    print(e)
#finally:
#    shutil.rmtree(random_array_dir)
 