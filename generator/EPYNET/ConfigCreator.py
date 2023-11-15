#
# Created on Mon Oct 16 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Create a dummy config w.r.t an INP file 
# Remark: config has .ini as its extension
# ------------------------------
#

from epynet import Network
from configparser import ConfigParser
import os
import numpy as np
import argparse

#define a dummy configs
parser = argparse.ArgumentParser()
parser.add_argument('--wn_inp_path', default = r'G:\Other computers\My Laptop\PhD\Codebase\signal-recovery\inputs\ctown.inp', type = str, help = 'water EPANET INPUT path')
parser.add_argument('--config_variant', default = '7v1', type = str, help = 'additional name for config')
parser.add_argument('--num_scenarios', default = 10,type=float,help='number of generated scenarios')

parser.add_argument('--junc_demand_lo', default =0,type=float,help='lowest demand')
parser.add_argument('--junc_demand_hi', default = 60,type=float,help='highest demand')
parser.add_argument('--junc_demand_is_quantile', default = True,type=float,help='flag indicates quantile. If True, dmd_lo and dmd_hi shoul be in range of [0,100]')
parser.add_argument('--junc_ele_lo', default =None,type=float,help='lowest elevation')
parser.add_argument('--junc_ele_hi', default = None,type=float,help='highest elevation')
parser.add_argument('--junc_ele_is_quantile', default = False,type=float,help='flag indicates quantile. If True, junc_ele_lo and junc_ele_hi shoul be in range of [0,100]')

parser.add_argument('--pump_open_prob', default =1.0,type=float,help='status ratio whether open or not ')
parser.add_argument('--pump_speed_lo', default =None,type=float,help='lowest speed')
parser.add_argument('--pump_speed_hi', default = None,type=float,help='highest speed')
parser.add_argument('--pump_speed_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pump_speed_lo and pump_speed_hi shoul be in range of [0,100]')
parser.add_argument('--pump_length_lo', default =None,type=float,help='lowest length')
parser.add_argument('--pump_length_hi', default = None,type=float,help='highest length')
parser.add_argument('--pump_length_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pump_length_lo and pump_length_hi shoul be in range of [0,100]')

parser.add_argument('--tank_level_lo', default =0,type=float,help='lowest level')
parser.add_argument('--tank_level_hi', default =30,type=float,help='highest level')
parser.add_argument('--tank_level_is_quantile', default = True,type=float,help='flag indicates quantile. If True, tank_level_lo and tank_level_hi shoul be in range of [0,100]')
parser.add_argument('--tank_ele_lo', default =None,type=float,help='lowest elevation')
parser.add_argument('--tank_ele_hi', default = None,type=float,help='highest elevation')
parser.add_argument('--tank_ele_is_quantile', default = False,type=float,help='flag indicates quantile. If True, tank_ele_lo and tank_ele_hi shoul be in range of [0,100]')
parser.add_argument('--tank_dia_lo', default =None,type=float,help='lowest diameter')
parser.add_argument('--tank_dia_hi', default = None,type=float,help='highest diameter')
parser.add_argument('--tank_dia_is_quantile', default = False,type=float,help='flag indicates quantile. If True, tank_dia_lo and tank_dia_hi shoul be in range of [0,100]')

parser.add_argument('--valve_open_prob', default =1.0,type=float,help='status ratio whether open or not ')
parser.add_argument('--valve_dia_lo', default =None,type=float,help='lowest diameter')
parser.add_argument('--valve_dia_hi', default = None,type=float,help='highest diameter')
parser.add_argument('--valve_dia_is_quantile', default = False,type=float,help='flag indicates quantile. If True, valve_dia_lo and valve_dia_hi shoul be in range of [0,100]')

parser.add_argument('--pipe_roughness_lo', default =None,type=float,help='lowest roughness')
parser.add_argument('--pipe_roughness_hi', default = None,type=float,help='highest roughness')
parser.add_argument('--pipe_roughness_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pipe_roughness_lo and pipe_roughness_hi shoul be in range of [0,100]')
parser.add_argument('--pipe_dia_lo', default =None,type=float,help='lowest diameter')
parser.add_argument('--pipe_dia_hi', default =None,type=float,help='highest diameter')
parser.add_argument('--pipe_dia_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pipe_dia_lo and pipe_dia_hi shoul be in range of [0,100]')
parser.add_argument('--pipe_length_lo', default =None,type=float,help='lowest length')
parser.add_argument('--pipe_length_hi', default = None,type=float,help='highest length')
parser.add_argument('--pipe_length_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pipe_length_lo and pipe_length_hi shoul be in range of [0,100]')
parser.add_argument('--pipe_minorloss_lo', default =None,type=float,help='lowest minorloss')
parser.add_argument('--pipe_minorloss_hi', default = None,type=float,help='highest minorloss')
parser.add_argument('--pipe_minorloss_is_quantile', default = False,type=float,help='flag indicates quantile. If True, pipe_minorloss_lo and pipe_minorloss_hi shoul be in range of [0,100]')

parser.add_argument('--res_head_lo', default = 15,type=float,help='lowest reservoir total head. If None, auto pick')
parser.add_argument('--res_head_hi', default = 30,type=float,help='highest reservoir total head. If None, auto pick')
parser.add_argument('--head_add_ele', default = True,type=float,help='total_head = ele_head + pressure_head. This flag allows the ele head addition')


args = parser.parse_args([])


def get_range(arr: np.ndarray,lo: float,hi:float,is_quantile:bool=False)-> tuple[np.ndarray,np.ndarray]:
  """get range of a hydraulic parameter

  Args:
      arr (np.ndarray): array contains values of a specific hydraulic parameters (e.g., junction demands)
      lo (float): quantile or None to choose min
      hi (float): quantile or None to choose max
      is_quantile (bool, optional): indicates if lo, hi are quantiles. Defaults to False.

  Returns:
      tuple[np.ndarray,np.ndarray]: lower bound and upper bound of this hydraulic param
  """
  if is_quantile:
    assert lo is not None and hi is not None
    data_lo = np.percentile(arr,lo) 
    data_hi = np.percentile(arr,hi) 
  else:
    data_lo = np.min(arr) if lo is None else lo
    data_hi = np.max(arr) if hi is None else hi
  return data_lo, data_hi
  

def create_dummy_config(config_dir_path:str) -> tuple[ConfigParser, str]:
  """Create dummy config given an INP file.
  For each hydraulic parameter,
    We simply get [Q1,Q3] or [min,max] as its range

  Args:
      config_dir_path (str, optional): place to store the created config

  Raises:
      FileNotFoundError: water network INP path is invalid

  Returns:
      tuple[ConfigParser, str]: config object file, config file path
  """
  wn_inp_path = args.wn_inp_path
  config = ConfigParser()
  if not os.path.isfile(wn_inp_path):
    raise FileNotFoundError()

  config_path = os.path.join(os.getcwd(), config_dir_path,os.path.basename(wn_inp_path[:-4])+f'_{args.config_variant}_'+'_EPYNET_config.ini') 

  # load a network
  wn = Network(wn_inp_path)

  config['general'] = {
      'wn_inp_path'     : wn_inp_path,
      'config_path'     : config_path,
      'storage_dir'     : 'datasets/'+os.path.basename(wn_inp_path)[:-4],
      'doe'             : 'uniform',#'hammersly','lhs'
      'num_scenarios'   : args.num_scenarios,
  }
  junction_dict={}
  base_demands = wn.junctions.basedemand.to_numpy() 
  junction_dict['demand_lo'], junction_dict['demand_hi'] = get_range(base_demands,args.junc_demand_lo,args.junc_demand_hi,args.junc_demand_is_quantile)
  
  elevations = wn.junctions.elevation.to_numpy()
  junction_dict['ele_lo'], junction_dict['ele_hi'] = get_range(elevations,args.junc_ele_lo,args.junc_ele_hi,args.junc_ele_is_quantile)

  config['junction'] = junction_dict
  #############################################################################################
  pump_dict = {}
  if len(wn.pumps)>0:
    pump_dict['open_prob'] = args.pump_open_prob if args.pump_open_prob else 1.0
    base_speeds =  wn.pumps.speed.to_numpy()
    pump_dict['speed_lo'], pump_dict['speed_hi'] = get_range(base_speeds,args.pump_speed_lo,args.pump_speed_hi, args.pump_speed_is_quantile)
    
    base_lengths =  wn.pumps.length.to_numpy()
    pump_dict['length_lo'], pump_dict['length_hi'] = get_range(base_lengths,args.pump_length_lo,args.pump_length_hi,args.pump_length_is_quantile)
    config['pump'] = pump_dict
    
  ########################################################################################
  tank_dict={}
  if len(wn.tanks)>0:
    min_levels = wn.tanks.minlevel.to_numpy() #np.array([min_lv for min_lv in wn_min_level])
    max_levels = wn.tanks.maxlevel.to_numpy() #np.array([max_lv for max_lv in wn_max_level])
    tank_dict['level_lo'],_ = get_range(min_levels,args.tank_level_lo,100,args.tank_level_is_quantile)
    _,tank_dict['level_hi'] = get_range(max_levels,0,args.tank_level_hi,args.tank_level_is_quantile)

    elevations = wn.tanks.elevation.to_numpy()
    tank_dict['ele_lo'], tank_dict['ele_hi'] = get_range(elevations,args.tank_ele_lo,args.tank_ele_hi,args.tank_ele_is_quantile)

    diameters = wn.tanks.diameter.to_numpy()
    tank_dict['dia_lo'], tank_dict['dia_hi'] = get_range(diameters,args.tank_dia_lo,args.tank_dia_hi,args.tank_dia_is_quantile)

    config['tank'] = tank_dict
  
  ##############################################################################################
  
  valve_dict={}
  if len(wn.valves)>0:
    valve_dict['open_prob']= args.valve_open_prob if args.valve_open_prob else 1.0
    valve_type_dict = {}
    for valve in wn.valves:
        if valve.valve_type not in valve_type_dict:
          valve_type_dict[valve.valve_type] = []
       
        valve_type_dict[valve.valve_type].append(valve.setting if  valve.setting > 0 else 0. )

    for valve_type in valve_type_dict.keys():
      #valve_type_dict[valve_type] = np.max(valve_type_dict[valve_type]) * args.setting_var_scale
      valve_dict[f'setting_{valve_type}_lo']= np.min(valve_type_dict[valve_type]) 
      valve_dict[f'setting_{valve_type}_hi']= np.percentile(valve_type_dict[valve_type],95)
    
    diameters = wn.valves.diameter.to_numpy()
    valve_dict['dia_lo'], valve_dict['dia_hi'] = get_range(diameters,args.valve_dia_lo,args.valve_dia_hi,args.valve_dia_is_quantile)
    config['valve'] = valve_dict 
  ##############################################################################################
  pipe_dict={}
  roughnesses = wn.pipes.roughness.to_numpy()
  pipe_dict['roughness_lo'], pipe_dict['roughness_hi'] = get_range(roughnesses,args.pipe_roughness_lo,args.pipe_roughness_hi,args.pipe_roughness_is_quantile)

  diameters = wn.pipes.diameter.to_numpy() 
  pipe_dict['diameter_lo'],pipe_dict['diameter_hi'] = get_range(diameters,args.pipe_dia_lo,args.pipe_dia_hi,args.pipe_dia_is_quantile)

  
  lengths = wn.pipes.length.to_numpy() 
  pipe_dict['length_lo'],pipe_dict['length_hi'] = get_range(lengths,args.pipe_length_lo,args.pipe_length_hi,args.pipe_length_is_quantile)

  minorlosses = wn.pipes.minorloss.to_numpy() 
  pipe_dict['minorloss_lo'],pipe_dict['minorloss_hi'] = get_range(minorlosses,args.pipe_minorloss_lo,args.pipe_minorloss_hi,args.pipe_minorloss_is_quantile)

  config['pipe'] = pipe_dict
  ##############################################################################################
  reservoir_dict={}
  if len(wn.reservoirs)>0:
    #res_heads =  wn.reservoirs.head.to_numpy()
    junc_elevations = wn.junctions.elevation
    tmp=tmp2=0
    if args.head_add_ele:
      sorted_junc_elevations = np.sort(junc_elevations)
      top_10_sorted_junc_elevations = sorted_junc_elevations[-10:]
    while tmp == tmp2:
      random_pressures = args.res_head_lo + np.random.random(size=2) * (args.res_head_hi - args.res_head_lo)
      if args.head_add_ele:
        random_heights = np.random.choice(np.arange(10),size=2)
        tmp = top_10_sorted_junc_elevations[random_heights[0]] + random_pressures[0]
        tmp2 = top_10_sorted_junc_elevations[random_heights[1]] + random_pressures[1]
      else:
        tmp,tmp2 = random_pressures[0], random_pressures[1]
    head_lo = tmp if tmp < tmp2 else tmp2
    head_hi = tmp if tmp != head_lo else tmp2
    reservoir_dict['head_lo'], reservoir_dict['head_hi'] = head_lo,  head_hi
    config['reservoir'] = reservoir_dict
  ################################################################################################
  os.makedirs(config_dir_path,exist_ok=True)
  with open(config_path, 'w') as configfile:
      config.write(configfile)

  return config,config_path


def print_config(config: ConfigParser):
  """support beautifying printing

  Args:
      config (ConfigParser): config or dict
  """
  for section in list(config.keys()):
    print(f'\n[{section}]')
    for k,v in config[section].items():
      print(f'{k} : {v}')

if __name__=='__main__':
  config,config_path = create_dummy_config(config_dir_path=r'configs\v7.1')
  print(f'Config path = {config_path}')
  print_config(config=config)


