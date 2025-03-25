#
# Created on Mon Oct 16 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: utils functions for data generation
# We re-write some EPYNET code to generation in parallel without error
# ------------------------------
#

import numpy as np
import networkx as nx
import configparser
import os
import argparse
from epynet import epanet2
import epynet
import pint

def get_networkx_graph(wn,include_reservoir=True,graph_type='multi_directed'):
    """support to form a networkx graph from the Epynet water network
    ref: https://github.com/BME-SmartLab/GraphConvWat/blob/be97b45fbc7dfdba22bb1ee406424a7c568120e5/utils/graph_utils.py
    :param Epynet.Network wn: water network object
    :param bool include_reservoir: Flag indicates involve links from reservoirs, defaults to True
    """
    if graph_type == 'undirected':
        G = nx.Graph()
    elif graph_type == 'directed':
        G = nx.DiGraph()
    elif graph_type == 'multi_undirected':
        G = nx.MultiGraph()
    elif  graph_type == 'multi_directed':
        G = nx.MultiDiGraph()
    else:
        raise NotImplementedError()
    
    node_list = []
    collection = wn.junctions if not include_reservoir else wn.nodes
    for node in collection:
        node_list.append(node.uid)


    for pipe in wn.pipes:
        if (pipe.from_node.uid in node_list) and (pipe.to_node.uid in node_list):
            G.add_edge(pipe.from_node.uid, pipe.to_node.uid, weight=1., length=pipe.length)
        else:
            print(f'WARNING! pipe {pipe.uid} is not connect to any node in node list')
    for pump in wn.pumps:
        if (pump.from_node.uid in node_list) and (pump.to_node.uid in node_list):
            G.add_edge(pump.from_node.uid, pump.to_node.uid, weight=1., length=0.)
        else:
            print(f'WARNING! pump {pump.uid} is not connect to any node in node list')
    for valve in wn.valves:
        if (valve.from_node.uid in node_list) and (valve.to_node.uid in node_list):
            G.add_edge(valve.from_node.uid, valve.to_node.uid, weight=1., length=0.)
        else:
            print(f'WARNING! valve {valve.uid} is not connect to any node in node list')
    
    return G



def args2config(args,config_name,config_path=None):
    """convert argparse args to config for saving

    :param dict args: arguments 
    :param str config_name: name of the new config
    :param str config_path: optional storage path 
    """
    config = configparser.ConfigParser()
    if config_path is not None:
        os.makedirs(config_path,exist_ok=True)
    
    config.defaults().update(vars(args))

    full_path = os.path.join(config_path,config_name)
    with open(full_path, 'w') as configfile:
        config.write(configfile)

    return full_path

def config2args(config_path):
    """convert config to args

    :param str config_name: name of the new config
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    defaults =  config.defaults()
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    return parser.parse_args()


def set_object_value_wo_ierror(obj, code, value):
    """this function is similar to setattrr(obj,code,value) but omit the ierror flag causes crash when running RAY
    It is a sequence of [setattr -> set_static_property -> ENsetnodevalue] but we alternate to:
    [setattr -> set_static_property -> ENsetnodevalue2] to skip ierror
    :param Object obj: object from EPYNET network
    :param int code: EPANET Param Code
    :param Any value: value
    """
    #tank._values[epanet2.EN_TANKLEVEL] = tank_level
    #self.wn.ep.ENsetnodevalue2(tank.index, epanet2.EN_TANKLEVEL, tank_level)

    assert hasattr(obj,'_values') and hasattr(obj,'index') and obj.network() is not None
    #ref https://github.com/Vitens/epynet/blob/992ce792c6b6427ee0d35325645c8185bc888928/epynet/baseobject.py#L42
    obj.network().solved = False
    obj._values[code] = value
    try:
        if isinstance(obj, epynet.Node):
            ENsetnodevalue2(obj.network().ep, obj.index, code, value)
        else:
            ENsetlinkvalue2(obj.network().ep, obj.index, code, value)
    except Exception as e:
        print(f'ERROR AT OBJ = {obj.uid},code = {code} , value = {value}')
        raise Exception(e)

def ENhasppatern(obj):
    if not isinstance(obj, epynet.Junction):
        return False
    
    try:
        p = obj.pattern
        return p is not None
    except Exception:
        return False
    



def ENsetnodevalue2(ep, index, paramcode, value):
        """Sets the value of a parameter for a specific node.
        Arguments:
        index:  node index
        paramcode: Node parameter codes consist of the following constants:
                      EN_ELEVATION  Elevation
                      EN_BASEDEMAND ** Base demand
                      EN_PATTERN    ** Demand pattern index
                      EN_EMITTER    Emitter coeff.
                      EN_INITQUAL   Initial quality
                      EN_SOURCEQUAL Source quality
                      EN_SOURCEPAT  Source pattern index
                      EN_SOURCETYPE Source type (See note below)
                      EN_TANKLEVEL  Initial water level in tank
                           ** primary demand category is last on demand list
                   The following parameter codes apply only to storage tank nodes
                      EN_TANKDIAM      Tank diameter
                      EN_MINVOLUME     Minimum water volume
                      EN_MINLEVEL      Minimum water level
                      EN_MAXLEVEL      Maximum water level
                      EN_MIXMODEL      Mixing model code
                      EN_MIXFRACTION   Fraction of total volume occupied by the inlet/outlet
                      EN_TANK_KBULK    Bulk reaction rate coefficient
        value:parameter value"""
        ierr = ep._lib.EN_setnodevalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode), epanet2.ctypes.c_float(value))
        if ierr!=0: raise Exception(ierr)
        del ierr
        #raise ENtoolkitError(self,100) if self._lib.EN_setnodevalue(self.ph, ctypes.c_int(index), ctypes.c_int(paramcode), ctypes.c_float(value)) != 0 else None
        #if ierr!=0: raise ENtoolkitError(self, ierr)


def ENsetlinkvalue2(ep, index, paramcode, value):
        ierr = ep._lib.EN_setlinkvalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode), epanet2.ctypes.c_float(value))
        if ierr!=0: raise Exception(ierr)
        del ierr

def ENdeletepattern(wn, pattern_uid, delete_pattern_in_rules=True):
    patten_index = epanet2.ctypes.c_int()
    
    wn.ep._lib.EN_getpatternindex(wn.ep.ph,epanet2.ctypes.c_char_p(pattern_uid.encode(wn.ep.charset)), epanet2.ctypes.byref(patten_index))
    wn.ep._lib.EN_deletepattern(wn.ep.ph, patten_index)
    if delete_pattern_in_rules:
        wn.ep._lib.EN_deleterule(wn.ep.ph, patten_index)


def ENdeletepatternbyindex(wn, pattern_index, delete_pattern_in_rules=True):
    wn.ep._lib.EN_deletepattern(wn.ep.ph, epanet2.ctypes.c_int(pattern_index))
    if delete_pattern_in_rules:
        wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(pattern_index))

def ENsetdemandpattern(wn, node_index ,demand_category, pattern_index):
    wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),  epanet2.ctypes.c_int(pattern_index))

def ENsetdemandpatterntoallcategories(wn, node_index, base_demand , pattern_index):
    demand_category = 1
    ierr= wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),  epanet2.ctypes.c_int(pattern_index))
    ierr= wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),  epanet2.ctypes.c_double(base_demand))
    while ierr == 0:
        demand_category+=1
        ierr= wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),  epanet2.ctypes.c_int(pattern_index))
        if ierr == 0:
            ierr= wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category),  epanet2.ctypes.c_double(base_demand))

def ENsetheadcurveindex(wn, pump_index, curve_index):
        ierr = wn.ep._lib.EN_setheadcurveindex(wn.ep.ph, epanet2.ctypes.c_int(pump_index), epanet2.ctypes.c_int(curve_index))
        if ierr!=0: raise Exception(ierr)
     

def ENdeletecontrol(ep, control_index):
    ierr= ep._lib.EN_deletecontrol(ep.ph,  epanet2.ctypes.c_int(control_index))
    if ierr!=0: raise  Exception( ierr) #epanet2.ENtoolkitError(ep, ierr)
    del ierr

def ENdeleterule(ep, rule_index):
    ierr= ep._lib.EN_deleterule(ep.ph,  epanet2.ctypes.c_int(rule_index))
    if ierr!=0: raise  Exception( ierr)
    del ierr

def ENdeleteallcontrols(wn):
    current_control_index = 1 #start index
    ierr= wn.ep._lib.EN_deletecontrol( wn.ep.ph,  epanet2.ctypes.c_int(current_control_index))
    #print(f'pre-ENdeleteallcontrols i = {current_control_index}, ierr = {ierr}')
    while(ierr == 0): #no error
        current_control_index = 1   
        ierr=  wn.ep._lib.EN_deletecontrol( wn.ep.ph,  epanet2.ctypes.c_int(current_control_index))
        #print(f'pre-ENdeleteallcontrols i = {current_control_index}, ierr = {ierr}')
    #[wn.ep._lib.EN_deletecontrol( wn.ep.ph,  epanet2.ctypes.c_int(trial_index)) for trial_index in range(1,20)]
    
    del ierr
     
    #ctype = epanet2.ctypes.c_int()
    #lindex = epanet2.ctypes.c_int()
    #setting= epanet2.ctypes.c_int()
    #nindex = epanet2.ctypes.c_int()
    #level= epanet2.ctypes.c_int()
    #for i in range(1,10):
    #    err=  wn.ep._lib.EN_getcontrol( wn.ep.ph, epanet2.ctypes.c_int(i), epanet2.ctypes.byref(ctype), 
    #                    epanet2.ctypes.byref(lindex), epanet2.ctypes.byref(setting), 
    #                    epanet2.ctypes.byref(nindex), epanet2.ctypes.byref(level) )
    #    print(f'After-ENdeleteallcontrols i = {i}, ierr = {err}')
    

def ENdeleteallpatterns(wn):
    current_pattern_index = 1 #start index
    ierr = wn.ep._lib.EN_deletepattern(wn.ep.ph, epanet2.ctypes.c_int(current_pattern_index))
    
    print(f'1-ENdeleteallpatterns id {current_pattern_index}-ierr = {ierr}')
    while(ierr == 0): #no error
        current_pattern_index = 1   
        ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_pattern_index))
    print(f'1-ENdeleteallpatterns id {current_pattern_index}-ierr = {ierr}')
    del ierr

def ENdeleteallrules(wn):
    current_rule_index = 1 #start index
    ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_rule_index))
    
    #print(f'1-ENdeleteallrules id {current_rule_index}-ierr = {ierr}')
    while(ierr == 0): #no error
        current_rule_index = 1   
        ierr = wn.ep._lib.EN_deleterule(wn.ep.ph, epanet2.ctypes.c_int(current_rule_index))
        #print(f'ENdeleteallrules id {current_rule_index}-ierr = {ierr}')

    del ierr

def ENconvert(from_unit, to_unit, hydraulic_param, values):
    """ENconvert helps convert value 

    :param str from_unit: original flow unit should be in []
    :param str to_unit: target flow unit should be in []
    :param str hydraulic_param: currently support ['pressure', 'demand', 'head',' velocity', 'flow']
    :param np.array values: _description_
    """
    
    us_flow_units = ['CFS','GPM', 'MGD', 'IMGD', 'AFD']
    si_flow_units = ['LPS','LPM', 'MLD', 'CMH', 'CMD']
    supported_flow_units =  list(set(us_flow_units).union(si_flow_units))
    assert from_unit in supported_flow_units
    assert to_unit in  supported_flow_units
    assert hydraulic_param in ['pressure', 'demand', 'head',' velocity', 'flow']
    assert isinstance(values, np.ndarray)


    ureg = pint.UnitRegistry()
    
    #define quantity
    #basic units: https://github.com/hgrecco/pint/blob/master/pint/default_en.txt
    #flow and demand
    ureg.define('GPM = gallon / minute')
    ureg.define('cubic_meter = meter**3')
    ureg.define('CMH = cubic_meter / hour')
    ureg.define('meter_H2O = 100 * centimeter_H2O')
    ureg.define('CFS = cubic_feet / second')
    ureg.define('MGD = 1000000 * gallon / day')
    ureg.define('IMGD = 1000000 * imperial_gallon / day')
    ureg.define('AFD =  acre_feet / day')
    ureg.define('LPS = liter / second = lps')
    ureg.define('LPM =  liter / minute')
    ureg.define('MLD =  1000000 * liter / day')
    ureg.define('CMD =  cubic_meter / day')

    if hydraulic_param in ['demand','flow']:
        leg1 = ureg.Quantity(values, from_unit)
    else:
        if (from_unit in us_flow_units and to_unit in us_flow_units)\
            or (from_unit in si_flow_units and to_unit in si_flow_units):
            return values

        if hydraulic_param == 'pressure':
            leg1_punit = 'psi' if from_unit in us_flow_units else 'meter_H2O'
            leg1 = ureg.Quantity(values,leg1_punit)
        elif hydraulic_param == 'head':
            leg1_punit = 'feet_H2O' if from_unit in us_flow_units else 'meter_H2O'
            leg1 = ureg.Quantity(values,leg1_punit)
        elif hydraulic_param == 'velocity':
            leg1_punit = 'fps' if from_unit in us_flow_units else 'mps'
            leg1 = ureg.Quantity(values,leg1_punit)


    if hydraulic_param in ['demand','flow']:
        leg2 = leg1.to(to_unit)
    elif hydraulic_param == 'pressure':
        leg2_punit = 'psi' if to_unit in us_flow_units else 'meter_H2O'
        leg2 =  leg1.to(leg2_punit)  
    elif hydraulic_param == 'head':
        leg2_punit = 'feet_H2O' if to_unit in us_flow_units else 'meter_H2O'
        leg2 =  leg1.to(leg2_punit)  
    elif hydraulic_param == 'velocity':
        leg2_punit = 'fps' if to_unit in us_flow_units else 'mps'
        leg2 = leg1.to(leg2_punit)  

    
    return leg2.magnitude
    #return leg2

class RaggedArrayList(object):
    def stack_ragged(self,array_list, axis = 1):
        lengths = [arr.shape[axis] for arr in array_list ]
        idx = np.cumsum(lengths[:-1])
        stacked = np.concatenate(array_list,axis) 
        return stacked, idx, lengths

    def __init__(self, array_list,axis = 1) -> None:
        self.axis = axis
        if array_list:
            self._stacked_array, self._indices, self._lengths = self.stack_ragged(array_list,axis=self.axis)
        else:
            self._stacked_array = None
            self._indices = None
            self._lengths = []

    def split(self):
        return np.split(self._stacked_array,self._indices,axis=1) if  len(self._indices)>0  else [self._stacked_array]
    
    def __len__(self):
        if self._lengths:
            return self._indices[-1] + self._lengths[-1]
        else:
            return 0 

    def __getitem__(self,index):
        assert index < len(self._lengths)
        cur_length  = self._lengths[index]

        if index < 0:
            index = len(self._lengths) + index


        if  index < len(self._lengths) -1:
            next_idx     = self._indices[index] 
            if self.axis == 0:
                return self._stacked_array[ next_idx - cur_length: next_idx]
            else:
                return self._stacked_array[:, next_idx - cur_length: next_idx]
        else:
            if self.axis == 0:
                return self._stacked_array[-cur_length:]
            else:
                return self._stacked_array[:,-cur_length:]

    def __setitem__(self,index,value):
        assert index < len(self._lengths)
        cur_length = self._lengths[index]
        
        if index < 0:
            index = len(self._lengths) + index

        if  index < len(self._lengths) -1:
            next_idx     = self._indices[index] 
            
            if self.axis == 0:
                post_segment = self._stacked_array[next_idx:]
                prev_segment = self._stacked_array[:next_idx - cur_length]
            else:
                post_segment = self._stacked_array[:,next_idx:]
                prev_segment = self._stacked_array[:,:next_idx - cur_length]
            self._stacked_array = np.concatenate([prev_segment, value, post_segment],axis=self.axis)
            
            self._lengths[index] = value.shape[self.axis]
            self._indices = np.cumsum(self._lengths[:-1])
        else:
            self.pop()
            self.append(value)

    def append(self,new_array):
        if self._lengths:
            assert self._stacked_array.shape[self.axis-1] == new_array.shape[self.axis-1]
            new_length = new_array.shape[self.axis]
            self._indices = np.append(self._indices,self._indices[-1] + self._lengths[-1])
            self._lengths.append(new_length)
            self._stacked_array = np.concatenate([self._stacked_array , new_array], axis=self.axis) #np.concatenate(, new_array)
        else:
            self._stacked_array, self._indices, self._lengths = self.stack_ragged([new_array],axis=self.axis)

    def pop(self):
        out_array = None
        if self._lengths:
            print(f'self._lengths = {self._lengths}')
            if len(self._lengths) == 1:
                out_array = self._stacked_array
                self._stacked_array = None
            else:
                if self.axis == 0 :
                    out_array = self._stacked_array[-self._lengths[-1]:]
                    self._stacked_array = self._stacked_array[:self._indices[-1]]
                else:
                    out_array = self._stacked_array[:,-self._lengths[-1]:]
                    self._stacked_array = self._stacked_array[:,:self._indices[-1]]
            self._indices = self._indices[:-1]
            self._lengths.pop()
        
        return out_array


class RaggedArrayDict(RaggedArrayList):
    def __init__(self, array_dict, axis=1) -> None:
        if array_dict:
            self._keys = list(array_dict.keys())
            array_list = list(array_dict.values())
            super().__init__(array_list, axis)
        else:
            self._keys =  []
            super().__init__(None,axis)

    @staticmethod
    def from_keylen_and_stackedarray(keylen_dict, stacked_array,axis=1):
        lengths = list(keylen_dict.values())
        indices = np.cumsum(lengths[:-1])
        ragged_tokens = np.split(stacked_array,indices,axis=axis) if len(indices)>0 else [stacked_array]
        feed_dict = {k: ragged_tokens[i] for i,k in enumerate(keylen_dict)}
        return RaggedArrayDict(feed_dict, axis=axis)
    
    @staticmethod
    def from_keylen_and_daskstackedarray(keylen_dict, stacked_array,axis=1):
        def da_split(stacked_array,indices,axis):
            splitted_arrays=[]
            start_index = 0
            for ind in indices:
                a = stacked_array[start_index:ind]
                splitted_arrays.append(a)
                start_index = ind
            a = stacked_array[start_index:]
            splitted_arrays.append(a)
            return splitted_arrays
        
        lengths = list(keylen_dict.values())
        indices = np.cumsum(lengths[:-1])
        ragged_tokens = da_split(stacked_array,indices,axis=axis) if len(indices)>0 else [stacked_array]
        feed_dict = {k: ragged_tokens[i] for i,k in enumerate(keylen_dict)}
        return RaggedArrayDict(feed_dict, axis=axis)

    @staticmethod
    def from_RaggedArrayDict(keys, stacked_array, indices , lengths, axis= 1 ):
        x = RaggedArrayDict(None,axis)
        x._keys = list(keys)
        x._stacked_array = np.array(stacked_array,dtype= stacked_array.dtype)
        x._indices = np.array(indices,dtype= indices.dtype)
        x._lengths = list(lengths)
        return x

    def split(self):
        tokens =  super().split()
        return tokens, self._keys

    def get_chunk(self, from_id, to_id):
        return self.from_RaggedArrayDict(self._keys, self._stacked_array[from_id:to_id], self._indices, self._lengths, axis = self.axis)
        
    def __getitem__(self,key):
        assert isinstance(key,str)
        if key in self._keys:
            return super().__getitem__(self._keys.index(key))
        else:
            return None
        
    def pop(self):
        out_array = super().pop()
        self._keys = self._keys[:-1]
        return out_array
    
    def __setitem__(self,key,value):
        if key not in self._keys:
            self._keys.append(key)
            super().append(value)
        else:
            index = self._keys.index(key) 
            super().__setitem__(index,value)

FlowUnits= { 0 :"CFS"   ,       #cubic feet / sec
             1 :"GPM"   ,       #gallons / min
             2 :"AFD"   ,         #acre-feet / day
             3 :"MGD"   ,       #million gallon / day    
             4 :"IMGD"  ,       #Imperial MGD 
             5 :"LPS"   ,       #liters / sec
             6 :"LPM"   ,       #liters / min
             7 :"MLD"  ,        #megaliters / day
             8 :"CMH"  ,        #cubic meters / hr 
             9 :"CMD"  }        #cubic meters / day)