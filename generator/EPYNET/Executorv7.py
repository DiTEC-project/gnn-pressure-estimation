#
# Created on Thu May 04 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: generate a synthesis snapshot given tokens
# Version: 7.0
# ------------------------------
#

import glob
import os
import ray
from epynet import Network
from copy import deepcopy
import numpy as np
from epynet import epanet2

import wntr
import wntr.epanet.util as wutils
import networkx as nx
import epynet_utils as eutils
from TokenGeneratorByRange import ParamEnum

class WDNExecutor(object):

    def __init__(self,featlen_dict, config, valve_type_dict, args,  wn=None):
        self.sort_node_name         = False
        self.min_valve_setting      = 1e-4
        self.ordered                = False
        wn_to_copy                  = wn
        self.custom_base_index      = 100
       
        self.featlen_dict           = deepcopy(featlen_dict)
        self.config                 = config
        #self.valve_type_dict        = valve_type_dict
        self.ele_std                = args.ele_std
        self.ele_kmean_init         = args.ele_kmean_init
        self.update_elevation_method= args.update_elevation_method

        self.expected_attr          = args.att.strip().split(',')
        self.pressure_upperbound    = args.pressure_upperbound
        self.pressure_lowerbound    = args.pressure_lowerbound
        self.flowrate_threshold     = args.flowrate_threshold
        self.init_valve_state       = args.init_valve_state
        self.init_pipe_state        = args.init_pipe_state
        self.accept_warning_code    = args.accept_warning_code    
        

        self.min_diameter           = self.config.getfloat('pipe','diameter_lo') 
        wn_inp_path                 = self.config.get('general','wn_inp_path')

        
        self.skip_nodes             = self.config.get('general','skip_nodes').strip().split(',')  if self.config.has_option('general', 'skip_nodes') else []  
        self.skip_links             = self.config.get('general','skip_links').strip().split(',')  if self.config.has_option('general', 'skip_links') else []  
   
        ###################################
        self.gen_demand             = args.gen_demand
        self.gen_elevation          = args.gen_elevation
        self.gen_diameter           = args.gen_diameter
        self.gen_pipe_roughness     = args.gen_roughness            #False#True
        self.gen_valve_init_status  = args.gen_valve_init_status    #False#True
        self.gen_valve_setting      = args.gen_valve_setting 
        
        self.gen_pump_init_status   = args.gen_pump_init_status 
        self.gen_pump_speed         = args.gen_pump_speed 
        self.gen_tank_level         = args.gen_tank_level 
        self.gen_res_total_head     = args.gen_res_total_head

        self.gen_tank_elevation     = args.gen_tank_elevation
        self.gen_tank_diameter      = args.gen_tank_diameter
        self.gen_pipe_length        = args.gen_length
        self.gen_pipe_minorloss     = args.gen_minorloss
        self.gen_pump_length        = args.gen_pump_length
        self.gen_valve_diameter     = args.gen_valve_diameter


        self.replace_nonzero_basedmd= args.replace_nonzero_basedmd
        self.update_totalhead_method= args.update_totalhead_method
        self.mean_cv_threshold      = args.mean_cv_threshold #5. #10. #args.mean_group_var_threshold #10.0
        self.neighbor_std_threshold = args.neighbor_std_threshold
        self.allow_error            = args.allow_error
        self.debug                  = args.debug

        self.remove_control         = args.remove_control 
        self.remove_rule            = args.remove_rule 
        self.remove_pattern         = args.remove_pattern 
        self.convert_results_by_flow_unit = args.convert_results_by_flow_unit
        ##################################
        if wn_to_copy: 
            self.wn = wn_to_copy 
        else:
            self.wn = Network(wn_inp_path)

        self.flow_unit = wutils.FlowUnits(self.wn.ep.ENgetflowunits()) 

        if args.skip_resevoir_result:
            self.skip_nodes.extend(self.wn.reservoirs.uid.to_list())

        if self.remove_pattern:
            patterns = self.wn.patterns
            if len(patterns)> 0 :
                for p in patterns:
                    uid = p.uid
                    eutils.ENdeletepattern(self.wn,uid)
             
        if self.remove_rule:
            eutils.ENdeleteallrules(self.wn)
           
        if self.remove_control:
            eutils.ENdeleteallcontrols(self.wn)

        wntr_wn = wntr.network.WaterNetworkModel(wn_inp_path)
        self.wn_g = wntr_wn.get_graph().to_undirected()
        self.ori_diameters = [p.diameter for p in self.wn.pipes]
        self.ori_elevations = [p.elevation for p in self.wn.junctions]

        
        #Headloss formula enum: 0-HW, 1-DW, 2-CM
        self.head_loss_type = wntr_wn.options.hydraulic.headloss#EN_HEADLOSSFORM = 7 


        patterns = self.wn.patterns
        for i,_ in enumerate(self.wn.junctions):
            if not str(self.custom_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_base_index + i),values=[0])
        
        self.custom_res_pattern_base_index = self.custom_base_index + len(self.wn.junctions)
        for i,_  in enumerate(self.wn.reservoirs):
            if not str(self.custom_res_pattern_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_res_pattern_base_index + i) ,values=[0])


        
        # curves = self.wn.curves
        # self.custom_pump_curve_base_index = self.custom_res_pattern_base_index + len(self.wn.reservoirs)
        # for i,pump  in enumerate(self.wn.pumps):
            
        #     if not str(self.custom_pump_curve_base_index + i) in curves:
        #         self.wn.add_curve(str(self.custom_pump_curve_base_index + i) ,values=[[0,0]])
        #         c_index= self.wn.ep.ENgetcurveindex(str( self.custom_res_pattern_base_index + i))
        #         self.wn.ep.ENsetheadcurveindex(pump.index,c_index )
        


    def filter_skip_elements(self,df, skip_list):
        mask = df.index.isin(skip_list)
        ret = df.loc[np.invert(mask)]
        return ret

    
    def split_token_to_features(self,t,featlen_dict,axis=1):
        features=[]
        start = 0
        for length in featlen_dict.values():
            end = start + length
            if axis == 0:
                features.append(t[start:end] if length > 0 else None)
            else:
                features.append(t[:,start:end] if length > 0 else None)
            start+=length
        return features
    
    
    def epynet_simulate2(self,
                         tokens,
                         scene_id):
        """EN_DURATION 0 Simulation duration
        EN_HYDSTEP 1 Hydraulic time step
        EN_QUALSTEP 2 Water quality time step
        EN_PATTERNSTEP 3 Time pattern time step
        EN_PATTERNSTART 4 Time pattern start time
        EN_REPORTSTEP 5 Reporting time step
        EN_REPORTSTART 6 Report starting time
        EN_RULESTEP 7 Time step for evaluating rule-based controls
        EN_STATISTIC 8 Type of time series post-processing to use:
                                    EN_NONE (0) = none
                                    EN_AVERAGE (1) = averaged
                                    EN_MINIMUM (2) = minimums
                                    EN_MAXIMUM (3) = maximums
                                    EN_RANGE (4) = ranges

        :param _type_ tokens: _description_
        :param _type_ scene_id: _description_
        """
        self.wn.reset()
        

        ragged_tokens = eutils.RaggedArrayDict.from_keylen_and_stackedarray(self.featlen_dict,tokens,axis=0)

        junc_demands        = ragged_tokens[ParamEnum.JUNC_DEMAND]
        junc_elevations     = ragged_tokens[ParamEnum.JUNC_ELEVATION]
        pump_statuses       = ragged_tokens[ParamEnum.PUMP_STATUS]
        pump_speed          = ragged_tokens[ParamEnum.PUMP_SPEED]
        pump_lengths        = ragged_tokens[ParamEnum.PUMP_LENGTH]
        pipe_roughness      = ragged_tokens[ParamEnum.PIPE_ROUGHNESS]
        pipe_lengths        = ragged_tokens[ParamEnum.PIPE_LENGTH]
        pipe_minorlosses    = ragged_tokens[ParamEnum.PIPE_MINORLOSS]
        pipe_diameters      = ragged_tokens[ParamEnum.PIPE_DIAMETER]
        tank_elevations     = ragged_tokens[ParamEnum.TANK_ELEVATION]
        tank_diameters      = ragged_tokens[ParamEnum.TANK_DIAMETER]
        tank_levels         = ragged_tokens[ParamEnum.TANK_LEVEL]
        valve_statuses      = ragged_tokens[ParamEnum.VALVE_STATUS]
        valve_settings      = ragged_tokens[ParamEnum.VALVE_SETTING]
        valve_diameters     = ragged_tokens[ParamEnum.VALVE_DIAMETER]
        res_heads           = ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD]

        self.wn.ep.ENsettimeparam(epanet2.EN_DURATION,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTEP,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTART,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTART,1)
        self.wn.ep.ENsettimeparam(epanet2.EN_RULESTEP,1)
        
       
        
        support_node_attr_keys = ['demand','head','pressure']
        support_link_attr_keys = ['velocity','flow'] #'flowrate','status',
        
      
        for i,junc in enumerate(self.wn.junctions):
            if self.gen_demand:
                #this only affects if we have zero/one demand category
                if not self.replace_nonzero_basedmd or (junc.basedemand != 0 and self.replace_nonzero_basedmd):
                    junc.basedemand = 1.0
                    
                junc.pattern = str(self.custom_base_index + i)
                junc.pattern.values = [junc_demands[i]]
                #In EPANET <=2.2, we have no way to delete the demand category if there are more than 1 exists...
                #Thus, we copy the base_demand and pattern into each demand category
                eutils.ENsetdemandpatterntoallcategories(self.wn, junc.index, junc.basedemand,  junc.pattern.index)
            
            #static features  
            if self.gen_elevation:
                junc.elevation = junc_elevations[i]
        
       
        for i,pump in enumerate(self.wn.pumps):
            # if self.remove_pattern: 
            #     pump.curve.values = np.zeros_like(pump.curve.values).tolist()
                
            if self.gen_pump_init_status:
                pump.initstatus =int(pump_statuses[i])
                #print(f'pump {pump.uid} expected status = {bool(pump_statuses[i])} | actual status = {pump.initstatus}')
                #eutils.set_object_value_wo_ierror(pump,epanet2.EN_INITSTATUS,  bool(pump_statuses[i]) )

            if self.gen_pump_speed:
                pump.speed = pump_speed[i] 

            if self.gen_pump_length:
                eutils.set_object_value_wo_ierror(pump, epanet2.EN_LENGTH,  pump_lengths[i] )


        for i,tank in enumerate(self.wn.tanks):
            if self.gen_tank_level:
                tank_level =tank_levels[i] #tank.minlevel + tank_levels[i] * (tank.maxlevel - tank.minlevel)
                eutils.set_object_value_wo_ierror(obj=tank, code = epanet2.EN_TANKLEVEL, value= tank_level)
            if self.gen_tank_elevation:
                eutils.set_object_value_wo_ierror(obj=tank, code = epanet2.EN_ELEVATION, value= tank_elevations[i])
            if self.gen_tank_diameter:
                eutils.set_object_value_wo_ierror(obj=tank, code = epanet2.EN_TANKDIAM, value= tank_diameters[i])
        
        debug_flag = False

        tmp_graph = eutils.get_networkx_graph(wn=self.wn, include_reservoir=True,graph_type='undirected')
        for i,valve in enumerate(self.wn.valves):
            if self.init_valve_state is not None:
                valve.initstatus = int(self.init_valve_state) 
            
            if self.gen_valve_init_status:
                if self.init_valve_state is not None and self.debug and not debug_flag:
                    print(f'WARN! init value state is overrided')
                    debug_flag = True
                if not bool(valve_statuses[i]):
                    tmp_graph.remove_edge(valve.from_node.uid, valve.to_node.uid)
                    if nx.is_connected(tmp_graph):
                        valve.initstatus = int(valve_statuses[i]) 
                    else:
                        if self.debug:
                            print(f'WARN! Unable to off valve {valve.uid} due to the graph disconnection')
                        tmp_graph.add_edge(valve.from_node.uid, valve.to_node.uid)
                        valve.initstatus = True
                else:
                    valve.initstatus = int(valve_statuses[i]) 
               
            if self.gen_valve_setting: # open
                #scaled_valve_setting  =  self.valve_type_dict[valve.valve_type][0] +  valve_settings[i] * (self.valve_type_dict[valve.valve_type][1] - self.valve_type_dict[valve.valve_type][0])
                #valve.setting = max(self.min_valve_setting, scaled_valve_setting) 
                if valve_settings[i] > 0: #0 mean unused
                    #valve.setting = valve_settings[i]
                    eutils.set_object_value_wo_ierror(obj=valve, code = epanet2.EN_INITSETTING, value=valve_settings[i])   

            if self.gen_valve_diameter:
                eutils.set_object_value_wo_ierror(obj=valve, code = epanet2.EN_DIAMETER, value=np.maximum(valve_diameters[i],1e-12) )   

                
        pipe_uids = []
        for i,pipe in enumerate(self.wn.pipes):
            #if  self.init_pipe_state is not None and not pipe.check_valve:
            if  self.init_pipe_state is not None and not pipe.check_valve:
                    #pipe.initstatus = self.init_pipe_state 
                    eutils.set_object_value_wo_ierror(obj=pipe, code = epanet2.EN_INITSTATUS, value= int(self.init_pipe_state) )
                    pipe_uids.append(pipe.uid)
            if self.gen_pipe_roughness:
                pipe.roughness = pipe_roughness[i]

            if self.gen_pipe_length:
                eutils.set_object_value_wo_ierror(obj=pipe, code = epanet2.EN_LENGTH, value= np.maximum(pipe_lengths[i],1e-12))
            if self.gen_pipe_minorloss:
                eutils.set_object_value_wo_ierror(obj=pipe, code = epanet2.EN_MINORLOSS, value= np.maximum(pipe_minorlosses[i],1e-12))   

            if self.gen_diameter:
                #pipe.diameter =  pipe_diameters[i] 
                            
                eutils.set_object_value_wo_ierror(obj=pipe, code = epanet2.EN_DIAMETER, value=np.maximum(pipe_diameters[i],1e-12) )   

                #new_diameter = self.ori_diameters[i] +  pipe_diameters[i] 
                #if new_diameter > 10:
                #    pipe.diameter = new_diameter  


        

       

        #total head should be the last element                  
        if self.gen_res_total_head:
            for i,res in enumerate(self.wn.reservoirs):
               
                res.set_object_value(epanet2.EN_ELEVATION,1.0)
                if  self.update_totalhead_method is None:
                    tmp = res_heads[i]
                elif self.update_totalhead_method  == 'add_max_elevation':
                    elevations =[n.elevation for n in self.wn.junctions] #[n.elevation for n in self.wn.nodes]# [self.wn.nodes[n].elevation for n in neighbors]
                    tmp = max(elevations) +  res_heads[i]
                
                p_index= self.wn.ep.ENgetpatternindex(str( self.custom_res_pattern_base_index + i))
                self.wn.ep.ENsetpattern(p_index,[tmp])
                res.set_object_value(epanet2.EN_PATTERN,p_index)
        
        
        
        sim_results = {}

        prefix_name = 'tmp_'+str(scene_id)

        for file in glob.glob(f"{prefix_name}.*"):
            os.remove(file)

        #RUN SIM
        def ENrunH(ep):
            """Runs a single period hydraulic analysis, 
            retrieving the current simulation clock time t"""
            ierr= ep._lib.EN_runH(ep.ph, epanet2.ctypes.byref(ep._current_simulation_time))
            #if ierr>=100: 
            #    raise epanet2.ENtoolkitError(ep, ierr)
            return ierr

        def solve_return_error(wn,simtime=0):
            if wn.solved and wn.solved_for_simtime == simtime:
                return
            #wn.reset()
            wn.ep.ENsettimeparam(4, simtime)
            wn.ep.ENopenH()
            wn.ep.ENinitH(0)
            code = ENrunH(wn.ep)
            assert code is not None
            wn.ep.ENcloseH()
            wn.solved = True
            wn.solved_for_simtime = simtime
            return code
        
        code = solve_return_error(self.wn)
        
        #out_feature_lens
        #skipped_uids =  self.filter_skip_elements(self.wn.nodes.uid,self.skip_nodes)
        #print(f'before_skip-pressure_results shape  = {self.wn.nodes.pressure.shape}')
        
        if  self.skip_nodes is not None:
            pressure_df = self.wn.nodes.pressure
            pressure_results = self.filter_skip_elements(pressure_df,self.skip_nodes).values 
            pressure_results = np.reshape(pressure_results,[1,-1])
        else:
            pressure_results = self.wn.nodes.pressure.values
            pressure_results = np.reshape(pressure_results,[1,-1])
        
        if self.convert_results_by_flow_unit is not None:
            from_unit = eutils.FlowUnits[ self.wn.ep.ENgetflowunits()]
            to_unit = self.convert_results_by_flow_unit
            
            if from_unit != to_unit:
                pressure_results = eutils.ENconvert(from_unit=from_unit,
                                                to_unit=to_unit,
                                                hydraulic_param='pressure',
                                                values=pressure_results)
           
            
        is_nan = np.isnan(pressure_results).any()
        if is_nan and self.debug:
            print('is nan')
        error = is_nan

        if code >0:
            if self.accept_warning_code:
                error = error or code > 6
            else:
                error = error or code > 0
            if self.debug:
                print(f'Detected abnormal code- code {code}')

        if self.pressure_lowerbound is not None:
            negative_error = any(pressure_results.min(axis=1) < self.pressure_lowerbound)  
            if negative_error and self.debug:
                print(f'negative_error') # at { skipped_uids[np.argmin(pressure_results,axis=1)].to_list()  }
            error = error or negative_error
     
        if self.pressure_upperbound is not None:
            extreme_error = any(pressure_results.max(axis=1) > self.pressure_upperbound)  
            if extreme_error and self.debug:
                print(f'extreme_error ')#{ skipped_uids[np.argmax(pressure_results,axis=1)].to_list()  }
            error = error or extreme_error

        if self.neighbor_std_threshold is not None:
            
            hop = 2
            uids = np.array(list(self.wn.nodes.uid))
            tmp = self.wn.nodes.pressure.values
            def get_neighbor_std(node):
                test_neighbor_uids =  list(set(nx.single_source_shortest_path_length(self.wn_g , node.uid, cutoff=hop).keys()).difference([node.uid]))
                neighbor_ids = np.where(np.isin(uids,test_neighbor_uids))[0]
                neighbor_values = np.take(tmp,neighbor_ids)
                neighbor_variance = np.std(neighbor_values)
                return neighbor_variance

            #neighbor_variances = []
            #for _,node in enumerate(self.wn.nodes) :
            #    neighbor_variance = get_neighbor_variance(node)
            #    neighbor_variances.append(neighbor_variance)
            neighbor_stds = list(map(get_neighbor_std,self.wn.nodes))

            neighbor_stds = np.array(neighbor_stds)
            mean_neighbor_stds = np.mean(neighbor_stds)
            error = error or mean_neighbor_stds > self.neighbor_std_threshold
            
            print(f' neighbor std = {mean_neighbor_stds}')
            if  mean_neighbor_stds > self.neighbor_std_threshold and self.debug:
                print(f'high neighbor std = {mean_neighbor_stds}')
            

        if self.mean_cv_threshold is not None:
            tmp = pressure_results #self.wn.nodes.pressure.values
            cv = float(np.var(tmp,axis=1) / np.mean(tmp,axis=1))
            error = error or cv > self.mean_cv_threshold
            if  cv > self.mean_cv_threshold and self.debug:
                print(f'too high cv = {cv}')

        #flowrate_results = results.link['flowrate']
        #error = error | any(flowrate_results.min(axis=1) < self.flowrate_threshold) 
        sim_result_indices = None
        for attr in self.expected_attr: 
            if attr in support_node_attr_keys:
                sim_result = getattr(self.wn.nodes,attr) if hasattr(self.wn.nodes,attr) else getattr(self.wn.junctions,attr)
                if  self.skip_nodes is not None:
                    sim_result = self.filter_skip_elements(sim_result,self.skip_nodes)  
                   
            elif attr in support_link_attr_keys:
                sim_result = getattr(self.wn.links,attr)
                if  self.skip_links is not None:
                    sim_result =  self.filter_skip_elements(sim_result,self.skip_links)  
                
            if self.sort_node_name:
                sim_result = sim_result.sort_index(axis=1)

            sim_result_indices = sim_result.index.tolist()
            sim_result = np.reshape(sim_result.to_numpy(),[1,-1])
            if self.convert_results_by_flow_unit is not None:
                from_unit = eutils.FlowUnits[ self.wn.ep.ENgetflowunits()]
                to_unit = self.convert_results_by_flow_unit
                
                if from_unit != to_unit:
                    if self.debug:
                        print(f'detected unit convension from {from_unit} to {to_unit}...')
                    sim_result = eutils.ENconvert(from_unit=from_unit,
                                                    to_unit=to_unit,
                                                    hydraulic_param=attr,
                                                    values=sim_result)


            

            sim_results[attr] = sim_result
        
        #debug
        if sim_results['pressure'] is None or len(sim_results['pressure']) <= 0:
            print('weriddd')
        return sim_results, error, sim_result_indices


    def update_batch_dict(self,batch_dict, single_dict):
            for key,value in single_dict.items():
                if key not in batch_dict:
                    batch_dict[key] = value
                else:
                    batch_dict[key] = np.concatenate([batch_dict[key],value],axis=0)
            return batch_dict

    def check_order(self,l1,l2):
        if len(l1) != len(l2):
            return False
        else:
            for i in range(len(l1)):
                if l1[i] != l2[i]:
                    return False
            return True

    def simulate(self,batch_tokens,scence_ids):
        batch_results = {}
        batch_size = batch_tokens.shape[0]
        do_saved = False
        stored_ordered_name_list=None
        for id in range(batch_size):
            tokens = batch_tokens[id]
            single_result,error,ordered_name_list = self.epynet_simulate2(tokens,scence_ids[id]) #self.epynet_simulate2(tokens,scence_ids[id])
            
            if stored_ordered_name_list is not None:
                assert self.check_order(ordered_name_list,stored_ordered_name_list)
            stored_ordered_name_list = ordered_name_list
            
            if not error or self.allow_error:
                batch_results = self.update_batch_dict(batch_results,single_result)
                
                #if not do_saved:
                #    do_saved=True
                #    self.wn.ep.ENsaveinpfile(f'{scence_ids[id]}.inp')
        return batch_results, stored_ordered_name_list
    
@ray.remote(num_cpus=0)
class WDNRayExecutor(WDNExecutor):
    pass
