#
# Created on Wed Aug 09 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: Measure inference time
# Ref: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
# ------------------------------
#
import torch
import numpy as np
from typing import Callable
class Timer():
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.starter, self.ender = torch.cuda.Event(enable_timing=True),  torch.cuda.Event(enable_timing=True)
        self.timings = []
        self.num_graphs = []
        self.finished_warmup=False

    def auto_measure(self, inference_func: Callable, num_graphs_per_batch: int, gpu_warmup_times: int =10)-> Callable:
       
        def inference( *args, **kwargs):
            
            if gpu_warmup_times > 0 and not self.finished_warmup:
                for _ in range(gpu_warmup_times):
                    _ = inference_func(*args, **kwargs)
                self.finished_warmup = True

            self.starter.record()
            results = inference_func(*args, **kwargs)
            self.ender.record()
            torch.cuda.synchronize()
            curr_time = self.starter.elapsed_time(self.ender)

            self.timings.append(curr_time)
            self.num_graphs.append(num_graphs_per_batch)
            return results
        
        return inference

    def compute_time(self, len_dataset: int) ->float:
        assert len(self.timings) == len(self.num_graphs)
        assert len_dataset > 0

        total_time = np.array(self.timings).dot(np.array(self.num_graphs))
        
        mean_time = (total_time / len_dataset).astype(float)

        return mean_time
    
    def compute_throughput(self,len_dataset: int) ->float:
        assert len(self.timings) == len(self.num_graphs)
        assert len_dataset > 0


        total_times = np.array(self.timings) * np.array(self.num_graphs) 
        total_times = total_times / len_dataset 
        
        total_times = total_times / 1000
        total_time = np.sum(total_times).astype(float)

        throughput = float(len(self.timings) * max(self.num_graphs)) / total_time

        return throughput
