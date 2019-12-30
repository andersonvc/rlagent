import torch
import random

class MemoryBuffer():
    def __init__(self,max_size,feature_cnt):
        self.max_size = max_size

        self.state_buffer=torch.zeros((max_size,feature_cnt)).float()
        self.sprime_buffer=torch.zeros((max_size,feature_cnt)).float()
        self.action_buffer=torch.zeros(max_size).long()
        self.reward_buffer=torch.zeros(max_size).float()
        self.done_buffer=torch.zeros(max_size).bool()

        self.start_idx = 0
        self.end_idx = max_size
        self.curr_idx = 0
        self.is_full = False
    
    def append(self,s,sprime,action,reward,isdone):

        self.state_buffer[self.curr_idx]=torch.Tensor(s)
        self.sprime_buffer[self.curr_idx]=torch.Tensor(sprime)
        self.action_buffer[self.curr_idx]=action
        self.reward_buffer[self.curr_idx]=reward
        self.done_buffer[self.curr_idx]=isdone

        self.curr_idx = (self.curr_idx+1)%self.end_idx
        if not self.is_full and self.curr_idx == self.max_size-1:
            self.is_full = True
    
    def sample(self,sample_cnt):
        
        max_idx = self.max_size if self.is_full else self.curr_idx
        sample_indices = random.sample(range(max_idx),sample_cnt)

        return (self.state_buffer[sample_indices],
                self.sprime_buffer[sample_indices],
                self.action_buffer[sample_indices],
                self.reward_buffer[sample_indices],
                self.done_buffer[sample_indices])
    
    def is_ready(self,sample_cnt):

        # Dont sample if requested sample cnt > total samples in memory
        if self.is_full and sample_cnt>self.max_size:
            return False
        
        elif not self.is_full and sample_cnt>self.curr_idx:
            return False
        
        return True