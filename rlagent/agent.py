from .transition_models import StateTransitionModel

import torch
from torch import nn, optim
import copy
from collections import deque


import numpy as np
from numpy.random import default_rng
rng = default_rng()


class Agent:
    def __init__(self,**args):
        
        print(args)
        
        self.action_cnt = args['action_cnt'] #env.action_space.n
        self.feature_cnt = args['feature_cnt'] #env.observation_space.shape[0]
        
        self.criterion = nn.MSELoss()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.smoothing = args['smoothing'] if 'smoothing' in args else 0.5
        
        self.target_transition_network = args['transition_model'].to(self.device) if 'transition_model' in args else StateTransitionModel(self.feature_cnt,self.action_cnt).to(self.device)
        self.base_transition_network = copy.deepcopy(self.target_transition_network).to(self.device)
        self.lr = args['lr'] if 'lr' in args else 0.01
        self.optimizer = optim.Adam(self.target_transition_network.parameters(),lr=self.lr)
        
        self.alpha = args['alpha'] if 'alpha' in args else 1.0
        self.gamma = args['gamma'] if 'gamma' in args else 1.0
        self.batch_size = args['batch_size'] if 'batch_size' in args else 20
        self.batch_cnt = args['batch_cnt'] if 'batch_cnt' in args else 10
        self.memory_size = args['memory_size'] if 'memory_size' in args else 200
        
        self.memory = deque([],maxlen=self.memory_size)
        
        self.epsilon = args['epsilon'] if 'epsilon' in args else 1.0
        self.epsilon_decay = args['epsilon_decay'] if 'epsilon_decay' in args else 0.99
        self.epsilon_min = args['epsilon_min'] if 'epsilon_min' in args else 0.1

        self.weight_filepath = args['weight_filepath'] if 'weight_filepath' in args else ''
        
    
    def _update_base_network(self):
        """
        Double DQN uses a smoothing factor when updating base transition model. 
        Smoothing improves the model's training stability by 'smoothing' large variations in the target model weight update values  
        """
        
        base_params = self.base_transition_network.named_parameters()
        target_params = self.target_transition_network.named_parameters()
        
        blended_weights = {}
        for base_param,target_param in zip(base_params,target_params):
            blended_weights[base_param[0]] = self.smoothing*base_param[1].data+(1-self.smoothing)*target_param[1].data
        
        self.base_transition_network.load_state_dict(blended_weights)
    
    def _update_epsilon(self):
        if self.epsilon >self.epsilon_min:
            self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)       
        
    def get_action(self,s,use_epsilon_decay=True):
        if np.random.random() < self.epsilon and use_epsilon_decay:
            action = np.random.randint(0,self.action_cnt-1)
        else:
            action = torch.argmax(self.base_transition_network.forward(torch.from_numpy(s.astype(np.float32)).to(self.device))).item()            
        return action
    
    def append_replay(self,transition):
        self.memory.appendleft(transition)
    
    def save_weights(self,filepath=''):
        if filepath:
            torch.save(self.base_transition_network.state_dict(),filepath)
        elif self.weight_filepath:
            torch.save(self.base_transition_network.state_dict(),self.weight_filepath)
        else:
            print('Cannot save weights, no filepath provided or stored when initializing agent')

    def load_weights(self,filepath=''):
        if filepath:
            self.base_transition_network.load_state_dict(torch.load(filepath))
            self.target_transition_network.load_state_dict(torch.load(filepath))
        elif self.weight_filepath:
            self.base_transition_network.load_state_dict(torch.load(self.weight_filepath))
            self.target_transition_network.load_state_dict(torch.load(self.weight_filepath))
        else:
            print("Cannot load weights, no filepath provided or stored when initializing agent")


    def update_transition_model(self):

        if len(self.memory)<self.batch_size:
            return 0.0,0.0
        
        losses = []
        for _ in range(self.batch_cnt):
            self.optimizer.zero_grad()
            
            minibatch_ix = rng.choice(len(self.memory), size=self.batch_size, replace=False)

            state = torch.from_numpy(np.array([self.memory[v]['s'] for v in minibatch_ix]).astype(np.float32)).to(self.device)
            next_state = torch.from_numpy(np.array([self.memory[v]['s_prime'] for v in minibatch_ix]).astype(np.float32)).to(self.device)
            reward = np.array([self.memory[v]['r'] for v in minibatch_ix])
            action_ix = np.array([self.memory[v]['a'] for v in minibatch_ix])

            target = self.base_transition_network.forward(state)
            for i in range(len(action_ix)):
                target[i,action_ix[i]]=reward[i]

            future_transition = self.base_transition_network.forward(next_state)
            has_future_states = torch.reshape(torch.from_numpy(np.array([not self.memory[v]['is_done'] for v in minibatch_ix])),(-1,1)).to(self.device)
            discounted_reward = torch.max(future_transition*has_future_states,axis=1).values*self.gamma

            for i in range(len(action_ix)):
                target[i,action_ix[i]]+=discounted_reward[i]

            # Backprop Target Network
            prediction = self.target_transition_network.forward(state)            
            loss = self.criterion(prediction,target)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            
        self._update_epsilon()
        self._update_base_network()
        return np.mean(losses),np.std(losses)