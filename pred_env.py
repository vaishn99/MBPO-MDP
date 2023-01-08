import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable

import math

from blackhc.mdp import dsl
from blackhc import mdp
import time

from blackhc.mdp import lp
import functools
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from numpy import random

from operator import itemgetter

from collections import defaultdict


##############################################



##############################################


class pred_env:
    # initialize
    def __init__(self,horizon_length,k):
        
        self.prev_state =None
        self.curr_state =None
           
        self.state_list=list()
        self.action_list=list()
        self.state_to_action_map=dict()

        
        self.P=defaultdict()
        self.R=defaultdict()
        
        self.horizon_len=horizon_length
        self.terminal_state=None
        self.k=k
    
    def reset(self):
        
        self.state_list=list()
        self.action_list=list()
        self.state_to_action_map=dict()

        
        self.P=defaultdict()
        self.R=defaultdict()
        
    # Parameter Estimation 
    
    def update_param_given_epi(self,D_real):
        
        episodes=D_real.buffer
        

        # following SARSA format
        for epi_id in range(len(episodes)):
    
            t_states, t_actions, t_rewards,t_nstates,t_log_probs = self.cvt_axis(episodes[epi_id])
            i=0 
            while i<len(t_states):
                
                # updating the list of states
                
                if any([torch.equal(x,t_states[i]) for x in self.state_list])!=True:
                    self.state_list.append(t_states[i])
                    self.state_to_action_map.update({t_states[i]:[]})
                
                
                
                # updating the list of actions
            
            

                if any([torch.equal(x,t_actions[i]) for x in self.action_list])!=True:
                    self.action_list.append(t_actions[i])
                    
                if any([torch.equal(x,t_actions[i]) for x in self.state_to_action_map[self.smooth_check(self.state_to_action_map,t_states[i])]])!=True:
                    self.state_to_action_map[self.smooth_check(self.state_to_action_map,t_states[i])].append(t_actions[i])
                
                # # update state,action to next state count map
                
                tru_tup,flag=self.double_smooth_check(self.P,(t_states[i],t_actions[i]))
                if flag!=True:
                    self.P[(t_states[i],t_actions[i])]={t_nstates[i]:1}
                    self.R[(t_states[i],t_actions[i])]={t_nstates[i]:t_rewards[i]}
                    
                else:
                    if any([torch.equal(x,t_nstates[i]) for x in self.P[tru_tup].keys()])!=True:
                        self.P[tru_tup].update({t_nstates[i]:1})
                        self.R[tru_tup].update({t_nstates[i]:t_rewards[i]})
                        
                    else:
                        sec_tup=self.smooth_check(self.P[tru_tup],t_nstates[i])
                        self.P[tru_tup][sec_tup]+=1
                                    
                i+=1  
                
            if self.terminal_state is None and i<self.horizon_len:
                self.terminal_state=t_nstates[i-1]
                self.state_list.append(t_nstates[i-1])
        return 
    
    # Support functions
    def double_smooth_check(self,A,a):
        
        for ele in A.keys():
            if torch.equal(a[0],ele[0]) and torch.equal(a[1],ele[1]):
                return ele,True
        return a,False
    
    def smooth_check(self,A,a):

        for ele in A.keys():
            if torch.eq(a,ele).all():
                return ele
        return a
     
    def cvt_axis(self,traj):
        
        t_states =[]
        t_actions =[]
        t_nstates =[]
        t_rewards=[]
        t_log_probs=[]
        
        for i in range(len(traj[0])):
            t_states.append(traj[0][i])
            t_actions.append(traj[1][i])
            t_rewards.append(traj[2][i])
            t_nstates.append(traj[3][i])
            t_log_probs.append(traj[4][i])

        return (t_states, t_actions, t_rewards,t_nstates,t_log_probs) 
    
    def get_parameters(self):
        print("\nState list")
        print(self.state_list)
        print("\nAction list")
        print(self.action_list)
        print("\nState to action map")
        print(self.state_to_action_map)
        print("\nstate_action to next state")
        for x in self.P:
            print(x)
            print(self.P[x])
        print("\n state_action to reward map")
        for x in self.R:
            print(x)
            print(self.R[x])
        return
    
    def Is_terminal_state(self,s_t):
        if torch.equal(self.terminal_state,s_t):
            return True
        return False
    
    def set_start_state(self):
        if len(self.state_list)>0:
            
            p=[1]*len(self.state_list)
            p=[x/len(self.state_list) for x in p]
            s_t_index=np.random.choice(np.arange(len(self.state_list)),p=p)
            s_t=self.state_list[s_t_index]
            while torch.equal(s_t,self.terminal_state):
                s_t_index=np.random.choice(np.arange(len(self.state_list)),p=p)
                s_t=self.state_list[s_t_index]
            self.curr_state=s_t
        return
    
    def list_check_up(self,A,s_t):
        for x in A:
            if torch.equal(x,s_t):
                return x
        return None
   
    # Fake Data generation functions
    
    def step_v1(self,a_t):
        
        next_state=0
        un_norm_distr=self.P[self.double_smooth_check(self.P,(self.curr_state,a_t))[0]]
        norm_factor=sum(list(un_norm_distr.values()))
        choices=list(un_norm_distr.keys())
        p=[x/norm_factor for x in un_norm_distr.values()]
        
        next_state_id=np.random.choice(np.arange(len(choices)),p=p)
        next_state=choices[next_state_id]
        rew_dict=self.R[self.double_smooth_check(self.R,(self.curr_state,a_t))[0]]
        next_state_repr=None
        for x in rew_dict.keys():
            if torch.equal(x,next_state):
                next_state_repr=x
                break
        reward=rew_dict[next_state_repr]
        
        Is_done=False
        if self.Is_terminal_state(next_state):
            Is_done=True

        return next_state,reward,Is_done,None
# sample a state from D_real

