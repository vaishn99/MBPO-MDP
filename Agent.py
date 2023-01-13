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

#############################

from utils import Network,init_weights,ReplayMemory
from pred_env import pred_env

#############################



class Agent():

    def __init__(self,observation_space,action_space,gamma=0.99,learning_rate=1e-3,horizon_len=20,k=10,fraction_of_real=0.5,batch_size=200):

        self.model = Network(observation_space.n,action_space.n)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        self.horizon_len=horizon_len # Assuming we already know the horizon length
        self.env_model =pred_env(self.horizon_len,k)
        self.D_fake=ReplayMemory(capacity=10000)
        self.fraction_of_real=fraction_of_real
        self.batch_size=batch_size


    def init_env_model(self):
        self.env_model.set_start_state()
        self.D_fake.flush_all()
    def reset(self):
        # init_weights(self.model)
        self.env_model.reset()
        self.D_fake.flush_all()

    def action(self, state):
        
        probs = self.model(Variable(state))
        action = probs.multinomial(1).data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()

        return(action, log_prob)
    
    def update_D_fake(self,num_of_epi,start_state=None):
        
    
        if start_state is None:
            s_t=self.env_model.curr_state
        else:
            s_t=start_state

        self.env_model.curr_state=s_t
        
        
        result=[]
        

        trajs=[]

        for traj_id in range(num_of_epi):
            
            if self.env_model.Is_terminal_state(self.env_model.curr_state):
                self.env_model.set_start_state()
            s_t=self.env_model.curr_state
            
            states=[]
            log_probs=[]
            rewards=[]
            actions=[]
            nstates=[]
            
            for t in range(self.env_model.k):
                a_t, log_prob = self.action(s_t)
                while True:
                    rlt=self.env_model.list_check_up(self.env_model.state_to_action_map,s_t)
                    if rlt is None:
                        print(self.env_model.state_to_action_map)
                        print(s_t)
                        print(a_t)
                        print(done)
                        print("Pover")
                        return 
                    else:
                        if any([torch.equal(a_t,x) for x in self.env_model.state_to_action_map[rlt]])!=True:
                                a_t, log_prob = self.action(s_t)
                        else:
                            break
                    
                ns_t, r_t, done, _ = self.env_model.step_v1(a_t)

                states.append(s_t)
                actions.append(a_t)
                log_probs.append(log_prob)
                rewards.append(r_t)
                nstates.append(ns_t)
                
                s_t=ns_t
                self.curr_state=ns_t
                if done:
                    break
            self.D_fake.push(states, actions, rewards,nstates, log_probs)      
        return 
       
    def cvt_axis(self,trajs):
        t_states = []
        t_actions = []
        t_rewards = []
        t_nstates = []
        t_log_probs = []

        for traj in trajs:
            t_states.append(traj[0])
            t_actions.append(traj[1])
            t_rewards.append(traj[2])
            t_nstates.append(traj[3])
            t_log_probs.append(traj[4])

        return (t_states, t_actions, t_rewards,t_states,t_log_probs)
    
    def reward_to_value(self,t_rewards, gamma):

        t_Rs = []

        for rewards in t_rewards:
            Rs = []
            R = torch.zeros(1, 1)

            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                Rs.insert(0, R)
            t_Rs.append(Rs)
            
        return(t_Rs)

    def cal_log_prob(self, state, action):

        probs = self.model(Variable(state))
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()

        return(log_prob)
    
    def MBPO_train_1(self,D_real,mult_fcator=None):
        
        # Given D_real,and a multiplicative factor,will generate fake_data 
        # ST :len(D_fake)=multipl_factor*len(D_real)
        
        self.env_model.reset()
        self.env_model.update_param_given_epi(D_real)
        self.init_env_model()
        multiple_factor = (1-self.fraction_of_real)/self.fraction_of_real
        if mult_fcator is not None:
            multiple_factor=mult_fcator
        self.update_D_fake(int(multiple_factor*D_real.position))
        data_list=self.D_fake.buffer
        
        
        
        t_states, t_actions, t_rewards,t_nstates,t_log_probs = self.cvt_axis(data_list)
        t_Rs = self.reward_to_value(t_rewards, self.gamma)

        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                q_log_prob += log_probs[t].data.numpy()
            Z_ = math.exp(p_log_prob) / math.exp(q_log_prob)
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
        b = b / Z
        
        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = 0.

            for t in range(len(Rs)):
                loss = loss - (log_probs[t] * (Variable(Rs[t] - b).expand_as(log_probs[t]))).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)
            
        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        utils.clip_grad_value_(self.model.parameters(),40)
        self.optimizer.step()
        
        return
      
    def MBPO_train_2(self,D_real,fraction_of_real=None):
        
        # Given D_real,and the fraction of real to fake trajs,then train the policy on data comprising D_fake and D_real
        # ST real_ratio  follows the value given
        
        self.env_model.reset()
        self.env_model.update_param_given_epi(D_real)
        self.init_env_model()
        # mult_factor = (1-self.fraction_of_real)/self.fraction_of_real
        mult_factor = 1
        self.update_D_fake(int(mult_factor*D_real.position))
        
        frc_of_real=self.fraction_of_real
        if fraction_of_real is not None:
            frc_of_real=fraction_of_real
        self.batch_size=D_real.position
        num_of_real_epi=int(self.batch_size*frc_of_real)
        num_of_fake_epi=self.batch_size-num_of_real_epi
        pos_list=np.random.choice(a=len(self.D_fake.buffer),size=min([num_of_fake_epi,len(self.D_fake.buffer)]))
        fake_data_list=[self.D_fake.buffer[pos] for pos in pos_list]
        
        pos_list=np.random.choice(a=len(D_real.buffer),size=min([num_of_real_epi,len(D_real.buffer)]))
        real_data_list=[D_real.buffer[pos] for pos in pos_list]
        
        data_list=real_data_list+fake_data_list
        
        t_states, t_actions, t_rewards,t_nstates,t_log_probs = self.cvt_axis(data_list)
        t_Rs = self.reward_to_value(t_rewards, self.gamma)

        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                q_log_prob += log_probs[t].data.numpy()
            Z_ = math.exp(p_log_prob) / math.exp(q_log_prob)
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
        b = b / Z
        
        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = 0.

            for t in range(len(Rs)):
                loss = loss - (log_probs[t] * (Variable(Rs[t] - b).expand_as(log_probs[t]))).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)
            
        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        utils.clip_grad_value_(self.model.parameters(),40)
        self.optimizer.step()
        return

    def train_(self, D_real):
        
        # Pure policy gradient
        
        data_list=D_real.buffer
        
        t_states, t_actions, t_rewards,t_nstates,t_log_probs = self.cvt_axis(data_list)
        t_Rs = self.reward_to_value(t_rewards, self.gamma)

        Z = 0
        b = 0
        losses = []
        Z_s = []

        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            p_log_prob = 0
            q_log_prob = 0
            for t in range(len(Rs)):
                p_log_prob += (self.cal_log_prob(states[t], actions[t])).data.numpy()
                q_log_prob += log_probs[t].data.numpy()
            Z_ = math.exp(p_log_prob) / math.exp(q_log_prob)
            Z += Z_
            Z_s.append(Z_)
            b += Z_ * sum(Rs) / len(Rs)
        b = b / Z


        for (states, actions, Rs, log_probs) in zip(t_states, t_actions, t_Rs, t_log_probs):
            loss = 0.

            for t in range(len(Rs)):
                loss = loss - (log_probs[t] * (Variable(Rs[t] - b).expand_as(log_probs[t]))).sum()

            Z_ = Z_s.pop(0)
            loss = loss / Z_
            losses.append(loss)
            
        loss = sum(losses) / Z

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        utils.clip_grad_value_(self.model.parameters(),40)
        self.optimizer.step()



