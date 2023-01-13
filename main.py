import torch
# import torch.autograd as autograd
# import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.nn.utils as utils
# from torch.autograd import Variable

# import math

# from blackhc.mdp import dsl
# from blackhc import mdp
import time

# from blackhc.mdp import lp
import functools
# import numpy as np

from tqdm import tqdm
# from matplotlib import pyplot as plt
# from numpy import random

# from operator import itemgetter

# from collections import defaultdict
import numpy as np



##############################################

from Agent import Agent
from env import MULTI_ROUND_NDMP,solver
from utils import ReplayMemory

##############################################

class Test_bench:
    def __init__(self):
        self.horizon_len=20     
        self.num_of_real_epi=10
        self.k=10
        self.mult_factor=1.0
        self.real_ratio=0.5
        self.capacity=10000
        self.batch_size=2000
        self.learning_rate=0.001
        self.gamma=0.99
        
        self.num_of_outerloop=100 # outer loop
        self.num_of_innerloop=10 # inner loop
        
        self.D_real=ReplayMemory(self.capacity)
        self.env=None
        self.Agent=None
    def init_play_ground(self,env):
        self.env=env
        self.env.reset()
        self.A1=Agent(self.env.observation_space,self.env.action_space,gamma=self.gamma,learning_rate=self.learning_rate,horizon_len=self.horizon_len,k=self.k,fraction_of_real=self.real_ratio,batch_size=self.batch_size)
        self.A1.reset()
        
    def reset_play_ground(self):
        self.env.reset()
        self.A1.reset()
        self.D_real.flush_all()
        
    def display_env(self):    
        self.env.render()
        self.env.render_widget.width=500
        time.sleep(0.200)
        
    def update_D_real(self,num_of_epochs=None):
    
        num_of_episodes=self.num_of_real_epi
        if num_of_epochs is not None:
            num_of_episodes=num_of_epochs
        
        
        self.env.reset()
        
        s_t_index=self.env._state.index
        s_t=F.one_hot(torch.tensor(s_t_index),num_classes=self.env.observation_space.n).unsqueeze(dim=0)
        s_t=s_t.type(torch.FloatTensor)

        trajs=[]
        # D_real.flush_all()

        result=[]


        for traj_id in range(num_of_episodes):
            self.env.reset()
            # display_env()
            s_t_index=self.env._state.index
            
            states=[]
            log_probs=[]
            rewards=[]
            actions=[]
            nstates=[]
        
            for t in range(self.horizon_len):
                s_t=F.one_hot(torch.tensor(s_t_index),num_classes=self.env.observation_space.n).unsqueeze(dim=0)
                s_t=s_t.type(torch.FloatTensor)
                a_t, log_prob = self.A1.action(s_t)
                ns_t, r_t, done, _ = self.env.step(a_t.numpy()[0][0])
                # display_env()
                if t!=0:
                    nstates.append(s_t)
                states.append(s_t)
                actions.append(a_t)
                log_probs.append(log_prob)
                rewards.append(r_t)
                s_t_index=ns_t
                if done:
                    break
            # time.sleep(2)
            s_t=F.one_hot(torch.tensor(s_t_index),num_classes=self.env.observation_space.n).unsqueeze(dim=0)
            s_t=s_t.type(torch.FloatTensor)
            nstates.append(s_t)    
            self.D_real.push(states, actions, rewards,nstates, log_probs)
        
        return
    
    def perform_pure_fake(self,mul_factor,init_params):
        
        # MBPO based agent 

        self.reset_play_ground()
        result=[]
        
        for param in self.A1.model.fc1.parameters():
            param.data = torch.nn.parameter.Parameter(init_params)

        # print("\nBefore training:")

        # print(list(self.A1.model.fc1.parameters()))
        
        
        
        for x in range(self.num_of_outerloop):
            result.append(self.update_D_real(num_of_epochs=10))
            for i in range(self.num_of_innerloop):
                self.A1.MBPO_train_1(self.D_real,mul_factor)
                # A1.MBPO_train_2(D_real)
                # A1.train_(D_real)
                # pass
        # print("\nAfter training:")
        # print(list(self.A1.model.fc1.parameters()))
        
        return result
    
    def perform_mixed_strategy(self,fraction,init_params):
        
        # MBPO based agent 

        self.reset_play_ground()
        result=[]
        
        for param in self.A1.model.fc1.parameters():
            param.data = torch.nn.parameter.Parameter(init_params)
        
        # print("\nBefore training:")

        # print(list(self.A1.model.fc1.parameters()))
        
        for x in range(self.num_of_outerloop):
            result.append(self.update_D_real(num_of_epochs=10))
            for i in range(self.num_of_innerloop):
                # self.A1.MBPO_train_1(self.D_real)
                self.A1.MBPO_train_2(self.D_real,fraction)
                # self.A1.train_(D_real)
                # pass
        # print("\nAfter training:")
        # print(list(self.A1.model.fc1.parameters()))
        
        return result
    
    def perform_pure_real(self,init_params):
        
        # MBPO based agent 

        self.reset_play_ground()
        result=[]
        
        for param in self.A1.model.fc1.parameters():
            param.data = torch.nn.parameter.Parameter(init_params)

        # print("\nBefore training:")

        # print(list(self.A1.model.fc1.parameters()))
        
        
        for x in range(self.num_of_outerloop):
            result.append(self.update_D_real(num_of_epochs=10))
            for i in range(self.num_of_innerloop):
                # self.A1.MBPO_train_1(self.D_real)
                # self.A1.MBPO_train_2(D_real)
                self.A1.train_(self.D_real)
                # pass
        # print("\nAfter training:")
        # print(list(self.A1.model.fc1.parameters()))
        
        return result
    
    
if __name__=='__main__':
    print("Entered training")
    play_ground=Test_bench()    # create an instance of Test_bench
    env = MULTI_ROUND_NDMP.to_env() # initialize the environment
    play_ground.init_play_ground(env=env)   # initialize Playground,Agent will be initilized within the play_ground
    play_ground.reset_play_ground() # setting to the Default state of Playground
    
    Q_array=solver.compute_q_table(max_iterations=10000, all_close=functools.partial(np.allclose, rtol=1e-10, atol=1e-10)) # Q_value associated with the environment
    
    # play_ground.perform_pure_real()
    # play_ground.perform_pure_fake(mult_factor=1)
    # play_ground.perform_mixed_strategy()
    
    # my_data = np.genfromtxt("./experiment/init_param_small.csv", delimiter=',')
    # init_params=torch.from_numpy(my_data)
    # result=play_ground.perform_pure_real(init_params.float())
    
    fract_list=np.arange(0,1.1,.1)
    prim_policy_list=[]
    for fract in fract_list:
        fract=np.round(fract,2)
        my_data = np.genfromtxt("./experiment/init_param_small.csv", delimiter=',')
        init_params=torch.from_numpy(my_data)
        result=play_ground.perform_mixed_strategy(fract,init_params.float())
        prim_policy_list.append(list(play_ground.A1.model.fc1.parameters()))
    prim_policy_list=np.array(prim_policy_list)
    np.savetxt("./experiment/exp_rslt_1.csv", prim_policy_list, delimiter=",")
    
    
    fract_list=np.arange(0,1.1,.1)
    sec_policy_list=[]
    number_of_times=10

    for fract in fract_list:
        fract=np.round(fract,2)
        policy_for_fract=[]
        for i in range(number_of_times):
            my_data = np.genfromtxt("./experiment/init_param_small.csv", delimiter=',')
            init_params=torch.from_numpy(my_data)
            result=play_ground.perform_mixed_strategy(fract,init_params.float())
            policy_for_fract.append(list(play_ground.A1.model.fc1.parameters()))
        policy_for_fract=np.array(policy_for_fract)
        sec_policy_list.append(policy_for_fract)
        print("Done with",fract)
    sec_policy_list=np.array(sec_policy_list)
    np.savetxt("./experiment/exp_rslt_2.csv", sec_policy_list, delimiter=",")
    
    true_Q=solver.compute_q_table(max_iterations=10000, all_close=functools.partial(np.allclose, rtol=1e-10, atol=1e-10))
    np.savetxt("./experiment/true_Q.csv", true_Q, delimiter=",")   