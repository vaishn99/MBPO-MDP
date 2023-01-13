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


def  _multi_round_nmdp_simple():
    with dsl.new() as mdp:
        # Write down the MDP dynamics here 
        
        start = dsl.state()
        S_1=dsl.state()
        end = dsl.terminal_state()
        
        A_0=dsl.action()
        A_1=dsl.action()

        start & A_0 > dsl.reward(0) | dsl.reward(10)
        start & A_0 > start * 10 | end
        start & A_1 > dsl.reward(0) | dsl.reward(0) | dsl.reward(0)
        start & A_1 > start * 10 | end * 1 | S_1 * 1
        
        S_1 & A_0 > dsl.reward(0) | dsl.reward(10)
        S_1 & A_0 > S_1 * 1 | start
        S_1 & A_1 > dsl.reward(0) | dsl.reward(0)
        S_1 & A_1 > start * 5 | end
        
        dsl.discount(0.5)

        return mdp.validate()
    
def  _multi_round_nmdp_complex():
    with dsl.new() as mdp:
        # Write down the MDP dynamics here 
        
        start = dsl.state()
        S_1=dsl.state()
        S_2=dsl.state()
        S_3=dsl.state()
        S_4=dsl.state()
        S_5=dsl.state()
        end = dsl.terminal_state()
        
        A_0=dsl.action()
        A_1=dsl.action()

        start & A_0 > dsl.reward(10) | dsl.reward(0)
        start & A_0 > end * 1 | start
        start & A_1 > dsl.reward(0) | dsl.reward(0)
        start & A_1 > start * 1 | S_1
        
        S_1 & A_0 > dsl.reward(0) | dsl.reward(0)
        S_1 & A_0 > S_1 * 1 | start
        S_1 & A_1 > dsl.reward(0) | dsl.reward(0)
        S_1 & A_1 > S_1 * 1 | S_2
        
        S_2 & A_0 > dsl.reward(0) | dsl.reward(0)
        S_2 & A_0 > S_2 * 1 | S_1
        S_2 & A_1 > dsl.reward(0) | dsl.reward(0)
        S_2 & A_1 > S_2 * 1 | S_3
        
        S_3 & A_0 > dsl.reward(0) | dsl.reward(0)
        S_3 & A_0 > S_3 * 1 | S_2
        S_3 & A_1 > dsl.reward(0) | dsl.reward(0)
        S_3 & A_1 > S_3 * 1 | S_4
        
        S_4 & A_0 > dsl.reward(0) | dsl.reward(0)
        S_4 & A_0 > S_4 * 1 | S_3
        S_4 & A_1 > dsl.reward(0) | dsl.reward(0)
        S_4 & A_1 > S_4 * 1 | S_5
        
        S_5 & A_0 > dsl.reward(0) | dsl.reward(0)
        S_5 & A_0 > S_5 * 1 | S_4
        S_5 & A_1 > dsl.reward(10) | dsl.reward(0)
        S_5 & A_1 > end * 1 | S_1
        
        dsl.discount(0.5)

        return mdp.validate() 

MULTI_ROUND_NDMP = _multi_round_nmdp_simple()

solver = lp.LinearProgramming(MULTI_ROUND_NDMP)