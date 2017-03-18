import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def normalized_columns_initializer(weights, std=1.0):
    out=torch.randn(weights.size())
    out *= std/torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('conv')!=-1:
        weight_shape=list(m.weight.data.size())
        fan_in=np.prod(weight_shape[1:4])
        fan_out=np.prod(weight_shape[2:4])*weight_shape[0]
        w_bound= np.sqrt(6./ (fan_in+fan_out))
        m.weight.data.uniform_(-w_bound,w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear')!=-1:
        weight_shape=list(m.weight.data.size())
        fan_in=weight_shape[1]
        fan_out=weight_shape[0]
        w_bound = np.sqrt(1)
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(1)

class ActorCritic(torch.nn.Module):

    def __init__(self,num_inputs,num_outputs,lstm_size):
        super(ActorCritic,self).__init__()
        # used the init in torch.nn.Module

        self.linear1=nn.Linear(num_inputs*1*42*42,200)
	self.value_linear=nn.Linear(200,1)
	self.action_linear=nn.Linear(200,num_outputs)
	self.action_linear.weight.data=normalized_columns_initializer(
            self.action_linear.weight.data,0.01
        )
	self.value_linear.weight.data=normalized_columns_initializer(
            self.value_linear.weight.data,0.01
        )

	self.apply(weights_init)

        self.train()

    def forward(self,inputs):
        inputs, (hx,cx) =inputs
	x=inputs.view(-1,1*42*42)
        x=F.relu(self.linear1(x))

        return self.value_linear(x),self.action_linear(x),(hx,cx)


