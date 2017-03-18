import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ActorCritic(torch.nn.Module):

    def __init__(self,num_inputs,num_outputs,lstm_size):
        super(ActorCritic,self).__init__()
        # used the init in torch.nn.Module

        self.conv1=nn.Conv2d(num_inputs,32,3,stride=2,padding=1)
        self.conv2=nn.Conv2d(32,32,3,stride=2,padding=1)
        self.conv3=nn.Conv2d(32,32,3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm=nn.LSTMCell(32*3*3,lstm_size)

        self.critic_linear=nn.Linear(lstm_size,1)
        self.actor_linear=nn.Linear(lstm_size,num_outputs)
        self.train()

    def forward(self,inputs):
        inputs, (hx,cx) =inputs
        x=F.elu(self.conv1(inputs))
        x=F.elu(self.conv2(x))
        x=F.elu(self.conv3(x))
        x=F.elu(self.conv4(x))

        x=x.view(-1,32*3*3)
        hx, cx=self.lstm(x,(hx,cx))
        x=hx

        return self.critic_linear(x),self.actor_linear(x),(hx,cx)


