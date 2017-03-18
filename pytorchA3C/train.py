import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms

def train(rank,args,shared_model):
    torch.manual_seed(args.seed + rank)
    # in order to avoid same random number

    env=create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model=ActorCritic(env.observation_space.shape[0],env.action_space.n,args.lstm_size)

    for param, shared_param in zip(model.parameters(),shared_model.parameters()):
        shared_param.grad.data=param.grad.data

    optimizer=optim.Adam(shared_model.parameters(),lr=args.lr)

    model.train()

    #values = []
    #log_probs= []

    state=env.reset()
    state=torch.from_numpy(state)
    done=True

    episode_length=0
    while True:
        episode_length+=1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx=Variable(torch.zeros(1,args.lstm_size))
            hx=Variable(torch.zeros(1,args.lstm_size))
        else:
            cx=Variable(cx.data)
            hx=Variable(hx.data)

        values=[]
        log_probs=[]
        rewards=[]
        entropies=[]

        for step in range(args.num_steps):
	    #env.render()
            value,logit,(hx,cx)=model((
                Variable(state.unsqueeze(0)),(hx,cx)))
            prob=F.softmax(logit)
	    #if rank==2:
		#print(prob.data)
            log_prob=F.log_softmax(logit)
	    #if rank==2:
		#print("prob:%.4f " %(prob.data[0][0]))
		#print("log_prob:%.4f " %(log_prob.data[0][0]))

            action=prob.multinomial().data

            log_prob=log_prob.gather(1,Variable(action))
            #log_prob of the chosen action
	    reward=0
	    #if args.env_name=='Breakout-v3':
	     #    state,reward,done,_=env.step(1)
	    	 #reward_sum+=reward

            state,reward2,done,_=env.step(action.numpy())
            done=done #or episode_length>=args.max_episode_length
            if args.env_name=='PongDeterministic-v3':
		reward=max(min(reward,1),-1)
	    reward+=reward2
            if episode_length>=args.max_episode_length:
		reward=-30
		done=True

            if done:
                episode_length=0
                state=env.reset()

            state=torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R=torch.zeros(1,1)
        if not done:
            value,_,_=model((Variable(state.unsqueeze(0)),(hx,cx)))
            R=value.data

        values.append(Variable(R))
        policy_loss=0
        value_loss=0
        gae=torch.zeros(1,1)
        R=Variable(R)

        for i in reversed(range(len(rewards))):
            R=args.gamma*R+rewards[i]
            advantage=R-values[i]
	    #advantage=Variable(advantage)

            value_loss=value_loss+advantage.pow(2)

            delta_t=rewards[i]+args.gamma*values[i+1].data-values[i].data
            #policy_loss=policy_loss-log_probs[i]*advantage
	    gae=gae*args.gamma+delta_t
	    policy_loss=policy_loss-Variable(gae)*log_probs[i]
	    #if rank==2:
		#print("advantage: %.2lf"%(advantage.data[0][0]))
		#print("log_probs : %.2lf " %(log_probs[i].data[0][0]))
            #print("log_prob:")
            #print(log_probs[i])
	
        optimizer.zero_grad()
        (policy_loss+value_loss).backward()
	#if rank==2:
	#	print("policy loss: ")
	#	print(policy_loss)
	#	print("valu_loss:")
	#	print(value_loss)

        optimizer.step()
