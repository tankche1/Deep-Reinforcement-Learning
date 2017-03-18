import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets,transforms
import time
from collections import deque

def test(rank,args,shared_model):
    torch.manual_seed(args.seed+rank)

    env=create_atari_env(args.env_name)
    env.seed(args.seed+rank)

    model=ActorCritic(env.observation_space.shape[0],env.action_space.n,args.lstm_size)

    model.eval()

    state=env.reset()
    state=torch.from_numpy(state)
    reward_sum=0
    done=True

    start_time=time.time()

    #actions=deque(maxlen=100)
    episode_length=0

    currentPath = os.getcwd()
    File = open(currentPath + '/record.txt', 'a+')
    print("\n\n\n\n------------------------------\n\n\n\n\n")
    File.write("\n\n\n\n------------------------------\n\n\n\n\n")
    File.close()

    cnt=0
    episode_number=0

    while True:
        env.render()
        cnt=cnt+1
        episode_length+=1
        if done:
            model.load_state_dict(shared_model.state_dict())
            hx=Variable(torch.zeros(1,args.lstm_size),volatile=True)
            cx = Variable(torch.zeros(1, args.lstm_size), volatile=True)
        else:
            hx=Variable(hx.data,volatile=True)
            cx = Variable(cx.data, volatile=True)

        #print(state)
        value,logit,(hx,cx)=model(
            (Variable(state.unsqueeze(0),volatile=True),(hx,cx)))
        prob=F.softmax(logit)
        #action=prob.max(1)[1].data.numpy()
	action=prob.multinomial().data
	
	#if(args.env_name=='Breakout-v3'):
	 #    state,reward,done,_=env.step(1)
        #     reward_sum+=reward  
        #state,reward,done,_ =env.step(action[0,0])
	state,reward,done,_=env.step(action.numpy())
        done=done #or episode_length >= args.max_episode_length
	if episode_length >= args.max_episode_length:
		done=True
		reward_sum-=30
        reward_sum+=reward

        #actions.append(action[0,0])
        #if actions.count(actions[0])==actions.maxlen:
        #    done=True
	#if reward!=0:
	  #  print("ep %d : game finished,reward: %d " %(episode_number,reward))+('' if reward == #-1 else ' !!!!!!!!')

        if done:
	    hour=int(time.strftime("%H",time.gmtime(time.time()-start_time)))
	    _min=int(time.strftime("%M",time.gmtime(time.time()-start_time)))
	    
            print("Time {},episode reward {}, episode length {} " .format(
                hour*60+_min+args.starttime,
                reward_sum,episode_length
            ))

            File = open(currentPath + '/record.txt', 'a+')
            File.write("Time {},episode reward {}, episode length {} \n" .format(
                hour*60+_min+args.starttime,
                reward_sum,episode_length
            ))
            File.close()

            reward_sum=0
            episode_length=0
            #actions.clear()
            state=env.reset()

            torch.save(model.state_dict(), currentPath + '/A3C.t7')
	    episode_number+=1
            time.sleep(60)

        state=torch.from_numpy(state)

