#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gym

import collections
import os

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

    def __init__(self, env):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
        model = Sequential()       # the model that is trained for Q value estimator (Q hat)
        model.add(Dense(25, activation='tanh', input_shape=(env.observation_space.shape + 1,)))  # input: state and action
        model.add(Dense(60,activation='tanh'))
        model.add(Dense(90, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
          
    def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
        keras.models.save_model(suffix,self.model.get_weights())    		
                    # save #np.save(suffix,self.model.get_weights())
                    
    def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
        
        return keras.models.load_model(model_file)    
                
    def load_model_weights(self,weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		
        return keras.models.load_model(weight_file) 



class Replay_Memory():

    def __init__(self,env, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        
        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        self.M = collections.deque(maxlen=memory_size)  # list-like container with fast appends and pops on either end
        state = env.reset()                   # observation of initial state
        next_state = state.copy()             # initialize next state
        
        while len(self.M) < burn_in:         # while the size of memory does not exceed burnin size
            done = False
            while not done:
                action = env.action_space.sample() 
                state = next_state.copy()
                next_state, reward, done, info = env.step(action)
                
                self.append(self.Transition(state,action,reward,next_state,done))

        
    def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
        
        size = len(self.M)      # the size of M
        random_list = np.random.permutation(size)
        batch = self.M[random_list[0:31]].copy()
        return batch
        
        

    def append(self, transition):
		# Appends transition to the memory. 	
    	self.M.append(transition)


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
    def __init__(self, environment_name,episode, epsilon, gamma, render=False,burn_in = 10000,memory_size=50000):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
        self.env = gym.make(environment_name)
        self.Qnet = QNetwork(self.env)
        self.Qnet_target = QNetwork(self.env)
        
        self.Replay = Replay_Memory(self.env, memory_size, burn_in)
        self.Transition = self.Replay.Transition
        
        
        
        self.gamma = gamma           # discount factor
        self.episode = episode       # number of episodes to run 
        self.memory = self.Replay.M  # memory object list of tuple size (n,5)
        self.eps = epsilon
    
    
    
    def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
        if np.random.randn(1) < self.eps:             # input a list of qvalues with all action space 
            action = self.env.action_space.sample()   # and output the action
        else:
            action = np.argmax(q_values)
        return action
        

    def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
        action = np.argmax(q_values)
        return action

    def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 
		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
        for i in range(self.episode):
            state = self.env.reset()
            done = False
            while not done:
                action = self.epsilon_greedy_policy([self.Qnet.predict(state.copy().append(act)) for act in range(self.env.action_space.n)])      # epsilon greedy policy
                next_state, reward, done, info = self.env.step(action)
                self.Replay.append(self.Transition(state,action,reward,next_state,done))   # store transition in memory
                batch = self.Replay.sample_batch(batch_size=32)       # sample minibatch from memory size(32,5)
                y = []
                for transition in batch:
                    r = transition[2]
                    s = transition[0]
                    if transition[4] == True:      # if done
                        y.append(r)    # append reward
                    else:
                        a_opt = self.greedy_policy([self.Qnet_target.predict(s.copy().append(act)) for act in range(self.env.action_space.n)])   #???
                        y.append(r+self.gamma*self.Qnet_target.predict(s.copy().append(a_opt))) 

        
        
        
        

    def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
        pass
     
    def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
        Replay = Replay_Memory(self.env,memory_size, burn_in)
        return Replay
        

# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()


def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

