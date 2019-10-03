#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

import collections
import os

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

    def __init__(self, state_dim,action_dim,learning_rate):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
        self.model = Sequential()       # the model that is trained for Q value estimator (Q hat)
        self.model.add(Dense(40,kernel_initializer='random_uniform', activation='relu', input_shape=(state_dim+1,)))  # input: state and action
        self.model.add(Dense(40,kernel_initializer='random_uniform',activation='relu'))
        self.model.add(Dense(1,kernel_initializer='random_uniform', activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['mse'])
          
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
        self.memory_size = memory_size
        self.Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        self.M = collections.deque(maxlen=memory_size)  # list-like container with fast appends and pops on either end
        
        while len(self.M) < burn_in:         # while the size of memory does not exceed burnin size
            state = env.reset()                   # observation of initial state
            next_state = state.copy()             # initialize next state
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
        random_list = np.random.permutation(size)    # randomly choose some number
        batch = []
        for i in random_list[0:32]:
            batch.append(self.M[i])

        return batch
        
        

    def append(self, transition):
		# Appends transition to the memory. 	
        self.M.append(transition)
        if len(self.M) > self.memory_size:
            self.M.popleft()

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
    def __init__(self, environment_name,episode, epsilon, gamma, C, learning_rate,render=False,burn_in = 10000,memory_size=50000):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
        self.env = gym.make(environment_name)

        self.Qnet = QNetwork(self.env.observation_space.shape[0],self.env.action_space.n, learning_rate)
        self.Qnet_target = QNetwork(self.env.observation_space.shape[0],self.env.action_space.n, learning_rate)
        self.Qnet_target.model.set_weights(self.Qnet.model.get_weights())
        

        self.Replay = Replay_Memory(self.env, memory_size, burn_in)
        self.Transition = self.Replay.Transition

        self.gamma = gamma           # discount factor
        self.episode = episode       # number of episodes to run 
        self.memory = self.Replay.M  # memory object list of tuple size (n,5)
        self.eps = epsilon
        self.C = C
        self.learning_rate = learning_rate
    
    def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
        if np.random.uniform(0,1,1) <= self.eps:             # input a list of qvalues with all action space 
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
        loss = []
        count = 0 
        for i in range(self.episode):
            state = self.env.reset()
            size = state.shape[0]
            done = False
            c = 0
            
            if i % 100 == 0:
                print('Episode: ', i)
                tot_reward = self.test()
                print('Reward: ',tot_reward)
            
            
            while not done:
                                   
                greedy = []
                for act in range(self.env.action_space.n):
                    greedy.append(self.Qnet.model.predict(np.concatenate((state.copy(),[act])).reshape(1,size+1)).tolist()[0][0] )
                    
                action = self.epsilon_greedy_policy(greedy)      # epsilon greedy policy
                next_state, reward, done, info = self.env.step(action)
                self.Replay.append(self.Transition(state,action,reward,next_state,done))   # store transition in memory
                batch = self.Replay.sample_batch(batch_size=32)       # sample minibatch from memory size(32,5)
                y = []
                x = []
              #  qnet = []
                for transition in batch:
                    state_x = transition[0].copy()
                    act_x = transition[1]
                    x.append(np.concatenate((state_x,[act_x])))
                    r = transition[2]  # reward
                    s = transition[0]  # state
                    # qnet is the lsit of Q(s,a) from minibatch
                   # qnet.append(self.Qnet.model.predict(np.concatenate((s.copy(),[act_x])).reshape(1,size+1)).tolist()[0][0])
                   
                    if transition[4] == True:      # if done
                        y.append(r)    # This terminates. append reward without Q value
                    else:
                        greedy = []  # the list containing q values for greedy algorithms
                        for act in range(self.env.action_space.n):
                            greedy.append(self.Qnet_target.model.predict(np.concatenate((s.copy(),[act])).reshape(1,size+1)).tolist()[0][0])
                            
                        a_opt = self.greedy_policy([greedy])   
                        y.append(r+self.gamma*self.Qnet_target.model.predict(np.concatenate((s.copy(),[a_opt])).reshape(1,size+1)).tolist()[0][0]) 
                
               # error = np.power(np.array(y) - np.array(qnet),2) 
                
                  # train model on gradient descent
                history = self.Qnet.model.fit(np.array(x),np.array(y),epochs = 1,verbose=0)
                loss.append( history.history['mse'] )
           #     acc.append( history.history['accuracy'] )
                c+=1 
                if c % self.C == 0:
                    self.Qnet_target.model.set_weights(self.Qnet.model.get_weights())
                count +=1 
                
                if count < 100000:  # decaying epsilon over iterations
                    self.eps -= (0.5-0.05)/100000
            self.env.reset()
             
        return loss

    def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
        total_reward_list = []
        state_dim = self.env.observation_space.shape[0]
        for i in range(100):
            state = self.env.reset()
            total_reward = 0
            done = False

            # Check if the game is terminated

            while done == False:
                # Take action and observe
                q_value_list = []
                for act in range(self.env.action_space.n):
                    q_value_list.append( self.Qnet.model.predict( np.concatenate((state.copy(),[act]) ).reshape(1,state_dim+1)).tolist()[0][0] )
                action = self.greedy_policy(q_value_list)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            total_reward_list.append(total_reward)
        reward_mean = np.mean(np.array(total_reward_list))
        return reward_mean
        
     
    def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
#        Replay = Replay_Memory(self.env,memory_size, burn_in)
        pass
        #return Replay
        

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
#    gpu_ops = tf.GPUOptions(allow_growth=True)
#    config = tf.ConfigProto(gpu_options=gpu_ops)
#    sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
#    keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 
    
    env_name = 'CartPole-v1'
    a = DQN_Agent(env_name,episode = 3000, epsilon= 0.5, gamma=0.99, C = 10000,learning_rate = 0.001)
    loss = a.train()
    
    
if __name__ == '__main__':
	main(sys.argv)

