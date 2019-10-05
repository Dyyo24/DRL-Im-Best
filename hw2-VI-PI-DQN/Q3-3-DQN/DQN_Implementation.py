#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

import collections
import os

import matplotlib.pyplot as plt

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

    def __init__(self, state_dim, learning_rate):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
        self.model = Sequential()
        # Input: state and action
        self.model.add(Dense(50, kernel_initializer='random_uniform', activation='tanh', input_shape=(state_dim+1,)))
# =============================================================================
#         self.model.add(Dense(50, kernel_initializer='random_uniform', activation='relu'))
#         self.model.add(Dense(50, kernel_initializer='random_uniform', activation='tanh'))
# =============================================================================
        # Output: Q-value
        self.model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=['mse'])
        pass
    
    def predict(self, state, action):
        '''
        Input:
            state: (N, state_dim)array
            action: (N, 1)array
        '''
        Q_value = self.model.predict( np.concatenate((state,action),axis=-1) )
        return Q_value
    
    def fit(self, state, action, target):
        '''
        Input:
            state: (N, state_dim)array
            action: (N, 1)array
            target: (N, 1)array
        '''
        X = np.concatenate((state,action),axis=-1)
        history = self.model.fit(X, target, verbose=0)
        return history
    
    def save_model_weights(self, path_to_weight_file):
		# Helper function to save your model / weights. 
        self.model.save_weights(path_to_weight_file)
        pass    		
                    
    def load_model(self, path_to_model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
        self.model = keras.models.load_model(path_to_model_file)  
        pass
                
    def load_model_weights(self, path_to_weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
        self.model.load_weights(path_to_weight_file)
        pass



class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.M = collections.deque(maxlen=memory_size)
        self.memory_size = memory_size
        pass
    
    def append(self, transition):
		# Append a transition to the memory. 	
        self.M.append(transition)
        pass
            
    def sample_batch(self, batch_size=32):
        '''
        Output: batch of data, (batch_size,trasition_length) array
        '''
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
        random_idx = np.random.choice(len(self.M), batch_size, replace=False)
        data_batch = []
        for idx in list(random_idx):
            data_batch.append(self.M[idx])
        return data_batch

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
    def __init__(self, environment_name, episode_max, epsilon, gamma, C, learning_rate, render=False,memory_size=50000,burn_in = 10000):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
        
        # Save hyperparameters
        self.gamma = gamma
        self.episode_max = episode_max 
        self.eps = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.burn_in = burn_in
        self.memory_size = memory_size
        # Create environment
	self.environment_name = environment_name
        self.env = gym.make(environment_name)
        # Create Qnet
        self.Qnet = QNetwork(self.env.observation_space.shape[0], self.learning_rate)
        pass
    
    def burn_in_memory(self):
        '''
        Initialize your replay memory with a burn_in number of episodes / transitions. 
        trasition: list of 5 element e.g. [state, action, reward, next_state, done]
        '''
        self.memory = Replay_Memory(self.memory_size, self.burn_in)
        memory_len = 0
        while memory_len<self.burn_in:         
            state = self.env.reset()                   
            done = False
            while not done and memory_len<self.burn_in:
                action = self.env.action_space.sample() 
                next_state, reward, done, info = self.env.step(action)
                self.memory.append([state, action, reward, next_state, done])
                state = next_state.copy()
                memory_len += 1
        print('Burn-in memory size = %d' % memory_len)
        pass
        
    def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
        if np.random.uniform() < self.eps:           
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action
        
    def epsilon_greedy_policy_video(self, q_values,eps):

		# Creating epsilon greedy probabilities to sample from.             

        if np.random.uniform() < eps:           
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action	

    def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
        action_opt = np.argmax(q_values,axis=-1)
        return action_opt

    def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 
		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
        
        # Create target Qnet with the same weights as Qnet
        self.Qnet_target = QNetwork(self.env.observation_space.shape[0], self.learning_rate)
        self.Qnet_target.model.set_weights(self.Qnet.model.get_weights())
        # Initialize memory
        self.burn_in_memory()
        # Initialize parameters
        loss = []
        reward_vec = []
        TD_error_vec = []
        iter_num = 0
        # Start training episodes
        for episode_num in range(self.episode_max):
            state = self.env.reset()
            done = False
            TD = []
            while not done:
                # Calculate Q-values and Select action
                # Construct input action (action_num, 1)array
                all_action_vec = np.arange(self.env.action_space.n).reshape(-1,1)
                # Construct input state (action_num, state_dim)array
                state_vec = np.tile(state, (self.env.action_space.n,1))
                # Observe predicted Q-value (action_num, 1)array
                all_q_value_vec = self.Qnet.predict(state_vec,all_action_vec)
                # Select epsilon optimal action
                action = self.epsilon_greedy_policy(all_q_value_vec)
                # Interact with the environment
                next_state, reward, done, info = self.env.step(action)
                self.memory.append([state,action,reward,next_state,done])
                state = next_state.copy()
                # Calculate TD error
                next_state_vec = np.tile(next_state, (self.env.action_space.n,1)) # (action_num, state_dim)
                next_state_q_vec = self.Qnet.predict(next_state_vec,all_action_vec) # (action_num, state_dim)
                TD.append( reward+(1-done)*self.gamma*np.max(next_state_q_vec) - np.max(all_q_value_vec) )

                # Training Qnet
                batch = self.memory.sample_batch()
                state_x = np.array([trasition[0] for trasition in batch])
                action_x = np.array([trasition[1] for trasition in batch]).reshape(-1,1)
                reward_x = np.array([trasition[2] for trasition in batch]).reshape(-1,1)
                next_state_x = np.array([trasition[3] for trasition in batch])
                done_x = np.array([trasition[4] for trasition in batch]).reshape(-1,1)
                # Calculate target Q-value
                # Prepare next state in mini batch to concatenate with all posible action array
                all_next_state_x = np.tile(next_state_x[:,np.newaxis,:], (1,self.env.action_space.n,1) ) # (batch_size, action_num, state_dim)
                all_next_state_x = all_next_state_x.reshape(-1,self.env.observation_space.shape[0]) # (batch_size*action_num, state_dim)
                # Prepare all posible action array in mini batch to concatenate with next state array
                all_action_x = np.tile(all_action_vec[np.newaxis,:,:], (len(batch),1,1) ) # (batch_size, action_num, 1)
                all_action_x = all_action_x.reshape(-1,1) # (batch_size*action_num, 1)
                # Calculate Q-value for all posible action from the next state
                all_q_value_vec_x = self.Qnet_target.predict(all_next_state_x, all_action_x) # (batch_size*action_num, 1)
                all_q_value_vec_x = all_q_value_vec_x.reshape(len(batch),self.env.action_space.n) # (batch_size,action_num)
                # Select greedy policy
                action_opt_x = self.greedy_policy(all_q_value_vec_x) # (batch_size,)
                action_opt_x = action_opt_x.reshape(len(batch),1) # (batch_size,1)
                # Calculate target Q-value
                Y = reward_x + (1-done_x) * self.gamma * self.Qnet_target.predict(next_state_x,action_opt_x)
                # train model on gradient descent
                history = self.Qnet.fit(state_x, action_x, Y)
                loss.append( history.history['mse'] )
                
                # Update target Qnet after C steps
# =============================================================================
#                 if iter_num % self.C ==0:
#                     self.Qnet_target.model.set_weights(self.Qnet.model.get_weights())
# =============================================================================
                # decaying epsilon over iterations
                if self.eps > 0.05:
                    self.eps -= 10e-7
                iter_num += 1
            self.Qnet_target.model.set_weights(self.Qnet.model.get_weights())
            
            TD_error_episode_average =np.mean(TD)
            TD_error_vec.append(TD_error_episode_average)

            if episode % int(self.num_episodes/3) == 0:        # record video
         	test_video(self, self.environment_name, episode)
		
            # Test the Qnet every 100 episodes
            if episode_num % 100 ==0:
                print('------------------------------')
                print('Episode: ', episode_num)
                print('Iteration: ', iter_num-1)
                print('Memory size: ', len(self.memory.M))
                print('T Difference: ', TD_error_episode_average)
                print('Epsilon: ', self.eps)
                test_reward = self.test()
                reward_vec.append(test_reward)
                print('Reward: ',test_reward)
                print('Loss: ', np.mean(np.array(loss[iter_num-500:iter_num-1])))
        plt.plot(reward_vec)
        plt.show()
        plt.plot(TD_error_vec)
        plt.show()
        self.Qnet.save_model_weights('my_model_weights.h5')
        np.save('TD.npy',TD_error_vec)
        np.save('reward.npy',reward_vec)
        return loss, reward_vec

    def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
       
        total_reward_list = []
        # Start evaluate over 20 episodes
        for i in range(20):
            state = self.env.reset()
            total_reward = 0
            done = False
            # Check if the game is terminated
            while not done:
                # Calculate Q-values and Select action
                # Construct input action (action_num, 1)array
                all_action_vec = np.arange(self.env.action_space.n).reshape(-1,1)
                # Construct input state (action_num, state_dim)array
                state_vec = np.tile(state, (self.env.action_space.n,1))
                # Observe predicted Q-value (action_num, 1)array
                all_q_value_vec = self.Qnet.predict(state_vec,all_action_vec)
                all_q_value_vec = all_q_value_vec.reshape(1,-1)
                # Select epsilon action
                action = self.greedy_policy(all_q_value_vec).item()
                # Interact with the environment
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            total_reward_list.append(total_reward)
        print('total reward max', max(total_reward_list))
        reward_mean = np.mean(np.array(total_reward_list))
        return reward_mean

        

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
        all_action_vec = np.arange(agent.env.action_space.n).reshape(-1,1)
                # Construct input state (action_num, state_dim)array
        state_vec = np.tile(state, (agent.env.action_space.n,1))
                # Observe predicted Q-value (action_num, 1)array
        all_q_value_vec = agent.Qnet.predict(state_vec,all_action_vec)
                # Select epsilon optimal action

        action = agent.epsilon_greedy_policy_video(all_q_value_vec,0.05)


#        action = agent.epsilon_greedy_policy(state, 0.05)

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
    
    env_name = 'CartPole-v0'
    a = DQN_Agent(env_name, episode_max = 3000, epsilon= 0.5, gamma=0.99, C = 5000, learning_rate = 0.001, memory_size=50000, burn_in = 10000)
    loss, reward_vec= a.train()
    
if __name__ == '__main__':
	main(sys.argv)

