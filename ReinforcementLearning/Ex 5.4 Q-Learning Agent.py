
# coding: utf-8

# # DAT257x: Reinforcement Learning Explained
# 
# ## Lab 5: Temporal Difference Learning
# 
# ### Exercise 5.4: Q-Learning Agent

# In[ ]:


import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../") 
    
from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.simulation import Experiment


# In[ ]:


class Agent(object):  
        
    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, state):
        raise NotImplementedError


# In[ ]:


class QLearningAgent(Agent):
    
    def __init__(self, actions, epsilon=0.01, alpha=0.5, gamma=1):
        super(QLearningAgent, self).__init__(actions)
        
        ## Initialize empty dictionary here
        self.Q = defaultdict(lambda: np.zeros(self.num_actions))
                        
        ## In addition, initialize the value of epsilon, alpha and gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma 
        
    def stateToString(self, state):
        mystring = ""
        if np.isscalar(state):
            mystring = str(state)
        else:
            for digit in state:
                mystring += str(digit)
        return mystring    
    
    def act(self, state):
        stateStr = self.stateToString(state)      
        action = np.random.randint(0, self.num_actions) 
        
        A = np.ones(self.num_actions, dtype=float) * self.epsilon / self.num_actions
        best_actions = np.argwhere(self.Q[stateStr] == np.amax(self.Q[stateStr])).flatten()
        for i in best_actions:
            A[i] += (1.0 - self.epsilon)/best_actions.size
        action = np.random.choice(np.arange(self.num_actions), p = A)
        #print(best_actions)
        return action
    
    def learn(self, state1, action1, reward, state2, done):
        state1Str = self.stateToString(state1)
        state2Str = self.stateToString(state2)
        
        ## TODO 3
        ## Implement the q-learning update here
        
        td_target = reward + self.gamma * np.amax(self.Q[state2Str])
        td_delta = td_target - self.Q[state1Str][action1]
        self.Q[state1Str][action1] += self.alpha * td_delta
        
        """
        Q-learning Update:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max(Q(s') - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
        or
        Q(s,a) <- Q(s,a) + alpha * td_delta
        """


# In[ ]:

"""
interactive = True
get_ipython().magic(u'matplotlib nbagg')
env = SimpleRoomsEnv()
agent = QLearningAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_qlearning(10, interactive)


# In[ ]:


interactive = False
get_ipython().magic(u'matplotlib inline')
env = SimpleRoomsEnv()
agent = QLearningAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_qlearning(50, interactive)


# In[ ]:


interactive = True
get_ipython().magic(u'matplotlib nbagg')
env = CliffWalkingEnv()
agent = QLearningAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_qlearning(10, interactive)


# In[ ]:


interactive = False
get_ipython().magic(u'matplotlib inline')
env = CliffWalkingEnv()
agent = QLearningAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_qlearning(100, interactive)


# In[ ]:


interactive = False
get_ipython().magic(u'matplotlib inline')
env = WindyGridworldEnv()
agent = QLearningAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_qlearning(50, interactive)

"""