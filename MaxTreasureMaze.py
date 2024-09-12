import numpy as np
import random

class MaxTreasureMazeGame:

    maze = []
    maze_size = 25
    treasure_count = 5
    treasure_location = []
    actions = ["U","D","L","R"]
    action_map = {"U":0,"D":1,"L":2,"R":3}
    terminal = (maze_size-1,maze_size-1)

    def __init__(self,M,epsilon,gamma) -> None:
        self.M = M
        self.epsilon = epsilon
        self.gamma = gamma
        self.initialize()
    
    def initialize(self):
        self.Q = np.zeros((
            self.maze_size,self.maze_size,self.treasure_count,len(self.actions)
        ))

        self.C = np.zeros_like(self.Q,dtype=int)

    def behavior_policy(self,i,j,v):
        """
        return an action depending on location of agent
        """
        greedy_action = self.Q[i,j,v,:].argmax()
        p = np.random.binomial(1,1-self.epsilon+self.epsilon/4)

        if p :
            return self.actions[greedy_action]
        else:
            return self.actions[random.randint(0,3)]

        
    def target_policy(self,i,j,v):
        """
        same for on-policy methods
        """
        return self.behavior_policy(i=i,j=j,v=v)

    def getISR(self,state,action):
        """
        return importance sampling ration. For on-policy case ISR=1
        """
        return 1
    def generate_episode(self):
        """
        Generate an episode according to the agents policy.
        Returns: 
        states: list of states that agent goes through.
        actions: list of actions taken by agent.
        rewards: reward gained by agent.
        """
        states = [(0,0,0)]
        actions = []
        rewards = []

        #use agent's policy
        while True:
            i,j,v = states[-1]
            action = self.behavior_policy(i=i,j=j,v=v)
            actions.append(action)
            reward = 0
            if action=="R":
                if j+1 >= self.M or self.maze[i,j+1]==0:
                    states.append((i,j,v))
                else:
                    if (i,j+1) in self.treasure_location:
                        states.append((i,j+1,v+1))
                        reward = 1
                    else:
                        states.append((i,j+1,v))
                    
            elif action=="D":
                if i+1 >= self.M or self.maze[i+1,j]==0:
                    states.append((i,j,v))
                else:
                    if (i+1,j) in self.treasure_location:
                        states.append((i+1,j,v+1))
                        reward = 1
                    else:
                        states.append((i+1,j,v))       
            elif action=="U":
                if i-1 < 0 or self.maze[i-1,j]==0:
                    states.append((i,j,v))
                else:
                    if (i-1,j) in self.treasure_location:
                        states.append((i-1,j,v+1))
                        reward = 1
                    else:
                        states.append((i-1,j,v))
            else:
                if j-1 < 0 or self.maze[i,j-1]==0:
                    states.append((i,j,v))
                else:
                    if (i,j-1) in self.treasure_location:
                        states.append((i,j-1,v+1))
                        reward = 1
                    else:
                        states.append((i,j-1,v))
            
            rewards.append(reward)
            new_state = states[-1]
            if (new_state[0],new_state[1])==self.terminal:
                return states,actions,rewards


    def mc_control(self) -> None:
        """MC Control for learning state-action function"""
        self.initialize()

        for m in range(self.M+1):
            states,actions,rewards = self.generate_episode()
            G = 0
            W = 1
            for t , (state,action,reward) in enumerate(zip(states[:-1],actions,rewards)):
               G =  self.gamma*G + reward
               self.C[state[0],state[1],state[2],self.action_map[action]] = self.C[state[0],state[1],state[2],self.action_map[action]] +1
               q = self.Q[state[0],state[1],state[2],self.action_map[action]]
               self.Q[state[0],state[1],state[2],self.action_map[action]] =  q + (1/self.C[state[0],state[1],state[2],self.action_map[action]])*(G-q)
               W = W*self.getISR()
        
        return

