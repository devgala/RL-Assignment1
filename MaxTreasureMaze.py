import numpy as np
import random
from  tqdm import tqdm

class MaxTreasureMazeGame:

    maze = np.array([
        [1,0,1,1,1,1,1,1,1,1],
        [1,0,1,0,0,0,1,0,0,0],
        [1,1,1,0,1,1,1,1,1,1],
        [1,0,1,0,1,0,0,0,0,1],
        [1,0,1,0,1,0,1,1,1,1],
        [1,0,1,1,1,0,1,0,1,1],
        [1,0,1,0,1,0,1,0,0,1],
        [1,0,1,0,1,1,1,0,0,1],
        [1,1,1,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,1,1,0,1]
    ])


    maze_size = 10
    treasure_count = 3
    treasure_location = [(0,2),(0,9),(5,3)]


    # maze  = np.array([
    #     [1,0,0],
    #     [1,1,0],
    #     [1,1,1]
    # ])
    # maze_size = 3
    # treasure_count = 1
    # treasure_location = [(1,1)]


    actions = ["U","D","L","R"]
    action_map = {"U":0,"D":1,"L":2,"R":3}
    terminal = (maze_size-1,maze_size-1)

    def __init__(self,M,T,epsilon,gamma) -> None:
        self.M = M
        self.epsilon = epsilon
        self.gamma = gamma
        self.T =  T
        self.initialize()
    
    def initialize(self):
        self.Q = np.zeros((
            self.maze_size,self.maze_size,self.treasure_count+1,len(self.actions)
        ))

        self.C = np.zeros_like(self.Q,dtype=np.int64)

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
        used_treasures = []
        #use agent's policy
        while True:
            i,j,v = states[-1]
            action = self.behavior_policy(i=i,j=j,v=v)
            # print("hello",i,j,v,action)
            actions.append(action)
            reward = 0
            if action=="R":
                if j+1 >= self.maze_size or self.maze[i,j+1]==0:
                    states.append((i,j,v))
                else:
                    if (i,j+1) in self.treasure_location and not ((i,j+1) in used_treasures):
                        states.append((i,j+1,v+1))
                        used_treasures.append((i,j+1))
                        reward = 1
                    else:
                        states.append((i,j+1,v))
                    
            elif action=="D":
                if i+1 >= self.maze_size or self.maze[i+1,j]==0:
                    states.append((i,j,v))
                else:
                    if (i+1,j) in self.treasure_location and not ((i+1,j)  in used_treasures):
                        states.append((i+1,j,v+1))
                        used_treasures.append((i+1,j))
                        reward = 1
                    else:
                        states.append((i+1,j,v))       
            elif action=="U":
                if i-1 < 0 or self.maze[i-1,j]==0:
                    states.append((i,j,v))
                else:
                    if (i-1,j) in self.treasure_location and not ((i-1,j)  in used_treasures):
                        states.append((i-1,j,v+1))
                        used_treasures.append((i-1,j))
                        reward = 1
                    else:
                        states.append((i-1,j,v))
            else:
                if j-1 < 0 or self.maze[i,j-1]==0:
                    states.append((i,j,v))
                else:
                    if (i,j-1) in self.treasure_location and not ((i,j-1) in used_treasures):
                        states.append((i,j-1,v+1))
                        used_treasures.append((i,j-1))
                        reward = 1
                    else:
                        states.append((i,j-1,v))
            
            rewards.append(reward)
            new_state = states[-1]
            if (new_state[0],new_state[1])==self.terminal:
                # print("end")
                rewards[-1] = rewards[-1] + 2 ** new_state[2]
                return states,actions,rewards
            
        reward = reward*0.5
        return states,actions,rewards

    def mc_control(self) -> None:
        """MC Control for learning state-action function"""
        self.initialize()

        for m in tqdm(range(self.M+1)):
            # print(m)
            states,actions,rewards = self.generate_episode()
            G = 0
            W = 1
            for t , (state,action,reward) in enumerate(zip((states[:-1])[::-1],actions[::-1],rewards[::-1])):
               G =  self.gamma*G + reward
               self.C[state[0],state[1],state[2],self.action_map[action]] = self.C[state[0],state[1],state[2],self.action_map[action]] + W
               q = self.Q[state[0],state[1],state[2],self.action_map[action]]
               self.Q[state[0],state[1],state[2],self.action_map[action]] =  q + (W/self.C[state[0],state[1],state[2],self.action_map[action]])*(G-q)
               W = W*self.getISR(state,action)
               if W==0:
                   break
            np.save(file="action-state",arr=self.Q)
        
        return
    
    def create_heat_map(self):
        heat_maps = []
        for k in range(0,self.treasure_count+1):
            heat_map = np.zeros((self.maze_size,self.maze_size))
            for i in range(0,self.maze_size):
                for j in range(0,self.maze_size):
                    heat_map[i,j] = self.Q[i,j,k,:].max()
           
            heat_maps.append(heat_map)
        return heat_maps
    
    def create_action_map(self):
        heat_maps = []
        for k in range(0,self.treasure_count+1):
            heat_map = np.zeros((self.maze_size,self.maze_size),dtype=type(str))
            for i in range(0,self.maze_size):
                for j in range(0,self.maze_size):
                    heat_map[i,j] = self.actions[self.Q[i,j,k,:].argmax()]
           
            heat_maps.append(heat_map)
        return heat_maps

class MaxTreasureMazeGameOffPolicy(MaxTreasureMazeGame):
    """
    Off policy MC control. This class defines the behavior and target policies.
    """

    def __init__(self, M,T ,epsilon, gamma) -> None:
        super().__init__(M,T, epsilon, gamma)

    def behavior_policy(self, i, j, v):
        """
        Behavior Policy: choose any action at random with uniform distribution 
        """
        return np.random.choice(self.actions)

    def target_policy(self, i, j, v):
        greedy_action = self.Q[i,j,v,:].argmax()
        return self.actions[greedy_action]
    
    def getISR(self,state,action):
        """
        for each iteration: 
        ISR = pi(At|St) / mu(At|St)
        for target policy: 
        pi(At|St) = 1;
        for behavior policy:
        mu(At|St) = 1/4
        """
        target_action = self.target_policy(state[0],state[1],state[2])
        if(target_action!=action):
            return 0
        return 4