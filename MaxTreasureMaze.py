import numpy as np

class MaxTreasureMazeGame:

    maze = []
    maze_size = 25
    treasure_count = 5
    treasure_location = []
    actions = ["U","D","L","R"]

    def __init__(self,M,epsilon) -> None:
        self.M = M
        self.epsilon = epsilon
        self.initialize()
    
    def initialize(self):
        self.Q = np.zeros((
            self.maze_size,self.maze_size,self.treasure_count,len(self.actions)
        ))

        self.returns = np.zeros_like(self.Q,dtype=list)

    def behavior_policy(self,i,j,action):
        """
        return an action depending on location of agent
        """
        if(action==None):
            return "R"
        if((action=="R" and j < self.M-1 and self.maze[i,j+1]==1)):
            return "R"
        elif(action=="R"):
            action = "D"
        if((action=="D" and i < self.M-1 and self.maze[i+1,j]==1) ):
            return "D"
        elif(action=="D"):
            action = "L"
        if(action == "L" and j > 0 and self.maze[i,j-1]==1):
            return "L"
        elif(action=="L"):
            action = "U"
        if(action=="U" and i > 0 and self.maze[i-1,j]==1):
            return "U"
        elif(action=="U"):
            return self.behavior_policy(i,j,"R")
        
    def target_policy(self,i,j,action):
        """
        same for on-policy methods
        """
        return self.behavior_policy(i=i,j=j,action=action)

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
            action = self.behavior_policy(i=i,j=j)


    def mc_control(self):
        """MC Control for learning state-action function"""
        self.initialize()

        for m in range(self.M+1):
            states,actions,rewards = self.generate_episode()

