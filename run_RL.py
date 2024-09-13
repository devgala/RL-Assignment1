from MaxTreasureMaze import MaxTreasureMazeGame,MaxTreasureMazeGameOffPolicy
import matplotlib.pyplot as plt
game = MaxTreasureMazeGame(100,1e9,0.5,1)
# print(type(game.maze))
# print(game.Q[:,:,0,0])
game.mc_control()
heat_maps = game.create_heat_map()

for i in range(0,3):
    plt.imshow(heat_maps[i],cmap='hot', interpolation='nearest')
    plt.savefig(f"heat_map_{i}.png")