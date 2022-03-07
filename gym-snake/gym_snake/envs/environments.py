from gym_snake.envs.snake_env import SnakeEnv

class Snake_16x16(SnakeEnv):
    def __init__(self):
        super().__init__(grid_size=10)

