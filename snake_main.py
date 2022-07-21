from gym import Env
from gym.spaces import MultiDiscrete, Box
import numpy as np
import mujoco_py
import cv2
from stable_baselines3 import PPO, A2C

class SnakeEnv(Env):
    def __init__(self):
        # Define snake model path
        self.snake_model_path = "Model/simplified-snake.xml"
        # Load snake model
        self.snake_model = mujoco_py.load_model_from_path(self.snake_model_path)
        # Load simulation engine
        self.sim = mujoco_py.MjSim(self.snake_model)
        # Visualize simulation environment
        self.viewer = mujoco_py.MjViewer(self.sim)
        # Actions we can take, left, stay, right
        self.action_space = MultiDiscrete([3, 3])
        # Distance array
        self.observation_space = Box(low=np.array([0]), high=np.array([72]))
        # Set goal pos
        self.goal = [6, 2.5]
        # Set start distance
        self.dist = (self.goal[0] - self.sim.data.qpos[6])**2 + (self.goal[1] - self.sim.data.qpos[7])**2
        self.previous_dist = self.dist
        self.steps = 2500

    def step(self, action):
        # Apply action
        self.previous_dist = self.dist
        self.sim.data.ctrl[0] = (action[0]-1) * 0.01
        self.sim.data.ctrl[1] = (action[1]-1) * 0.01
        self.sim.step()
        self.dist = (self.goal[0] - self.sim.data.qpos[6])**2 + (self.goal[1] - self.sim.data.qpos[7])**2
        # Reduce simulation length by 1 step
        self.steps -= 1

        # Calculate reward
        if self.dist < self.previous_dist:
            reward = 1
        else:
            reward = -1

        # Check if simulation is done
        if self.steps <= 0 or self.dist <= 2:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.dist, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            self.viewer.render()
        if mode == 'rgb_array':
            return self.viewer._read_pixels_as_in_window()

    def reset(self):
        # Reset pos
        # Load simulation engine
        self.sim = mujoco_py.MjSim(self.snake_model)
        # Visualize simulation environment
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.dist = (self.goal[0] - self.sim.data.qpos[6])**2 + (self.goal[1] - self.sim.data.qpos[7])**2
        self.previous_dist = self.dist
        self.steps = 2500

        return self.dist

if __name__ == "__main__":
    env = SnakeEnv()
    print("created environment")

    agent_model = PPO('MlpPolicy', env, verbose=1)
    # agent_model = A2C('MlpPolicy', env, verbose=1)
    print("created agent model")

    agent_model.learn(total_timesteps=25000)
    print("agent model learned 25000 steps")

    # agent_model.save("Snake_A2C")
    agent_model.save("Snake_PPO")
    print("saved agent model")

    del agent_model
    print("deleted agent model")

    agent_model = PPO.load("Snake_PPO", env=env)
    # agent_model = A2C.load("Snake_PPO", env=env)
    print("loaded agent model")

    obs = np.array([env.reset()])
    done = False
    img_array = []
    score = 0

    while not done:
        action, _states = agent_model.predict(obs, deterministic=True)
        # obs, rewards, done, info = env.step(env.action_space.sample())
        obs, rewards, done, info = env.step(action)
        obs = np.array([obs])
        # env.render()
        img = env.render(mode='rgb_array')

        img_array.append(img)
        score += rewards
    env.close()
    print(score)

    height, width, layers = img.shape
    size = (width,height)
    print(size)
    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 120, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()