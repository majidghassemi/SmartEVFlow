import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
import random
import time
import math
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics

class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardPlotCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

class EVRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(EVRoutingEnv, self).__init__()
        # Graph settings
        self.total_nodes = 50
        self.nodes = list(range(1, self.total_nodes + 1))
        self.source = 1
        self.destination = 22

        p = 0.1  # edge probability
        while True:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.nodes)
            for i in self.nodes:
                for j in self.nodes:
                    if i != j and random.random() < p:
                        distance = random.uniform(10, 20)
                        edge_data = {'angle': random.uniform(0, 10), 'air_density': 1.205}
                        self.graph.add_edge(i, j, distance=distance, edge_data=edge_data)
            if nx.has_path(self.graph, self.source, self.destination):
                break

        # Fixed charging station settings.
        self.charging_stations = {
            3: {'charging_speed': 100, 'waiting_time': 8},
            7: {'charging_speed': 100, 'waiting_time': 12},
            17: {'charging_speed': 100, 'waiting_time': 7}
        }

        # Vehicle parameters
        self.mass = 1800             # kg
        self.speed = 50              # km/h
        self.mass_factor = 1.1       
        self.rolling_resistance = 0.01
        self.drag_coefficient = 0.6
        self.cross_sectional_area = 3.5
        self.battery_capacity = 100  # kWh

        # Simulation parameters
        self.initial_battery = 20.0  # kWh
        self.battery_threshold = 5.0  # kWh (5% of capacity)
        self.current_node = self.source
        self.battery_level = self.initial_battery

        # Tracking metrics
        self.total_time = 0.0                  # minutes
        self.total_energy_consumed = 0.0         # kWh consumed while traveling
        self.total_distance_traveled = 0.0       # km
        self.total_energy_added = 0.0          # kWh added during charging
        self.path = [self.source]              # complete path (nodes visited)
        self.visited_charging_stations = []    # charging stations visited

        # Simplified consumption and travel time parameters:
        self.energy_per_km = 0.267
        self.time_per_km = 60 / 50 

        # Action space:
        # Actions 0 to (total_nodes - 1) correspond to moving to nodes [1 ... 13],
        # and action total_nodes means "charge".
        self.action_space = spaces.Discrete(len(self.nodes) + 1)

        # Observation space: [current_node, battery_level]
        low_obs = np.array([1, 0], dtype=np.float32)
        high_obs = np.array([self.total_nodes, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.done = False

    def reset(self, seed=None, options=None):
        self.current_node = self.source
        self.battery_level = self.initial_battery
        self.total_time = 0.0
        self.total_energy_consumed = 0.0
        self.total_distance_traveled = 0.0
        self.total_energy_added = 0.0
        self.path = [self.source]
        self.visited_charging_stations = []
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.current_node, self.battery_level], dtype=np.float32)

    def step(self, action):
        """
        Action meanings:
          - If action equals len(self.nodes): attempt to charge.
          - Otherwise, action corresponds to moving to the node self.nodes[action].
        Returns: observation, reward, terminated, truncated, info.
        """
        reward = 0.0
        info = {}

        if self.done:
            return self._get_obs(), 0.0, True, False, info

        if action == len(self.nodes):
            if self.current_node in self.charging_stations:
                cs = self.charging_stations[self.current_node]
                charging_speed = cs['charging_speed']   # kW
                waiting_time = cs['waiting_time']         # minutes
                charge_needed = self.battery_capacity - self.battery_level
                charging_time = waiting_time + (charge_needed / charging_speed) * 60
                self.total_time += charging_time
                reward -= charging_time  # penalty for charging time
                self.total_energy_added += charge_needed
                self.battery_level = self.battery_capacity
                if self.current_node not in self.visited_charging_stations:
                    self.visited_charging_stations.append(self.current_node)
            else:
                reward -= 10  # invalid charge action
        else:
            # Move action: target node is self.nodes[action]
            target_node = self.nodes[action]
            if self.graph.has_edge(self.current_node, target_node):
                edge_data = self.graph.get_edge_data(self.current_node, target_node)
                distance = edge_data['distance']
                energy_consumption = distance * self.energy_per_km
                travel_time = distance * self.time_per_km
                if self.battery_level < energy_consumption:
                    reward -= 50
                    info['reason'] = 'Insufficient battery for the move'
                    self.done = True
                else:
                    self.battery_level -= energy_consumption
                    self.total_energy_consumed += energy_consumption
                    self.total_time += travel_time
                    self.total_distance_traveled += distance
                    reward -= travel_time
                    self.current_node = target_node
                    self.path.append(target_node)
            else:
                reward -= 10

        if self.current_node == self.destination:
            reward += 50
            self.done = True
        if self.battery_level <= 0:
            reward -= 50
            info['reason'] = 'Battery depleted'
            self.done = True

        return self._get_obs(), reward, self.done, False, info

    def render(self, mode="human"):
        print(
            f"Node: {self.current_node:>2}, Battery: {self.battery_level:>5.1f}%, "
            f"Time: {self.total_time:>6.1f} min, Energy: {self.total_energy_consumed:>6.2f} kWh, "
            f"Distance: {self.total_distance_traveled:>6.2f} km"
        )


if __name__ == "__main__":
    env = EVRoutingEnv()
    env = RecordEpisodeStatistics(env)
    check_env(env, warn=True)

    reward_callback = RewardPlotCallback()

    # Create the PPO agent.
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the PPO agent and record episode rewards via the callback.
    model.learn(total_timesteps=1e5, callback=reward_callback)

    # Plot the reward curve.
    plt.figure()
    plt.plot(reward_callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Episode Rewards")
    plt.show()

    # Test the learned policy.
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.5)

    # Final summary output in the desired style.
    print("\nFinal Summary:")
    print(f"Complete path: {env.env.path}")
    print(f"Visited charging stations: {env.env.visited_charging_stations}")
    print(f"Total distance: {env.env.total_distance_traveled:.2f} KM")
    print(f"Total time is: {env.env.total_time:.2f} Minute")
    print(f"Charge at destination is {env.env.battery_level:.2f}%")
    print(f"Total charge consumed from source to destination: {env.env.total_energy_consumed:.2f} kWh")
    print(f"Total energy added at charging stations: {env.env.total_energy_added:.2f} kWh")
    print(f"Episode finished with total reward: {total_reward}")
    if 'reason' in info:
        print("Episode ended due to:", info['reason'])
    print("Episode rewards:", max(reward_callback.episode_rewards))