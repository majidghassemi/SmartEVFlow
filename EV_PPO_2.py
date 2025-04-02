# import gymnasium as gym
# import numpy as np
# import networkx as nx
# from gymnasium import spaces
# import time
# import math

# # Import PPO from stable-baselines3 (install via pip install stable-baselines3)
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env


# class EVRoutingEnv(gym.Env):
#     """
#     A simplified electric vehicle routing environment with a fixed graph.

#     Graph properties:
#       - Nodes: fixed set from the provided edges, namely:
#           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14}.
#       - Fixed directed edges with given distances:
#           (0,1,42.0), (1,3,45.0), (1,4,32.0),
#           (2,3,55.0), (2,4,54.0),
#           (3,2,55.0), (3,1,45.0),
#           (4,2,54.0), (4,1,32.0), (4,7,52.0),
#           (5,6,36.0), (5,11,53.0),
#           (6,5,36.0),
#           (7,10,53.0), (7,4,52.0), (7,8,30.0),
#           (8,13,39.0),
#           (9,11,36.0).
#       - Additionally, an edge (13,14,20.0) is added to ensure connectivity.
#       - Source is node 0 and destination is node 14.
    
#     Charging Stations:
#       - Fixed charging stations are:
#             charging_nodes(2, charging_speed=100, waiting_time=8)
#             charging_nodes(4, charging_speed=100, waiting_time=12)
#             charging_nodes(13, charging_speed=100, waiting_time=7)
#             charging_nodes(14, charging_speed=100, waiting_time=12)
    
#     Simulation Parameters:
#       - Initial Battery Level: 50 kWh.
#       - Battery Threshold: 5 kWh (if a move would drop battery below this, the move is disallowed).
#       - Energy consumption: 0.267 kWh/km.
#       - Travel time: 1.2 minutes/km.
    
#     The agent’s action space allows moving to any node (if an edge exists) or charging when available.
#     The final summary prints:
#        Complete path, visited charging stations, total distance, total time,
#        battery level at destination, total energy consumed, and energy added during charging.
#     """
#     metadata = {"render_modes": ["human"]}

#     def __init__(self):
#         super(EVRoutingEnv, self).__init__()

#         # Fixed graph based on provided edges.
#         self.graph = nx.DiGraph()
#         edges = [
#             (0, 1, 42.0),
#             (1, 3, 45.0),
#             (1, 4, 32.0),
#             (2, 3, 55.0),
#             (2, 4, 54.0),
#             (3, 2, 55.0),
#             (3, 1, 45.0),
#             (4, 2, 54.0),
#             (4, 1, 32.0),
#             (4, 7, 52.0),
#             (5, 6, 36.0),
#             (5, 11, 53.0),
#             (6, 5, 36.0),
#             (7, 10, 53.0),
#             (7, 4, 52.0),
#             (7, 8, 30.0),
#             (8, 13, 39.0),
#             (9, 11, 36.0)
#         ]
#         for (u, v, d) in edges:
#             # Here edge_data is kept simple.
#             self.graph.add_edge(u, v, distance=d, edge_data={'angle': 0.0, 'air_density': 1.205})
#         # Ensure node 14 exists and add edge (13,14,20.0) for connectivity.
#         self.graph.add_node(14)
#         if not self.graph.has_edge(13, 14):
#             self.graph.add_edge(13, 14, distance=20.0, edge_data={'angle': 0.0, 'air_density': 1.205})

#         # Get sorted list of nodes.
#         self.nodes = sorted(self.graph.nodes)
#         # Set source and destination.
#         self.source = 0
#         self.destination = 13

#         # Fixed charging station settings.
#         self.charging_stations = {
#             2: {'charging_speed': 100, 'waiting_time': 8},
#             4: {'charging_speed': 100, 'waiting_time': 12},
#             13: {'charging_speed': 100, 'waiting_time': 7},
#             14: {'charging_speed': 100, 'waiting_time': 12}
#         }

#         # Vehicle parameters.
#         self.mass = 1800
#         self.speed = 50
#         self.mass_factor = 1.1
#         self.rolling_resistance = 0.01
#         self.drag_coefficient = 0.6
#         self.cross_sectional_area = 3.5
#         self.battery_capacity = 100  # maximum battery (kWh)

#         # Simulation parameters.
#         self.initial_battery = 50.0  # kWh
#         self.battery_threshold = 5.0  # kWh
#         self.current_node = self.source
#         self.battery_level = self.initial_battery

#         # Tracking metrics.
#         self.total_time = 0.0         # minutes
#         self.total_energy_consumed = 0.0  # kWh consumed in travel
#         self.total_distance_traveled = 0.0  # km
#         self.total_energy_added = 0.0   # kWh added during charging
#         self.path = [self.source]       # complete path (nodes visited)
#         self.visited_charging_stations = []  # charging stations visited

#         # Simplified consumption/travel time.
#         self.energy_per_km = 0.267  # kWh per km
#         self.time_per_km = 60 / 50  # 1.2 minutes per km

#         # Action space: one action per node (move to that node) plus one for charging.
#         self.action_space = spaces.Discrete(len(self.nodes) + 1)
#         low_obs = np.array([min(self.nodes), 0], dtype=np.float32)
#         high_obs = np.array([max(self.nodes), self.battery_capacity], dtype=np.float32)
#         self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

#         self.done = False

#     def reset(self, seed=None, options=None):
#         self.current_node = self.source
#         self.battery_level = self.initial_battery
#         self.total_time = 0.0
#         self.total_energy_consumed = 0.0
#         self.total_distance_traveled = 0.0
#         self.total_energy_added = 0.0
#         self.path = [self.source]
#         self.visited_charging_stations = []
#         self.done = False
#         return self._get_obs(), {}

#     def _get_obs(self):
#         return np.array([self.current_node, self.battery_level], dtype=np.float32)

#     def step(self, action):
#         """
#         Action meanings:
#           - If action equals len(self.nodes): attempt to charge.
#           - Otherwise, action corresponds to moving to self.nodes[action].
#         Returns: observation, reward, terminated, truncated, info.
#         """
#         reward = 0.0
#         info = {}

#         if self.done:
#             return self._get_obs(), 0.0, True, False, info

#         # Charging action.
#         if action == len(self.nodes):
#             if self.current_node in self.charging_stations:
#                 cs = self.charging_stations[self.current_node]
#                 charging_speed = cs['charging_speed']
#                 waiting_time = cs['waiting_time']
#                 energy_to_add = self.battery_capacity - self.battery_level
#                 charging_time = waiting_time + (energy_to_add / charging_speed) * 60
#                 self.total_time += charging_time
#                 reward -= charging_time
#                 self.total_energy_added += energy_to_add
#                 self.battery_level = self.battery_capacity
#                 if self.current_node not in self.visited_charging_stations:
#                     self.visited_charging_stations.append(self.current_node)
#             else:
#                 reward -= 10  # invalid charge action
#         else:
#             # Move action.
#             target_node = self.nodes[action]
#             if self.graph.has_edge(self.current_node, target_node):
#                 edge_data = self.graph.get_edge_data(self.current_node, target_node)
#                 distance = edge_data['distance']
#                 energy_consumption = distance * self.energy_per_km
#                 travel_time = distance * self.time_per_km

#                 # Prevent move if battery would drop below threshold.
#                 if self.battery_level - energy_consumption < self.battery_threshold:
#                     reward -= 50
#                     info['reason'] = 'Battery would drop below threshold'
#                     self.done = True
#                 else:
#                     self.battery_level -= energy_consumption
#                     self.total_energy_consumed += energy_consumption
#                     self.total_time += travel_time
#                     self.total_distance_traveled += distance
#                     reward -= travel_time
#                     self.current_node = target_node
#                     self.path.append(target_node)
#             else:
#                 reward -= 10  # invalid move

#         # Termination conditions.
#         if self.current_node == self.destination:
#             reward += 100
#             self.done = True
#         if self.battery_level <= 0:
#             reward -= 200
#             info['reason'] = 'Battery depleted'
#             self.done = True

#         return self._get_obs(), reward, self.done, False, info

#     def render(self, mode="human"):
#         print(
#             f"Node: {self.current_node:>2}, Battery: {self.battery_level:>5.1f} kWh, "
#             f"Time: {self.total_time:>6.1f} min, Energy consumed: {self.total_energy_consumed:>6.2f} kWh, "
#             f"Distance: {self.total_distance_traveled:>6.2f} km"
#         )


# if __name__ == "__main__":
#     # Create and check the environment.
#     env = EVRoutingEnv()
#     check_env(env, warn=True)

#     # Create the PPO agent using an MLP policy.
#     model = PPO("MlpPolicy", env, verbose=1)

#     # Train the PPO agent.
#     model.learn(total_timesteps=10000)

#     # Test the learned policy.
#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#         env.render()
#         time.sleep(0.5)

#     # Final summary output in the desired style.
#     print("\nFinal Summary:")
#     print(f"Complete path: {env.path}")
#     print(f"Visited charging stations: {env.visited_charging_stations}")
#     print(f"Total distance: {env.total_distance_traveled:.2f} KM")
#     print(f"Total time is: {env.total_time:.2f} Minute")
#     print(f"Charge at destination is {env.battery_level:.2f}%")
#     print(f"Total charge consumed from source to destination: {env.total_energy_consumed:.2f} kWh")
#     print(f"Total energy added at charging stations: {env.total_energy_added:.2f} kWh")
#     print(f"Episode finished with total reward: {total_reward}")
#     if 'reason' in info:
#         print("Episode ended due to:", info['reason'])
