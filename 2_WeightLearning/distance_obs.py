import random
import time

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

from abel_agent import AbelAgent
import json
# random.seed(100)
np.random.seed(100)

class CustomWeightObserver(TreeObsForRailEnv):
    def __init__(self):
        super().__init__(max_depth=0)
        self.observation_space = [3]

    def reset(self):
        super().reset()

    def get(self, handle):
        agent = self.env.agents[handle]

        # Rather than calculate best possible path, let observation be arr of distances. Let Agent decide which is best.
        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)
        weights = []

        # Initialize weights
        for i in range(3):
            weights.append({
                'valid': True,
                'distance_to_goal': np.inf,
            })

        # Add distance_to_goal weight
        if num_transitions == 1:
            weights[0]['valid'] = False
            weights[1]['valid'] = True
            weights[2]['valid'] = False
        else:
            for index, direction in enumerate([(agent.direction + i) % 4 for i in range(-1, 2)]):
                current = weights[index]
                if possible_transitions[direction]:
                    new_position = self._new_position(agent.position, direction)
                    current['distance_to_goal'] = self.distance_map[handle, new_position[0], new_position[1], direction] / 100
                else:
                    current['valid'] = False
                    current['distance_to_goal'] = np.inf

        return weights

seed = random.randint(0,2**32)
print(f"Seed: {seed}")
env = RailEnv(width=20,
              height=20,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=10, min_dist=5, max_dist=99999, seed=seed),
              schedule_generator=complex_schedule_generator(),
              number_of_agents=1,
              obs_builder_object=CustomWeightObserver())

env_renderer = RenderTool(env, gl="PILSVG")

agent = AbelAgent(218, 5)
n_trials = 50

for trials in range(1, n_trials + 1):
    # Reset Environment
    obs = env.reset()
    env_renderer.reset()
    env_renderer.render_env(show=True, frames=True, show_observations=True)
    score = 0

    # Run Episode
    for step in range(100):
        action = agent.act(obs[0])
        next_obs, all_rewards, done, _ = env.step({0: action})
        score+=all_rewards[0]
        agent.step((obs[0], next_obs[0], action, all_rewards[0], done[0]))
        obs = next_obs.copy()
        env_renderer.render_env(show=True, frames=True, show_observations=True)
        time.sleep(0.1)
        if done["__all__"]:
            break

    agent.reset()

    print(f'Episode Nr. {trials}\t Score = {score}')

    # Update Metadata
    #   Read Data
    data = {}
    with open("weights.json", "r") as read_file:
        data = json.load(read_file)

    data['trials'] += 1
    #   Write Data

    with open('weights.json', 'w') as f:
        json.dump(data, f)

    # Update Score Log
    f = open("scores.txt", "a+")
    f.write(str(score) + "\n")

env_renderer.close_window()
