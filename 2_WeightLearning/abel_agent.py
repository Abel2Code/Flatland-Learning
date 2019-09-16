import numpy as np
import json

np.random.seed(1)

class AbelAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, obs):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        valid_objects = [tup for tup in enumerate(obs)]

        # Compute expected rewards for each valid_object
        expected_rewards = [np.NINF] * len(obs)

        for tup in valid_objects:
            index, object = tup
            expected_rewards[index] = -object['distance_to_goal']

        # Based on expected rewards, choose a direction
        return np.argmax(expected_rewards) + 1

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        obs, reward, done = memories

        # Get Data
        data = {}
        with open("weights.json", "r") as read_file:
            data = json.load(read_file)

        # Update Data Object
        data['steps'] += 1

        # Write Data
        with open('weights.json', 'w') as f:
            json.dump(data, f)

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return
