import numpy as np
import random
import json

np.random.seed(1)

class AbelAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self._discount = 0.9
        self._alpha = 0.2
        self._epsilon = 0.99

        self._features = ['distance_to_goal']

        with open("weights.json", "r") as read_file:
            self._weights = json.load(read_file)

    def reset(self):
        self._epsilon = 0

    def act(self, obs):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        valid_objects = [tup for tup in enumerate(obs) if tup[1]['valid']]

        # Compute expected rewards for each valid_object
        expected_rewards = self._evaluate(valid_objects)

        self._last_expected_rewards = expected_rewards

        chosen_action = self._choose_action(expected_rewards)
        self._expected_distance_from_goal = obs[chosen_action]['distance_to_goal']

        # Based on expected rewards, choose a direction
        return chosen_action + 1

    def _evaluate(self, valid_objects):
        # Compute expected rewards for each valid_object
        expected_rewards = []

        for tup in valid_objects:
            index, object = tup
            # Evaluate weight
            expected_rewards.append((index, object['distance_to_goal'] * self._weights['distance_to_goal']))

        return expected_rewards

    def _choose_action(self, expected_rewards):
        best_actions = expected_rewards
        if random.random() > self._epsilon:
            max_reward = max(expected_rewards, key=lambda x: x[1])[1]
            best_actions = [a for a in expected_rewards if a[1] == max_reward]
        else:
            print("Making random Move")
            print(self._epsilon)
            self._epsilon = self._epsilon * self._epsilon

        return random.choice(best_actions)[0]

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        if len(self._last_expected_rewards) <= 1:
            return

        obs, next_obs, action, reward, done = memories

        # Get Data
        data = self._weights

        # Update  Metadata
        data['steps'] += 1

        # Update Features
        # DIFFERENCE = REWARD + (DISCOUNT * VALUE_OF_NEXT_STATE)
        print(f'reward: {reward}')
        difference = reward + (self._discount * (self._weights['distance_to_goal'] * self._expected_distance_from_goal - 1)) - (self._weights['distance_to_goal'] * self._expected_distance_from_goal)

        data['distance_to_goal'] = self._weights['distance_to_goal'] + self._alpha * difference * self._expected_distance_from_goal

        # Write Data
        with open('weights.json', 'w') as f:
            json.dump(data, f)

        # Unnecessary Operation But improves Readability
        self._weights = data

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return
