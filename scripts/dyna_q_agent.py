import numpy as np
import random
from grid_world import GridWorld

class DynaQAgent:
    def __init__(
            self,
            env: GridWorld,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.9,
            planning_steps=10, 
            q_table=None, 
            model=None
        ):
        """
        Initialize the Dyna-Q agent.
        
        Parameters:
        - env: The environment the agent interacts with.
        - alpha: Learning rate.
        - gamma: Discount factor.
        - epsilon: Exploration rate.
        - planning_steps: Number of planning steps to perform per real step.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        if q_table is not None: 
            self.q_table = q_table
        else:
            self.q_table = np.zeros((env.grid_size[0], env.grid_size[1], 4))

        if model is not None:
            self.model = model
        else:
            self.model = {} 

        self.trajectory = []

        self.state = env.agent_position

        self.required_episodes = 0

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
        
        Parameters:
        - state: Current state of the agent.
        
        Returns:
        - action: Action chosen by the agent.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])  # Explore: choose a random action
        else:
            row, col = state
            return np.argmax(self.q_table[row, col, :])  # Exploit: choose the best action

    def learn(self, state, action, reward, next_state):
        self.learn_Q(state, action, reward, next_state)
        self.learn_model(state, action, reward, next_state)

    def learn_Q(self, state, action, reward, next_state):
        """
        Update the Q-value based on the agent's experience.
        
        Parameters:
        - state: Current state.
        - action: Action taken.
        - reward: Reward received.
        - next_state: Next state after taking the action.
        """
        # Update the Q-value
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.argmax(self.q_table[next_row, next_col, :])
        td_target = reward + self.gamma * self.q_table[next_row, next_col, best_next_action]
        self.q_table[row, col, action] += self.alpha * (td_target - self.q_table[row, col, action])

    def learn_model(self, state, action, reward, next_state):
        
        # Update the model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state)

    def planning(self):
        """
        Perform planning steps using the model.
        """
        for _ in range(self.planning_steps):
            # Randomly sample a previously observed state and action
            state = random.choice(list(self.model.keys()))
            action = random.choice(list(self.model[state].keys()))
            reward, next_state = self.model[state][action]
            
            # Update the Q-value based on the simulated experience
            self.learn_Q(state, action, reward, next_state)

    def train(self, num_episodes):
        """
        Train the agent for a specified number of episodes.
        
        Parameters:
        - num_episodes: Number of episodes to train the agent.
        """
        state = self.env.agent_position
        done = False

        self.trajectory.append(state)
        for episode in range(num_episodes):
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            self.trajectory.append(next_state)
            self.state = state

            if done:
                break

            self.learn(state, action, reward, next_state)
            self.planning()  # Perform planning steps
            state = next_state
        

    def test(self, num_episodes):
        state = self.env.agent_position

        for episode in range(num_episodes):
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            self.trajectory.append(next_state)
            state = next_state

            if done:
                self.required_episodes = episode
                break





