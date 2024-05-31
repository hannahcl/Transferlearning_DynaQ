from grid_world import GridWorld
from dyna_q_agent import DynaQAgent
from utils import viz_trajectory

def show_grid_world():
    # Define grid world parameters
    grid_size = (5, 5)
    obstacles = [(1, 1), (2, 2)]  # List of obstacles
    goals = [(4, 4)]  # List of goals
    
    # Create a grid world environment
    env = GridWorld(grid_size=grid_size, obstacles=obstacles, goals=goals)
    
    # Reset environment
    env.reset()
    env.render()
    
    # Perform actions in the environment
    actions = [3, 1, 3, 3]  # Example actions (right, down, right, right)
    for action in actions:
        next_state, reward, done = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

def train_and_evaluate():
    # Define grid world parameters
    grid_size = (10, 10)
    obstacles = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]  # List of obstacles
    fake_goals = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 1)]  # List of obstacles
    goals = [(0, 4)]  # List of goals
    
    # Create a grid world environment
    env = GridWorld(grid_size=grid_size, obstacles=obstacles, goals=goals, fake_goals=fake_goals)
    
    # Create a Dyna-Q agent
    agent = DynaQAgent(env, alpha=0.1, gamma=0.2, epsilon=0.9, planning_steps=10)
    
    # Train the agent
    agent.train(num_episodes=1500)
    
    return agent, env

if __name__ == "__main__":
    # show_grid_world()
    agent, env = train_and_evaluate()
    env.visualize_trajectory(agent.trajectory)

