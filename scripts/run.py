from grid_world import GridWorld
from dyna_q_agent import DynaQAgent
from utils import viz_trajectory

def create_env():
    # Define grid world parameters
    grid_size = (10, 10)
    obstacles = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]  # List of obstacles
    fake_goals = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 1)]  # List of obstacles
    goals = [(0, 4)]  # List of goals
    
    # Create a grid world environment
    env = GridWorld(grid_size=grid_size, obstacles=obstacles, goals=goals, fake_goals=fake_goals)

    return env

def train(env):
    # Create a Dyna-Q agent
    agent = DynaQAgent(env, alpha=0.1, gamma=0.2, epsilon=0.9, planning_steps=10)
    
    # Train the agent
    agent.train(num_episodes=1500)

    return agent

def evaluate(agent, env):
    # Get learned value function and model
    q_table = agent.q_table
    model = agent.model

    # Test how many steps are required to find the goal.
    num_evals = 10
    max_num_teststeps = 1500
    avg_steps = 0
    for eval in range(num_evals):
        env.reset()
        test_agent = DynaQAgent(
            env, q_table=q_table, model=model)
        test_agent.test(max_num_teststeps)
        avg_steps += len(test_agent.trajectory)
        print(f"Required steps for evaluation {eval+1}: {len(test_agent.trajectory)}")
        print(f'start: {test_agent.trajectory[0]}, end: {test_agent.trajectory[-1]}')
    print(f'Average steps required : {avg_steps/num_evals}')

if __name__ == "__main__":
    env = create_env()
    agent = train(env)
    evaluate(agent, env)

