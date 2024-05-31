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


def run_all():

    # Define grid world parameters
    grid_size = (10, 10)
    obstacles = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)] 
    fake_goals = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 1)] 
    goals = [(0, 4)]  
    
    env = GridWorld(grid_size=grid_size, obstacles=obstacles, goals=goals, fake_goals=fake_goals)

    train_n_times = 20
    test_n_times = 20

    max_episodes = 3000

    avg_train = 0
    avg_test = 0

    best_train_val = max_episodes
    best_train_q = None
    best_train_model = None

    env = create_env()
    print('Train from scratch')
    for train in range(train_n_times):
        env.reset()
        train_agent = DynaQAgent(env)
        train_agent.train(num_episodes=max_episodes)

        # print(f'train val: {len(train_agent.trajectory)} for episode {train+1}')
        avg_train += len(train_agent.trajectory)
        if len(train_agent.trajectory) < best_train_val:
            best_train_val = len(train_agent.trajectory)
            best_train_q = train_agent.q_table
            best_train_model = train_agent.model
            # print(f'New best train: {best_train_val}')
    print(f'Average train val: {avg_train/train_n_times}')
    avg_train = 0

    print('Train with trained model.')
    for test in range(test_n_times):
        env.reset()
        test_agent = DynaQAgent(env, q_table=best_train_q, model=best_train_model)
        test_agent.train(num_episodes=max_episodes)

        # print(f'test val: {len(test_agent.trajectory)} for episode {test+1}')
        avg_test += len(test_agent.trajectory)
    print(f'Average test val: {avg_test/test_n_times}')
    avg_test = 0

    print('Now we move the goal and see if the agent can adapt faster using the learned model.')
    env.goals = [(0, 7)]

    print('Train from scratch')
    for train in range(train_n_times):
        env.reset()
        train_agent = DynaQAgent(env)
        train_agent.train(num_episodes=max_episodes)

        # print(f'train val: {len(train_agent.trajectory)} for episode {train+1}')
        avg_train += len(train_agent.trajectory)
    print(f'Average train val: {avg_train/train_n_times}')

    print('Train with frist trained model.')    
    for test in range(test_n_times):
        env.reset()
        test_agent = DynaQAgent(env, q_table=best_train_q, model=best_train_model)
        test_agent.train(num_episodes=max_episodes)

        # print(f'test val: {len(test_agent.trajectory)} for episode {test+1}')
        avg_test += len(test_agent.trajectory)
    print(f'Average test val: {avg_test/test_n_times}')

if __name__ == "__main__":
    run_all()

    # env = create_env()
    # agent = train(env)
    # evaluate(agent, env)

