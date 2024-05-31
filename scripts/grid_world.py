import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, grid_size=(5,5), obstacles=[], goals=[], fake_goals=[]):
        """
        Initialize the grid world environment.
        
        Parameters:
        - grid_size: Tuple, size of the grid (rows, columns).
        - obstacles: List of tuples, positions of obstacles in the grid.
        - goals: List of tuples, positions of goals in the grid.
        """
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.fake_goals = fake_goals 
        self.goals = goals
        self.agent_position = (0, 0)  # Initial agent position
        
    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.agent_position = (0, 0)
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Parameters:
        - action: Integer, action to take (0: up, 1: down, 2: left, 3: right).
        
        Returns:
        - next_state: Tuple, next state after taking action.
        - reward: Float, reward received after taking action.
        - done: Boolean, whether the episode is done.
        """
        row, col = self.agent_position
        reward = 0
        done = False
        
        if action == 0:  # Up
            next_state = (min(row + 1, self.grid_size[0]-1), col)
        elif action == 1:  # Down
            next_state = (max(row - 1, 0), col)
        elif action == 2:  # Left
            next_state = (row, min(col + 1, self.grid_size[1]-1))
        elif action == 3:  # Right
            next_state = (row, max(col - 1, 0))
        else:
            raise ValueError("Invalid action. Action should be in [0, 1, 2, 3].")
   
        # Check if next state is an obstacle
        if next_state in self.obstacles:
            reward = -1  # Penalty for hitting an obstacle
            next_state = self.agent_position
        else:
            self.agent_position = next_state
            
            # Check if next state is a goal
            if next_state in self.goals:
                reward = 1
                done = True
                # print(f"Goal reached!")

            if next_state in self.fake_goals:
                reward = 0.5
        
        return self.agent_position, reward, done
    
    def render(self):
        """
        Render the current state of the grid world.
        """
        grid = np.zeros(self.grid_size)
        
        # Mark obstacles with -1
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Mark goals with +1
        for goal in self.goals:
            grid[goal] = 1
        
        # Mark agent with +0.5
        grid[self.agent_position] = 0.5
        
        print(grid)

    def visualize_trajectory(self, trajectory):
        """
        Visualize the trajectory of the agent along with obstacles, start, and goal positions.
        
        Parameters:
        - trajectory: List of positions representing the agent's trajectory.
        """
        rows, cols = self.grid_size
        grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        
        for (x, y) in self.obstacles:
            grid[y][x] = 'O'  # Mark obstacles with 'O'
        
        sx, sy = self.agent_position
        grid[sy][sx] = 'S'  # Mark start with 'S'
        
        for (x, y) in self.goals:
            grid[y][x] = 'G'  # Mark goals with 'G'

        for (x, y) in self.fake_goals:
            grid[y][x] = 'Y'  # Mark fake goals with 'G'
        
        for (x, y) in trajectory:
            if (x, y) != self.agent_position and (x, y) not in self.goals:
                grid[y][x] = '*'  # Mark trajectory with '*'
        
        plt.figure(figsize=(10, 10))
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == 'O':
                    plt.fill_between([x, x+1], y, y+1, color='black')
                elif grid[y][x] == 'S':
                    plt.text(x + 0.5, y + 0.5, 'S', ha='center', va='center', fontsize=12, color='green')
                elif grid[y][x] == 'G':
                    plt.text(x + 0.5, y + 0.5, 'G', ha='center', va='center', fontsize=12, color='red')
                elif grid[y][x] == 'Y':
                    plt.text(x + 0.5, y + 0.5, 'G', ha='center', va='center', fontsize=12, color='yellow')
                elif grid[y][x] == '*':
                    plt.fill_between([x, x+1], y, y+1, color='blue', alpha=0.5)
        
        plt.xlim(0, cols)
        plt.ylim(0, rows)
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.title(f"Trajectory with {len(trajectory)} steps")
        plt.show()


