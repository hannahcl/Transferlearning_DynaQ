
import matplotlib.pyplot as plt

def viz_trajectory(trajectory):
    # Extract x and y coordinates from the agent's trajectory
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]

    # Plot the agent's trajectory
    plt.plot(x, y)
    plt.title("Agent's Trajectory")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()