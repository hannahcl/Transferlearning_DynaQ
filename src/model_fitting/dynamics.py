import numpy as np
import sys
import matplotlib.pyplot as plt


def model(x, y):
    bound = 1
    alpha = 1
    beta = 1

    dx = -alpha*x*np.maximum(0, np.abs(x) - bound)
    dy = -beta*y
    return (dx, dy)

def streamplot():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    U, V = model(X, Y)

    plt.figure(figsize=(10, 6))
    plt.streamplot(X, Y, U, V, density=1.5, linewidth=1, arrowsize=1)
    plt.title('Streamplot of the Differential Equations')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.plot([-1, 1], [0, 0], color='black', linewidth=2)  # Add thick line between (-1, 0) and (1, 0)
    plt.grid(True)
    plt.show()

def sim(dt, end_time, x0, y0):
    dt = 0.01
    t = np.arange(0, end_time, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)

    x[0] = x0
    y[0] = y0

    for k in range(len(t)-1):
        dx, dy = model(x[k], y[k])
        x[k+1] = x[k] + dx*dt
        y[k+1] = y[k] + dy*dt 

    return t, x, y  

def example_trajectories():
    dt = 0.01
    end_time = 100

    # Initial conditions a
    x0_a = -1.
    y0_a = 1.

    # Initial conditions b
    x0_b = 0.5
    y0_b = 0.

    t, x_a, y_a = sim(dt, end_time, x0_a, y0_a)
    _, x_b, y_b = sim(dt, end_time, x0_b, y0_b)

    plt.figure(figsize=(12, 4))

    # (a) 
    plt.subplot(1, 3, 1)
    plt.plot(t, x_a, label=f'Initial condition a')
    plt.plot(t, x_b, label=f'Initial condiction b')
    plt.xlabel('Time (t)')
    plt.ylabel('$x(t)$')
    plt.title('$x(t)$ against time')
    plt.legend()

    # (b) 
    plt.subplot(1, 3, 2)
    plt.plot(t, y_a, label=f'Initial condition a')
    plt.plot(t, y_b, label=f'Initial condiction b')
    plt.xlabel('Time (t)')
    plt.ylabel('$y(t)$')
    plt.title('$y(t)$ against time')
    plt.legend()

    # (v) 
    plt.subplot(1, 3, 3)
    plt.plot(x_a, y_a, label=f'Initial condition a')
    plt.plot(x_b, y_b, label=f'Initial condiction b')
    plt.xlabel('$x(t)$')
    plt.ylabel('$y(t)$')
    plt.title('$y(t)$ against $x(t)$')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "streamplot":
            streamplot()
        elif arg == "traj":
            example_trajectories()
        else:
            print("Invalid argument. Please choose either 'streamplot' or 'example_trajectories'.")
    else:
        print("No argument provided. Please provide either 'streamplot' or 'example_trajectories'.")