import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def himmelblau(x1, x2):
    """Himmelblau function"""
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


def ackley(x1, x2):
    """Ackley function"""
    return (
        -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2)))
        - torch.exp(0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2)))
        + torch.e
        + 20
    )


def rastrigin(x1, x2):
    """
    Rastrigin 函数
    """
    return (
        20
        + x1**2
        + x2**2
        - 10 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))
    )


# Using PyTorch to implement the gradient descent algorithm to find the function's minimum
def gradient_descent_pytorch(func, learning_rate, tolerance, max_iters, start_point):
    """
    Using PyTorch to implement the gradient descent algorithm to find the function's minimum
    :param func: The function to be optimized
    :param start_point: The starting point of the optimization
    :param learning_rate: The learning rate of the optimization
    :param max_iters: The maximum number of iterations of the optimization
    :param tolerance: The tolerance of the optimization
    :return: A list of the path taken during the optimization
    """
    # Initialize x1 and x2 as trainable parameters
    x = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)

    path = []  # Store the path of iterations
    # Calculate the initial function value
    y = func(x[0], x[1]).item()
    path.append([x[0].item(), x[1].item(), y])
    print(
        f"Initial position: x1 = {x[0].item():.6f}, x2 = {x[1].item():.6f}, f(x1,x2) = {y:.6f}"
    )

    for i in range(1, max_iters + 1):
        # Zero the gradients
        if x.grad is not None:
            x.grad.zero_()

        # Calculate the function value
        y = func(x[0], x[1])

        # Backpropagate to calculate gradients
        y.backward()

        # Get the gradients
        grad = x.grad.detach().clone()

        # Update parameters
        with torch.no_grad():
            x -= learning_rate * grad

        # Record the new position and function value
        new_x1, new_x2 = x[0].item(), x[1].item()
        new_y = func(x[0], x[1]).item()
        path.append([new_x1, new_x2, new_y])
        print(
            f"Iteration {i}: x1 = {new_x1:.6f}, x2 = {new_x2:.6f}, f(x1,x2) = {new_y:.6f}"
        )

        # Check if the norm of the gradient is less than the tolerance
        grad_norm = torch.norm(grad).item()
        if grad_norm < tolerance:
            print(
                f"The gradient is small enough ({grad_norm:.6f} < {tolerance}), stopping."
            )
            break

    return np.array(path)


# Plot function graph
def plot_function(func, x1_range, x2_range, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    X1, X2 = np.meshgrid(
        np.linspace(x1_range[0], x1_range[1], 100),
        np.linspace(x2_range[0], x2_range[1], 100),
    )
    X1_tensor = torch.tensor(X1, dtype=torch.float32)
    X2_tensor = torch.tensor(X2, dtype=torch.float32)
    with torch.no_grad():
        Z_tensor = func(X1_tensor, X2_tensor)
    Z = Z_tensor.numpy()

    ax.plot_surface(X1, X2, Z, cmap="viridis", rstride=1, cstride=1)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    plt.title(title)

    plt.show()


# Define main function
def main():
    # Define a dictionary of functions
    functions = {
        "1": {
            "func": himmelblau,
            "name": "Himmelblau function",
            "x1_range": [-10, 10],
            "x2_range": [-10, 10],
            "start_point": [0, 0],
        },
        "2": {
            "func": ackley,
            "name": "Ackley function",
            "x1_range": [-40, 40],
            "x2_range": [-40, 40],
            "start_point": [-2, -2],
        },
        "3": {
            "func": rastrigin,
            "name": "rastrigin function",
            "x1_range": [-40, 40],
            "x2_range": [-40, 40],
            "start_point": [-1.2, 1],
        },
    }

    # Select function
    print("Select the function for the gradient descent demonstration:")
    print("1. Himmelblau function")
    print("2. Ackley function")
    print("3. rastrigin function")
    choice = input("Enter your choice (1 ~ 3): ")
    if choice not in functions:
        print("Invalid choice, please try again.")
        return
    print(f'You have selected: {functions[choice]["name"]}')

    # Set gradient descent parameters
    learning_rate = 0.01
    tolerance = 1e-6
    max_iters = 1000

    print(functions[choice]["start_point"])
    # Reform gradient descent to obtain the path
    # path = gradient_descent_pytorch(functions[choice]['func'], learning_rate, tolerance, max_iters, functions[choice]['start_point'])

    # Plot function graph
    plot_function(
        functions[choice]["func"],
        functions[choice]["x1_range"],
        functions[choice]["x2_range"],
        functions[choice]["name"],
    )


if __name__ == "__main__":
    main()
