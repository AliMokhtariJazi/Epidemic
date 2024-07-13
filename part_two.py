import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_nodes = 1000
edge_prob = 0.05
initial_infected = 5
infection_prob = 0.1
recovery_prob = 0.05

num_steps = 30

# Generate Erdős-Rényi network
G = nx.erdos_renyi_graph(num_nodes, edge_prob)

# Initialize states: 0 for susceptible, 1 for infected
states = np.zeros(num_nodes, dtype=int)
initial_infected_nodes = np.random.choice(num_nodes, initial_infected, replace=False)
states[initial_infected_nodes] = 1

def simulate_infection(G, states, infection_prob, recovery_prob, num_steps):
    """
    Simulates the infection spread in the network.

    Parameters:
    - G: NetworkX graph
    - states: Initial states of the nodes
    - infection_prob: Probability of infection spreading
    - recovery_prob: Probability of recovery
    - num_steps: Number of time steps for the simulation

    Returns:
    - history: List of states at each time step
    - edge_history: List of edges that transmitted the infection at each time step
    - prevalence: List of prevalence values at each time step
    """
    history = []
    edge_history = []
    prevalence = []

    for _ in range(num_steps):
        new_states = states.copy()
        edges_infected = []

        for node in range(len(states)):
            if states[node] == 1:  # Infected node
                for neighbor in G.neighbors(node):
                    if states[neighbor] == 0 and np.random.rand() < infection_prob:
                        new_states[neighbor] = 1
                        edges_infected.append((node, neighbor))
                if np.random.rand() < recovery_prob:
                    new_states[node] = 0  # Node recovers and becomes susceptible again

        states = new_states
        history.append(states.copy())
        edge_history.append(edges_infected)
        prevalence.append(np.mean(states))

    return history, edge_history, prevalence

def plot_prevalence():
    """
    Plots the prevalence of infected individuals over time.
    """
    _, _, prevalence = simulate_infection(G, states, infection_prob, recovery_prob, num_steps)

    plt.figure(figsize=(10, 6))
    plt.plot(prevalence, label='Prevalence')
    plt.xlabel('Time Steps')
    plt.ylabel('Prevalence (Fraction of Infected Individuals)')
    plt.title('Disease Spread in an Erdős-Rényi Network')
    plt.legend()
    plt.grid(True)
    plt.savefig('infection_simulation_prevalence.pdf')
    plt.show()

def create_animation():
    """
    Creates and saves an animation of the infection spread.
    """
    history, edge_history, prevalence = simulate_infection(G, states, infection_prob, recovery_prob, num_steps)

    def update(num, history, edge_history, prevalence, graph, pos, ax1, ax2):
        ax1.clear()
        current_states = history[num]
        edges_infected = edge_history[num]
        
        colors = ['blue' if state == 0 else 'red' for state in current_states]
        edge_colors = ['gray' for _ in range(len(graph.edges()))]
        edge_widths = [0.5 for _ in range(len(graph.edges()))]

        for edge in edges_infected:
            try:
                index = list(graph.edges()).index(edge)
                edge_colors[index] = 'orange'
                edge_widths[index] = 2
            except ValueError:
                continue

        nx.draw(graph, pos, node_color=colors, edge_color=edge_colors, width=edge_widths, with_labels=False, node_size=10, ax=ax1)
        ax1.set_title(f'Time Step {num + 1}')

        ax2.clear()
        ax2.plot(prevalence[:num + 1], color='blue')
        ax2.set_xlim(0, num_steps)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Prevalence')
        ax2.set_title('Prevalence Over Time')
        ax2.grid(True)

    pos = nx.spring_layout(G)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ani = animation.FuncAnimation(fig, update, frames=num_steps, fargs=(history, edge_history, prevalence, G, pos, ax1, ax2), interval=1000, repeat=False)

    ani.save(f'infection_simulation_with num_nodes={num_nodes} infection_rate={infection_prob} treatment_rate={recovery_prob} initial_infected={initial_infected}.mp4', writer='ffmpeg')
    # plt.show()

def create_snapshots():
    """
    Creates and saves snapshots of the infection spread for the first 6 time steps.
    """
    history, edge_history, prevalence = simulate_infection(G, states, infection_prob, recovery_prob, num_steps)
    pos = nx.spring_layout(G)
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        current_states = history[i]
        edges_infected = edge_history[i]
        
        colors = ['blue' if state == 0 else 'red' for state in current_states]
        edge_colors = ['gray' for _ in range(len(G.edges()))]
        edge_widths = [0.5 for _ in range(len(G.edges()))]

        for edge in edges_infected:
            try:
                index = list(G.edges()).index(edge)
                edge_colors[index] = 'orange'
                edge_widths[index] = 2
            except ValueError:
                continue

        nx.draw(G, pos, node_color=colors, edge_color=edge_colors, width=edge_widths, with_labels=False, node_size=10, ax=ax)
        ax.set_title(f'Time Step {i + 1}')

    plt.tight_layout()
    plt.savefig(f'infection_simulation_snapshots num_nodes={num_nodes} infection_rate={infection_prob} treatment_rate={recovery_prob} initial_infected={initial_infected}.pdf')
    plt.show()

def plot_prevalence_vs_recovery():
    """
    Plots how the final prevalence changes with the recovery rate from 0 to 1.
    """
    recovery_probs = np.linspace(0, 1, 100)
    final_prevalence = []
    for recovery_prob in recovery_probs:
        states = np.zeros(num_nodes, dtype=int)
        initial_infected_nodes = np.random.choice(num_nodes, initial_infected, replace=False)
        states[initial_infected_nodes] = 1
        _, _, prevalence = simulate_infection(G, states, infection_prob, recovery_prob, num_steps)
        final_prevalence.append(prevalence[-1])  # Append the last prevalence value
        print(f"Recovery Rate: {recovery_prob:.2f}", f"Final Prevalence: {prevalence[-1]:.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(recovery_probs, final_prevalence, label='Final Prevalence')
    plt.xlabel('Recovery Rate')
    plt.ylabel('Final Prevalence (Fraction of Infected Individuals)')
    plt.title('Final Prevalence vs. Recovery Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('prevalence_vs_recovery_rate.pdf')
    plt.show()



def plot_3d_prevalence(dynamic_param='recovery_rate'):
    """
    Creates a 3D plot where z is the prevalence, x is the dynamic parameter (infection rate or recovery rate),
    and y is the edge probability. Infection rate ranges between 0 and 0.2, and edge probability ranges between 0 and 0.1, 
    both with 100 slices.

    Parameters:
    - dynamic_param: The parameter to vary on the x-axis ('infection_rate' or 'recovery_rate')
    """
    # Parameters
    num_nodes = 1000
    initial_infected = 5
    num_steps = 30

    # Function for simulating infection spread
    def simulate_infection(G, states, infection_prob, recovery_prob, num_steps):
        """
        Simulates the infection spread in the network.
        
        Parameters:
        - G: NetworkX graph
        - states: Initial states of the nodes
        - infection_prob: Probability of infection spreading
        - recovery_prob: Probability of recovery
        - num_steps: Number of time steps for the simulation

        Returns:
        - final prevalence after num_steps
        """
        prevalence = []
        for _ in range(num_steps):
            new_states = states.copy()
            for node in range(len(states)):
                if states[node] == 1:  # Infected node
                    for neighbor in G.neighbors(node):
                        if states[neighbor] == 0 and np.random.rand() < infection_prob:
                            new_states[neighbor] = 1
                    if np.random.rand() < recovery_prob:
                        new_states[node] = 0  # Node recovers and becomes susceptible again
            states = new_states
            prevalence.append(np.mean(states))
        return prevalence[-1]  # Return final prevalence

    # Grid of dynamic parameter (infection rate or recovery rate) and edge probabilities
    dynamic_param_values = np.linspace(0, 0.2 if dynamic_param == 'infection_rate' else 1, 100)
    edge_probs = np.linspace(0, 0.1, 100)
    X, Y = np.meshgrid(dynamic_param_values, edge_probs)
    Z = np.zeros_like(X)

    # Calculate prevalence for each combination of dynamic parameter and edge probability
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dynamic_value = X[i, j]
            edge_prob = Y[i, j]
            G = nx.erdos_renyi_graph(num_nodes, edge_prob)
            states = np.zeros(num_nodes, dtype=int)
            initial_infected_nodes = np.random.choice(num_nodes, initial_infected, replace=False)
            states[initial_infected_nodes] = 1
            if dynamic_param == 'infection_rate':
                Z[i, j] = simulate_infection(G, states, dynamic_value, 0.05, num_steps)
            else:
                Z[i, j] = simulate_infection(G, states, 0.1, dynamic_value, num_steps)

    # Plotting the 3D surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Infection Rate' if dynamic_param == 'infection_rate' else 'Recovery Rate')
    ax.set_ylabel('Edge Probability')
    ax.set_zlabel('Prevalence')
    ax.set_title('Prevalence vs. ' + ('Infection Rate and Edge Probability' if dynamic_param == 'infection_rate' else 'Recovery Rate and Edge Probability'))
    filename = f'num_nodes={num_nodes} initial_infected={initial_infected} '
    filename += f'{"infection_rate" if dynamic_param == "recovery_rate" else "recovery_rate"}='
    filename += f'{recovery_prob if dynamic_param == "infection_rate" else infection_prob}.pdf'
    plt.savefig(filename)

    plt.show()



def main():
    # Uncomment the function you want to run
    # plot_prevalence()
    create_animation()
    # create_snapshots()
    # plot_prevalence_vs_recovery()
    # plot_3d_prevalence()
if __name__ == "__main__":
    main()
