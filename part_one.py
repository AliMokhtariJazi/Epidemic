import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
tau = 0.05  # Treatment rate
mu = 1 / 50  # Natural death rate
N0 = 1110  # Initial total population
Pi = mu * N0  # Birthrate matches the initial number of deaths to keep constant initial population

# Initial conditions
S0 = 1000
I0 = 10
C0 = 100
R0 = 0
y0 = [S0, I0, C0, R0]

# Differential equations for the original model
def deriv(y, t, beta, tau, mu, Pi):
    S, I, C, R = y
    N = S + I + C + R  # Total population
    lambda_ = beta * (10 * I + C) / N
    dSdt = Pi + tau * (C + I) / 2 - (lambda_ + mu) * S
    dIdt = lambda_ * S - (tau + 1 + 2 * mu) * I
    dCdt = I - (tau + 2 * mu) * C
    dRdt = tau * (C + I) / 2 - mu * R
    return [dSdt, dIdt, dCdt, dRdt]

# Function for Question 1: Solve the system for β = 1
def solve_system_for_beta_1():
    beta = 1
    t = np.linspace(0, 2000, 100000) 
    solution = odeint(deriv, y0, t, args=(beta, tau, mu, Pi))
    S, I, C, R = solution.T
    N = S + I + C + R
    plt.figure(figsize=(10,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, C, label='Carrier')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, N, label='Total Population', linestyle='--')
    plt.xlabel('Time / years')
    plt.ylabel('Population')
    plt.legend()
    plt.xlim(0, 70)
    plt.ylim(0, 1200)
    plt.savefig('epidemic_model.pdf')
    plt.show()
    
    final_prevalence = (I[-1] + C[-1]) / N[-1]
    if final_prevalence > 0:
        print(f"The epidemic becomes endemic with a final prevalence of {final_prevalence*100:.2f}%.")
    else:
        print("The epidemic dies out.")

# Function to Find Beta for a Target Prevalence
def find_beta(target_prevalence):
    t = np.linspace(0, 2000, 100000) 
    betas = np.linspace(0.01, 2, 200)
    prevalence = []
    for beta in betas:
        solution = odeint(deriv, y0, t, args=(beta, tau, mu, Pi))
        S, I, C, R = solution.T
        N = S + I + C + R
        final_prevalence = (I[-1] + C[-1]) / N[-1]
        prevalence.append(final_prevalence)
    closest_beta_index = np.argmin(np.abs(np.array(prevalence) - target_prevalence))
    closest_beta = betas[closest_beta_index]
    closest_prevalence = prevalence[closest_beta_index]
    if np.abs(closest_prevalence - target_prevalence) > 0.01:  # Allow a small tolerance
        return None, prevalence  # Indicate no suitable beta found
    return closest_beta, prevalence

# Function for Question 2: Simulate a change in transmission rate
def simulate_change_in_transmission_rate():
    beta_35, prevalence_35 = find_beta(0.35)
    beta_50, prevalence_50 = find_beta(0.5)

    if beta_35 is not None:
        print(f"Beta for 35% prevalence: {beta_35}")
    else:
        print("No suitable beta found for 35% prevalence within the tested range.")

    if beta_50 is not None:
        print(f"Beta for 50% prevalence: {beta_50}")
    else:
        print("No suitable beta found for 50% prevalence within the tested range.")

    def critical_beta_for_zero_prevalence():
        t = np.linspace(0, 2000, 100000) 
        betas = np.linspace(0.0, 0.5, 200)

        for beta in betas:
            solution = odeint(deriv, y0, t, args=(beta, tau, mu, Pi))
            S, I, C, R = solution.T
            N = S + I + C + R
            final_prevalence = (I[-1] + C[-1]) / N[-1]
            if final_prevalence > 0.0001:
                break
        return beta
    critical_beta = critical_beta_for_zero_prevalence()
    print(f"The epidemic dies out with a beta of {critical_beta:.2f}.")
    plt.figure(figsize=(10,6))
    betas = np.linspace(0.01, 2, 200)
    plt.plot(betas, prevalence_35, label='35% Prevalence')
    plt.axhline(y=0.35, color='r', linestyle='--', label='Target Prevalence: 35%')
    plt.axhline(y=0.5, color='b', linestyle='--', label='Target Prevalence: 50%')
    plt.axvline(x=beta_35, color='gray', linestyle='--', label=f'Beta for 35% Prevalence = {beta_35:.2f}')
    plt.axvline(x=critical_beta, color='green', linestyle='--', label=f'Critical Beta for 0 Prevalence = {critical_beta:.2f}')
    plt.xlabel('Beta')
    plt.ylabel('Prevalence')
    plt.xlim(0, 2)
    plt.ylim(0, 0.6)
    plt.legend(loc='lower right')
    plt.savefig('prevalence_vs_beta.pdf')
    plt.show()

# Function for Question 3: Plot cumulative deaths and rate of new infections
def plot_cumulative_deaths_and_new_infections():
    beta_35, _ = find_beta(0.35)
    if beta_35 is None:
        print("No suitable beta found for 35% prevalence within the tested range.")
        return
    t = np.linspace(0, 2000, 100000) 
    solution = odeint(deriv, y0, t, args=(beta_35, tau, mu, Pi))
    S, I, C, R = solution.T
    N = S + I + C + R
    
    cumulative_deaths = N0 - N
    new_infections = np.diff(I)

    plt.figure(figsize=(10,6))
    plt.plot(t, cumulative_deaths, label='Cumulative Deaths')
    plt.xlabel('Time / years')
    plt.ylabel('Cumulative Deaths')
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 140)
    plt.savefig('cumulative_deaths.pdf')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(t[1:], new_infections, label='New Infections')
    plt.xlabel('Time / years')
    plt.ylabel('New Infections')
    plt.legend()
    plt.xlim(0, 10)
    plt.savefig('new_infections.pdf')
    plt.show()

# Function for Question 4: Use eigenvalues to determine endemic or disease-free scenario
def determine_endemic_or_disease_free():
    beta = 0.07
    # Jacobian matrix at DFE
    J = np.array([
        [-mu, tau/2 - 10 * beta, tau/2 - beta, 0],
        [0, 10 * beta - tau/2 - 2 * mu - 1, beta, 0],
        [0, 1, -tau - 2 * mu, 0],
        [0, tau/2, tau/2, -mu]
    ])


    eigenvalues = np.linalg.eigvals(J)
    print(f"Eigenvalues of the Jacobian at DFE: {eigenvalues}")

    if np.all(np.real(eigenvalues) < 0):
        print(f"The disease-free equilibrium for beta = {beta}, is stable (disease-free scenario).")
    else:
        print(f"The disease-free equilibrium for beta = {beta}, is unstable (endemic scenario).")

# Main function to call all the parts
def main():
    solve_system_for_beta_1()
    simulate_change_in_transmission_rate()
    plot_cumulative_deaths_and_new_infections()
    determine_endemic_or_disease_free()

if __name__ == "__main__":
    main()
