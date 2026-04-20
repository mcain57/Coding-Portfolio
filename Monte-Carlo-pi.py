import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_samples):
    # Generate random points in unit square
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    
    # Check if point falls inside quarter circle
    inside = (x**2 + y**2) <= 1
    
    pi_estimate = 4 * np.sum(inside) / n_samples
    return pi_estimate, x, y, inside

def plot_simulation(x, y, inside, pi_estimate):
    plt.figure(figsize=(6, 6))
    plt.scatter(x[inside], y[inside], color='steelblue', s=0.5, label='Inside')
    plt.scatter(x[~inside], y[~inside], color='salmon', s=0.5, label='Outside')
    plt.title(f'Monte Carlo π estimation: {pi_estimate:.5f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('monte_carlo_pi.png')
    plt.show()

if __name__ == "__main__":
    n = 100_000
    pi_est, x, y, inside = estimate_pi(n)
    print(f"Estimated π: {pi_est:.5f}")
    print(f"Actual π:    {np.pi:.5f}")
    print(f"Error:       {abs(pi_est - np.pi):.5f}")
    plot_simulation(x, y, inside, pi_est)