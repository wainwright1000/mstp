"""
Bifurcation diagram generator for varying W₁ parameter.

Parameters: μ=0.54, β=14, ρ=1, W₂=0.3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from pathlib import Path
from matplotlib.lines import Line2D

# Set Arial font globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Default parameters
DEFAULT_PARAMS = {
    'mu': 0.54,
    'beta': 14,
    'rho': 1,
    'W1': -0.6,
    'W2': 0.3,
}


def response(x, mu, beta, rho, W1, W2):
    """Response function f(x)."""
    mu2 = 1 - mu
    exp1 = np.exp(beta * (W1 + rho * (1 - 2*x)))
    exp2 = np.exp(beta * (W2 + rho * (1 - 2*x)))
    return mu / (1 + exp1) + mu2 / (1 + exp2)


def fixed_point_equation(x, mu, beta, rho, W1, W2):
    """g(x) = f(x) - x."""
    return response(x, mu, beta, rho, W1, W2) - x


def response_derivative(x, mu, beta, rho, W1, W2):
    """Derivative f'(x) for stability analysis."""
    mu2 = 1 - mu
    exp1 = np.exp(beta * (W1 + rho * (1 - 2*x)))
    exp2 = np.exp(beta * (W2 + rho * (1 - 2*x)))
    d1 = 2 * beta * rho * mu * exp1 / (1 + exp1)**2
    d2 = 2 * beta * rho * mu2 * exp2 / (1 + exp2)**2
    return d1 + d2


def calculate_B(x, mu, beta, rho, W1, W2):
    """
    Calculate B(x) = mu*Y1 + (1-mu)*Y2 where Yj = Zj/(1+Zj)^2
    and Zj = exp(beta*(Wj + rho*(1-2x))).
    """
    mu2 = 1 - mu
    Z1 = np.exp(beta * (W1 + rho * (1 - 2*x)))
    Z2 = np.exp(beta * (W2 + rho * (1 - 2*x)))
    Y1 = Z1 / (1 + Z1)**2
    Y2 = Z2 / (1 + Z2)**2
    return mu * Y1 + mu2 * Y2


def find_fixed_points(mu, beta, rho, W1, W2, n_guess=200):
    """Find all fixed points in [0, 1] by scanning for sign changes."""
    x_grid = np.linspace(0, 1, n_guess)
    g_values = fixed_point_equation(x_grid, mu, beta, rho, W1, W2)
    
    fixed_points = []
    for i in range(len(x_grid) - 1):
        if g_values[i] * g_values[i+1] <= 0:
            try:
                root = brentq(fixed_point_equation, x_grid[i], x_grid[i+1],
                              args=(mu, beta, rho, W1, W2))
                fp_deriv = response_derivative(root, mu, beta, rho, W1, W2)
                is_stable = abs(fp_deriv) < 1
                if not any(abs(root - fp[0]) < 1e-6 for fp in fixed_points):
                    fixed_points.append((root, is_stable))
            except ValueError:
                pass
    return sorted(fixed_points, key=lambda x: x[0])


def generate_diagram(output_path='figures/bifurcation_W1.png', mu_value=0.54):
    """Generate bifurcation diagram for varying W₁."""
    params = DEFAULT_PARAMS.copy()
    params['mu'] = mu_value
    param_range = (-1.5, 1.5)
    n_points = 1000
    n_guess = 200
    
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    stable_x, stable_y = [], []
    unstable_x, unstable_y = [], []
    
    for p_val in param_values:
        p = params.copy()
        p['W1'] = p_val
        fps = find_fixed_points(**p, n_guess=n_guess)
        for fp, is_stable in fps:
            if is_stable:
                stable_x.append(p_val)
                stable_y.append(fp)
            else:
                unstable_x.append(p_val)
                unstable_y.append(fp)
    
    # Find class II STPs via B(x*) = 1/(2*beta*rho)
    all_W1_vals = stable_x + unstable_x
    all_x_vals = stable_y + unstable_y
    
    threshold_points = []
    for i, (W1_val, x_val) in enumerate(zip(all_W1_vals, all_x_vals)):
        B_val = calculate_B(x_val, params['mu'], params['beta'], params['rho'],
                           W1_val, params['W2'])
        B_crit = 1 / (2 * params['beta'] * params['rho'])
        diff = B_val - B_crit
        if abs(diff) < 0.1:
            threshold_points.append((x_val, W1_val, diff))
    
    threshold_points = sorted(set(threshold_points))
    
    intersection_ys = []
    for i in range(len(threshold_points) - 1):
        y1, W1_1, diff1 = threshold_points[i]
        y2, W1_2, diff2 = threshold_points[i + 1]
        if abs(y2 - y1) > 0.2 or abs(W1_2 - W1_1) > 0.1:
            continue
        if diff1 * diff2 < 0:
            t = abs(diff1) / (abs(diff1) + abs(diff2))
            y_interp = y1 + t * (y2 - y1)
            intersection_ys.append(y_interp)
    
    turning_x = []
    turning_y = []
    W1_critical = 1 / (2 * params['beta'] * params['rho']) * 3.5 - 1.5
    for y_val in intersection_ys:
        def g_at_y(W1):
            return abs(fixed_point_equation(y_val, params['mu'], params['beta'],
                                            params['rho'], W1, params['W2']))
        result = minimize_scalar(g_at_y, bounds=(W1_critical, 1.5), method='bounded')
        if result.fun < 0.01:
            turning_x.append(result.x)
            turning_y.append(y_val)
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    
    if unstable_x:
        ax.scatter(unstable_x, unstable_y, c='red', s=4, alpha=0.7,
                   zorder=1, edgecolors='none')
    if stable_x:
        ax.scatter(stable_x, stable_y, c='#6aa84f', s=4, alpha=0.7,
                   zorder=2, edgecolors='none')
    if turning_x:
        ax.scatter(turning_x, turning_y, c='black', s=100, marker='x',
                   linewidths=1.5, zorder=3)
    
    ax.set_xlabel(r'$W_1$', fontsize=13)
    ax.set_ylabel('Equilibria', fontsize=13)
    ax.set_title(r'Bifurcation diagram of $f_2(x)$ varying $W_1$', fontsize=16)
    ax.set_xlim(param_range)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.set_xticks(np.arange(-1.5, 1.51, 0.5))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='class I social tipping points'),
        Line2D([0], [0], color='#6aa84f', lw=2, label='stable equilibria'),
        Line2D([0], [0], color='black', marker='x', markersize=10, markeredgewidth=1.5,
               linestyle='None', label='class II social tipping points')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    
    return fig, ax


if __name__ == '__main__':
    generate_diagram()
