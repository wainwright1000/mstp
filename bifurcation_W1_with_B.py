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


def generate_diagram(output_path='figures/bifurcation_W1_with_B.png', mu_value=0.4):
    """Generate bifurcation diagram for varying W₁."""
    params = DEFAULT_PARAMS.copy()
    params['mu'] = mu_value
    param_range = (-1.5, 2)
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
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    
    if unstable_x:
        ax.scatter(unstable_x, unstable_y, c='red', s=4, alpha=0.7, 
                   zorder=1, edgecolors='none')
    if stable_x:
        ax.scatter(stable_x, stable_y, c='#6aa84f', s=4, alpha=0.7,
                   zorder=2, edgecolors='none')
    
    # Vertical line at normalized x = 1/(2*beta*rho)
    normalized_critical = 1 / (2 * params['beta'] * params['rho'])
    mu_critical = normalized_critical * 3.5 - 1.5  # Scale to W1 range
    ax.axvline(x=mu_critical, color='grey', linestyle='--', linewidth=1, 
               alpha=0.7, zorder=0)
    
    # Calculate and plot B(x*) at equilibrium points
    B_x_vals = []
    B_y_vals = []
    for W1_val, x_val in zip(stable_x + unstable_x, stable_y + unstable_y):
        B_val = calculate_B(x_val, params['mu'], params['beta'], params['rho'], 
                           W1_val, params['W2'])
        B_x_vals.append(B_val * 3.5 - 1.5)  # Scale to W1 range
        B_y_vals.append(x_val)
    
    if B_x_vals:
        ax.scatter(B_x_vals, B_y_vals, c='purple', s=3, alpha=0.6, 
                   zorder=4, edgecolors='none', label=r'$B(x^*)$')
    
    # Find where purple curve intersects vertical line
    target_B = mu_critical
    nearby_threshold = 0.1 * 3.5
    
    purple_points = [(y, B) for y, B in zip(B_y_vals, B_x_vals) 
                     if abs(B - target_B) < nearby_threshold]
    purple_points = sorted(set(purple_points))
    
    intersection_ys = []
    for i in range(len(purple_points) - 1):
        y1, B1 = purple_points[i]
        y2, B2 = purple_points[i + 1]
        
        if abs(y2 - y1) > 0.2:
            continue
        
        if (B1 - target_B) * (B2 - target_B) < 0:
            t = (target_B - B1) / (B2 - B1)
            y_interp = y1 + t * (y2 - y1)
            intersection_ys.append(y_interp)
    
    # Find turning points and draw horizontal lines
    new_turning_x = []
    new_turning_y = []
    
    for y_val in intersection_ys:
        def g_at_y(W1):
            return abs(fixed_point_equation(y_val, params['mu'], params['beta'], 
                                            params['rho'], W1, params['W2']))
        
        result = minimize_scalar(g_at_y, bounds=(mu_critical, 2.0), method='bounded')
        best_W1 = result.x
        best_g = result.fun
        
        if best_g < 0.01:
            new_turning_x.append(best_W1)
            new_turning_y.append(y_val)
            ax.plot([mu_critical, best_W1], [y_val, y_val],
                   color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
            ax.plot(mu_critical, y_val, 'ko', markersize=7, markerfacecolor='none',
                    markeredgewidth=1.5, zorder=5)
    
    # Plot turning points with x marks
    if new_turning_x:
        ax.scatter(new_turning_x, new_turning_y, c='black', s=100, marker='x', 
                   linewidths=1.5, zorder=3)
    
    ax.set_xlabel(r'$W_1$', fontsize=13)
    ax.set_ylabel('Equilibria', fontsize=13)
    ax.set_title(r'Bifurcation diagram of $f_2(x)$ varying $W_1$', fontsize=16)
    ax.set_xlim(param_range)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.set_xticks(np.arange(-1.5, 2.01, 0.5))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='class I social tipping points'),
        Line2D([0], [0], color='#6aa84f', lw=2, label='stable equilibria'),
        Line2D([0], [0], color='black', marker='x', markersize=10, markeredgewidth=1.5,
               linestyle='None', label='class II social tipping points'),
        Line2D([0], [0], color='purple', lw=1, label=r'$B(x^*)$'),
        Line2D([0], [0], color='grey', lw=1, linestyle='--', marker='o',
               markersize=7, markerfacecolor='none', markeredgecolor='black',
               markeredgewidth=1.5, label=r'$1/(2\beta\rho)$')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=13, framealpha=0.9)
    
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
