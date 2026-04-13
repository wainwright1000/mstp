"""
Bifurcation diagram generator for varying ρ parameter.

Parameters: μ=0.54, β=14, W₁=-0.6, W₂=0.3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
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


def find_turning_points(param_values, all_fixed_points, tolerance=0.03):
    """
    Find turning points by tracking branches across parameter values.
    A turning point (class II STP) occurs when a fixed point has no
    continuation at the next parameter value - i.e., the branch ends.
    
    Also detects boundary starts/ends (branches appearing/disappearing
    at the edge of the parameter range).
    """
    turning_points = []
    
    # Check boundary start (first sample)
    if len(all_fixed_points) > 0:
        fps_first = all_fixed_points[0]
        fps_second = all_fixed_points[1] if len(all_fixed_points) > 1 else []
        for fp_first, _ in fps_first:
            has_continuation = any(abs(fp_first - fp) < tolerance for fp, _ in fps_second)
            if not has_continuation:
                # This branch started at the left boundary
                turning_points.append((param_values[0], fp_first))
    
    for i in range(len(param_values) - 1):
        p_curr = param_values[i]
        p_next = param_values[i + 1]
        fps_curr = all_fixed_points[i]
        fps_next = all_fixed_points[i + 1]
        
        # Check which branches end at this step (present at i, absent at i+1)
        for fp_curr, _ in fps_curr:
            has_continuation = False
            for fp_next, _ in fps_next:
                if abs(fp_curr - fp_next) < tolerance:
                    has_continuation = True
                    break
            if not has_continuation:
                turning_points.append((p_curr, fp_curr))
        
        # Check which branches start at this step (absent at i, present at i+1)
        for fp_next, _ in fps_next:
            has_origin = False
            for fp_curr, _ in fps_curr:
                if abs(fp_curr - fp_next) < tolerance:
                    has_origin = True
                    break
            if not has_origin:
                turning_points.append((p_next, fp_next))
    
    # Remove duplicates with wider tolerance
    unique_turning = []
    for tp in turning_points:
        if not any(abs(tp[0] - utp[0]) < 0.08 and abs(tp[1] - utp[1]) < 0.08 
                   for utp in unique_turning):
            unique_turning.append(tp)
    
    return unique_turning


def generate_diagram(output_path='figures/bifurcation_rho.png', mu_value=0.54):
    """Generate bifurcation diagram for varying ρ."""
    params = DEFAULT_PARAMS.copy()
    params['mu'] = mu_value
    param_range = (0.001, 3)
    n_points = 1000
    n_guess = 200
    
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    stable_x, stable_y = [], []
    unstable_x, unstable_y = [], []
    all_fixed_points = []
    
    for p_val in param_values:
        p = params.copy()
        p['rho'] = p_val
        fps = find_fixed_points(**p, n_guess=n_guess)
        all_fixed_points.append(fps)
        for fp, is_stable in fps:
            if is_stable:
                stable_x.append(p_val)
                stable_y.append(fp)
            else:
                unstable_x.append(p_val)
                unstable_y.append(fp)
    
    turning_points = find_turning_points(param_values, all_fixed_points)
    turning_x = [tp[0] for tp in turning_points]
    turning_y = [tp[1] for tp in turning_points]
    
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
    
    ax.set_xlabel(r'$\rho$', fontsize=13)
    ax.set_ylabel('Equilibria', fontsize=13)
    title_mu = params['mu']
    ax.set_title(r'Equilibrium points for $f_2(\mu=' + f'{title_mu}' + r', x, \beta=14, W_1=-0.6, W_2=0.3, \rho)$', 
                 fontsize=14)
    ax.set_xlim(param_range)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.set_xticks(np.arange(0, 3.01, 0.5))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='class I social tipping points'),
        Line2D([0], [0], color='#6aa84f', lw=2, label='stable equilibria'),
        Line2D([0], [0], color='black', marker='x', markersize=10, markeredgewidth=1.5,
               linestyle='None', label='class II social tipping points')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
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
