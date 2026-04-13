"""
Bifurcation diagram generator for varying ρ parameter.

Parameters: μ=0.54, β=14, W₁=-0.6, W₂=0.3
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


def generate_diagram(output_path='figures/bifurcation_rho_with_B.png', mu_value=0.4):
    """Generate bifurcation diagram for varying ρ."""
    params = DEFAULT_PARAMS.copy()
    params['mu'] = mu_value
    param_range = (0.001, 3)
    n_points = 1000
    n_guess = 200
    
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    stable_x, stable_y = [], []
    unstable_x, unstable_y = [], []
    
    for p_val in param_values:
        p = params.copy()
        p['rho'] = p_val
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
    
    # Purple overlay: B(x*) scaled to rho axis
    B_x_vals = []
    B_y_vals = []
    for rho_val, x_val in zip(stable_x + unstable_x, stable_y + unstable_y):
        B_val = calculate_B(x_val, params['mu'], params['beta'], rho_val,
                           params['W1'], params['W2'])
        B_x_vals.append(B_val * 3)
        B_y_vals.append(x_val)
    
    if B_x_vals:
        ax.scatter(B_x_vals, B_y_vals, c='purple', s=3, alpha=0.6,
                   zorder=4, edgecolors='none', label=r'$B(x^*)$')
    
    # Find intersections where B(x*) = 1/(2*beta*rho)
    all_rho_vals = stable_x + unstable_x
    all_x_vals = stable_y + unstable_y
    
    nearby_threshold = 0.1  # In B units
    threshold_points = []
    for i, (rho_val, x_val) in enumerate(zip(all_rho_vals, all_x_vals)):
        B_val = B_x_vals[i] / 3
        if rho_val <= 0.01:
            continue
        B_crit = 1 / (2 * params['beta'] * rho_val)
        diff = B_val - B_crit
        if abs(diff) < nearby_threshold:
            threshold_points.append((x_val, B_val * 3, rho_val, diff))
    
    threshold_points = sorted(set(threshold_points))
    
    intersection_candidates = []
    for i in range(len(threshold_points) - 1):
        y1, Bs1, rho1, diff1 = threshold_points[i]
        y2, Bs2, rho2, diff2 = threshold_points[i + 1]
        
        # Skip large gaps (different branches)
        if abs(y2 - y1) > 0.2 or abs(rho2 - rho1) > 0.3:
            continue
        
        if diff1 * diff2 < 0:  # Sign change = intersection
            t = abs(diff1) / (abs(diff1) + abs(diff2))
            y_interp = y1 + t * (y2 - y1)
            rho_interp = rho1 + t * (rho2 - rho1)
            intersection_candidates.append((y_interp, rho_interp))
    
    # For each intersection y, solve for exact rho and draw markers
    new_turning_x = []
    new_turning_y = []
    
    for y_val, rho_guess in intersection_candidates:
        def intersection_condition(rho):
            if rho < 0.001:
                return float('inf')
            g_val = fixed_point_equation(y_val, params['mu'], params['beta'],
                                         rho, params['W1'], params['W2'])
            B_val = calculate_B(y_val, params['mu'], params['beta'], rho,
                               params['W1'], params['W2'])
            B_crit = 1 / (2 * params['beta'] * rho)
            return g_val**2 + (B_val - B_crit)**2
        
        lo = max(0.001, rho_guess - 0.2)
        hi = min(3.0, rho_guess + 0.2)
        result = minimize_scalar(intersection_condition, bounds=(lo, hi), method='bounded')
        
        if result.fun < 0.001:
            rho_at_int = result.x
            B_at_int = calculate_B(y_val, params['mu'], params['beta'], rho_at_int,
                                   params['W1'], params['W2'])
            B_scaled = B_at_int * 3
            
            new_turning_x.append(rho_at_int)
            new_turning_y.append(y_val)
            
            # Horizontal dashed line from circle (on purple) to X mark (native)
            ax.plot([B_scaled, rho_at_int], [y_val, y_val],
                   color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
            # Circle at purple-curve position
            ax.plot(B_scaled, y_val, 'ko', markersize=7, markerfacecolor='none',
                    markeredgewidth=1.5, zorder=5)
    
    if new_turning_x:
        ax.scatter(new_turning_x, new_turning_y, c='black', s=100, marker='x',
                   linewidths=1.5, zorder=3)
    
    ax.set_xlabel(r'$\rho$', fontsize=13)
    ax.set_ylabel('Equilibria', fontsize=13)
    ax.set_title(r'Bifurcation diagram of $f_2(x)$ varying $\rho$', fontsize=16)
    ax.set_xlim(param_range)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.set_xticks(np.arange(0, 3.01, 0.5))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='class I social tipping points'),
        Line2D([0], [0], color='#6aa84f', lw=2, label='stable equilibria'),
        Line2D([0], [0], color='black', marker='x', markersize=10, markeredgewidth=1.5,
               linestyle='None', label='class II social tipping points'),
        Line2D([0], [0], color='purple', lw=1, label=r'$B(x^*)$'),
        Line2D([0], [0], color='grey', lw=1, linestyle='--',
               label=r'$1/(2\beta\rho)$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, framealpha=0.9)
    
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
