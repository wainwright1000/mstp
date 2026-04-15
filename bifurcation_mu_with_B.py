"""
Bifurcation diagram generator for varying μ parameter.

Parameters: β=14, ρ=1, W₁=-0.6, W₂=0.3
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
    'mu': 0.4,
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
    This appears in the gradient formula: df/dx = 2*beta*rho*B(x).
    """
    mu2 = 1 - mu
    Z1 = np.exp(beta * (W1 + rho * (1 - 2*x)))
    Z2 = np.exp(beta * (W2 + rho * (1 - 2*x)))
    Y1 = Z1 / (1 + Z1)**2
    Y2 = Z2 / (1 + Z2)**2
    return mu * Y1 + mu2 * Y2


def scale_b_to_axis(B_val, param_range):
    """Map raw B(x*) to the native horizontal axis so purple overlays align visually across the four-panel figure."""
    scale = param_range[1] - param_range[0]
    offset = param_range[0]
    return B_val * scale + offset


def find_fixed_points(mu, beta, rho, W1, W2, n_guess=200):
    """Find all fixed points in [0, 1] by scanning for sign changes."""
    x_grid = np.linspace(0, 1, n_guess)
    g_values = fixed_point_equation(x_grid, mu, beta, rho, W1, W2)
    
    fixed_points = []
    for i in range(len(x_grid) - 1):
        # Detect sign change OR boundary root (where g(x) = 0 at endpoint)
        # Using <= catches roots at grid boundaries like x=1 where g(1)=0
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


def find_turning_points(param_values, all_fixed_points, tolerance=0.08):
    """
    Find turning points by tracking branches across parameter values.
    A turning point (class II STP) occurs when a fixed point has no
    continuation at the next parameter value - i.e., the branch ends.
    
    Parameters
    ----------
    param_values : array
        Array of parameter values
    all_fixed_points : list of lists
        List of fixed points at each parameter value
    tolerance : float
        Maximum distance in x to consider two points as the same branch
    
    Returns
    -------
    list of tuples
        (param_value, fixed_point) for each turning point
    """
    turning_points = []
    
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
                # This branch ended here
                turning_points.append((p_curr, fp_curr))
        
        # Check which branches start at this step (absent at i, present at i+1)
        for fp_next, _ in fps_next:
            has_origin = False
            for fp_curr, _ in fps_curr:
                if abs(fp_curr - fp_next) < tolerance:
                    has_origin = True
                    break
            if not has_origin:
                # This branch started here
                turning_points.append((p_next, fp_next))
    
    # Remove duplicates (within small parameter tolerance)
    unique_turning = []
    for tp in turning_points:
        if not any(abs(tp[0] - utp[0]) < 0.05 and abs(tp[1] - utp[1]) < 0.05 
                   for utp in unique_turning):
            unique_turning.append(tp)
    
    return unique_turning


def generate_diagram(output_path='figures/bifurcation_mu_with_B.png'):
    """Generate bifurcation diagram for varying μ."""
    params = DEFAULT_PARAMS.copy()
    param_range = (0, 1)
    n_points = 1000
    n_guess = 200
    
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    stable_x, stable_y = [], []
    unstable_x, unstable_y = [], []
    all_fixed_points = []
    
    for p_val in param_values:
        p = params.copy()
        p['mu'] = p_val
        fps = find_fixed_points(**p, n_guess=n_guess)
        all_fixed_points.append(fps)
        for fp, is_stable in fps:
            if is_stable:
                stable_x.append(p_val)
                stable_y.append(fp)
            else:
                unstable_x.append(p_val)
                unstable_y.append(fp)
    
    # Find turning points
    turning_points = find_turning_points(param_values, all_fixed_points)
    turning_x = [tp[0] for tp in turning_points]
    turning_y = [tp[1] for tp in turning_points]
    
    # Split stable points by branch using turning point thresholds
    if turning_points:
        lower_knee = min(turning_y)
        upper_knee = max(turning_y)
        
        bottom_stable_x = [sx for sx, sy in zip(stable_x, stable_y) if sy < lower_knee]
        bottom_stable_y = [sy for sy in stable_y if sy < lower_knee]
        top_stable_x = [sx for sx, sy in zip(stable_x, stable_y) if sy > upper_knee]
        top_stable_y = [sy for sy in stable_y if sy > upper_knee]
        middle_stable_x = [sx for sx, sy in zip(stable_x, stable_y) 
                          if lower_knee <= sy <= upper_knee]
        middle_stable_y = [sy for sy in stable_y if lower_knee <= sy <= upper_knee]
    else:
        bottom_stable_x, bottom_stable_y = stable_x, stable_y
        top_stable_x, top_stable_y = [], []
        middle_stable_x, middle_stable_y = [], []
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    
    # Plot curves
    if unstable_x:
        ax.scatter(unstable_x, unstable_y, c='red', s=4, alpha=0.7, 
                   zorder=1, edgecolors='none')
    
    # Bottom branch - thinner
    if bottom_stable_x:
        ax.scatter(bottom_stable_x, bottom_stable_y, c='#6aa84f', s=4, alpha=0.7,
                   zorder=2, edgecolors='none')
    
    # Top branch - thicker
    if top_stable_x:
        ax.scatter(top_stable_x, top_stable_y, c='#6aa84f', s=12, alpha=0.7,
                   zorder=2, edgecolors='none')
    
    # Middle segment - thin (if any stable points there)
    if middle_stable_x:
        ax.scatter(middle_stable_x, middle_stable_y, c='#6aa84f', s=4, alpha=0.7,
                   zorder=2, edgecolors='none')
    
    # Vertical line at x = 1/(2*beta*rho) (calculated value, not mu-related)
    mu_critical = 1 / (2 * params['beta'] * params['rho'])
    ax.axvline(x=mu_critical, color='grey', linestyle='--', linewidth=1, 
               alpha=0.7, zorder=0)
    
    # Calculate and plot B(x*) at equilibrium points
    # For each equilibrium (mu, x*), calculate B(x*) using that specific mu value
    # This shows B evaluated at the actual equilibrium points
    B_x_vals = []
    B_y_vals = []
    for mu_val, x_val in zip(stable_x + unstable_x, stable_y + unstable_y):
        B_val = calculate_B(x_val, mu_val, params['beta'], params['rho'], 
                           params['W1'], params['W2'])
        B_x_vals.append(scale_b_to_axis(B_val, param_range))  # x-coordinate is B(x*) mapped to axis
        B_y_vals.append(x_val)  # y-coordinate is x*
    
    if B_x_vals:
        ax.scatter(B_x_vals, B_y_vals, c='purple', s=3, alpha=0.6, 
                   zorder=4, edgecolors='none', label=r'$B(x^*)$')
    
    # Find where the purple B(x*) curve intersects the vertical line at x = 1/(2*beta*rho)
    # The purple curve is plotted at (B_value, x*) where B_value = calculate_B(x*, mu_actual, ...)
    # The vertical line is at x = target_B ≈ 0.036
    target_B = mu_critical  # 1/(2*beta*rho) ≈ 0.036
    
    # Linear interpolation: find pairs where B crosses target_B
    # Only consider points reasonably close to the vertical line
    nearby_threshold = 0.1  # Only consider B values within 0.1 of target
    
    purple_points = [(y, B) for y, B in zip(B_y_vals, B_x_vals) 
                     if abs(B - target_B) < nearby_threshold]
    
    # Sort by y and remove duplicates
    purple_points = sorted(set(purple_points))
    
    intersection_ys = []
    for i in range(len(purple_points) - 1):
        y1, B1 = purple_points[i]
        y2, B2 = purple_points[i + 1]
        
        # Only interpolate if y values are close (same branch)
        if abs(y2 - y1) > 0.2:  # Skip if gap is too large (different branches)
            continue
        
        # Check if target_B is between B1 and B2 (sign change)
        if (B1 - target_B) * (B2 - target_B) < 0:
            # Linear interpolation
            t = (target_B - B1) / (B2 - B1)
            y_interp = y1 + t * (y2 - y1)
            intersection_ys.append(y_interp)
    
    # For each intersection y, find equilibrium at that y and draw horizontal line
    new_turning_x = []
    new_turning_y = []
    
    for y_val in intersection_ys:
        # Find where g(x*) = 0 at this y by searching horizontally
        
        def g_at_y(mu):
            return abs(fixed_point_equation(y_val, mu, params['beta'], params['rho'],
                                            params['W1'], params['W2']))
        
        result = minimize_scalar(g_at_y, bounds=(mu_critical, 1.0), method='bounded')
        best_mu = result.x
        best_g = result.fun
        
        if best_g < 0.01:
            new_turning_x.append(best_mu)
            new_turning_y.append(y_val)
            ax.plot([mu_critical, best_mu], [y_val, y_val],
                   color='grey', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
            # Mark the intersection point with larger hollow black circle
            ax.plot(mu_critical, y_val, 'ko', markersize=7, markerfacecolor='none',
                    markeredgewidth=1.5, zorder=5)
    
    # Update turning points to the calculated ones
    turning_x = new_turning_x
    turning_y = new_turning_y
    
    # Plot the turning points (class II STPs) with x marks
    if turning_x:
        ax.scatter(turning_x, turning_y, c='black', s=100, marker='x', 
                   linewidths=1.5, zorder=3)
    
    # Labels and title (reduced font sizes)
    ax.set_xlabel(r'$\mu$', fontsize=13)
    ax.set_ylabel('Equilibria', fontsize=13)
    ax.set_title(r'Bifurcation diagram of $f_2(x)$ varying $\mu$', fontsize=16)
    ax.set_xlim(param_range)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=12)
    
    # Grid with 0.25 step
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    
    # Legend with lines
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
