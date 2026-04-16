"""
Two-parameter bifurcation diagram in the (W₁, W₂) plane.

Colours indicate the number of equilibria (fixed points) of f₂(x)
for each combination of W₁ and W₂, with μ, β, ρ held fixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
from matplotlib.colors import ListedColormap

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
    exp1 = np.exp(beta * (W1 + rho * (1 - 2 * x)))
    exp2 = np.exp(beta * (W2 + rho * (1 - 2 * x)))
    return mu / (1 + exp1) + mu2 / (1 + exp2)


def fixed_point_equation(x, mu, beta, rho, W1, W2):
    """g(x) = f(x) - x."""
    return response(x, mu, beta, rho, W1, W2) - x


def response_derivative(x, mu, beta, rho, W1, W2):
    """Derivative f'(x) for stability analysis."""
    mu2 = 1 - mu
    exp1 = np.exp(beta * (W1 + rho * (1 - 2 * x)))
    exp2 = np.exp(beta * (W2 + rho * (1 - 2 * x)))
    d1 = 2 * beta * rho * mu * exp1 / (1 + exp1) ** 2
    d2 = 2 * beta * rho * mu2 * exp2 / (1 + exp2) ** 2
    return d1 + d2


def find_fixed_points(mu, beta, rho, W1, W2, n_guess=100):
    """Find all fixed points in [0, 1] by scanning for sign changes."""
    x_grid = np.linspace(0, 1, n_guess)
    g_values = fixed_point_equation(x_grid, mu, beta, rho, W1, W2)

    fixed_points = []
    for i in range(len(x_grid) - 1):
        if g_values[i] * g_values[i + 1] <= 0:
            try:
                root = brentq(
                    fixed_point_equation,
                    x_grid[i],
                    x_grid[i + 1],
                    args=(mu, beta, rho, W1, W2),
                )
                fp_deriv = response_derivative(root, mu, beta, rho, W1, W2)
                is_stable = abs(fp_deriv) < 1
                if not any(abs(root - fp[0]) < 1e-6 for fp in fixed_points):
                    fixed_points.append((root, is_stable))
            except ValueError:
                pass
    return sorted(fixed_points, key=lambda x: x[0])


def generate_diagram(
    output_path='figures/bifurcation_W1W2_plane.png',
    mu_value=0.4,
    beta_value=None,
    W1_range=(-1.5, 2),
    W2_range=(-1.5, 2),
    n_points=300,
    n_guess=100,
):
    """Generate a 2-D bifurcation diagram in the (W₁, W₂) plane."""
    params = DEFAULT_PARAMS.copy()
    params['mu'] = mu_value
    if beta_value is not None:
        params['beta'] = beta_value

    W1_vals = np.linspace(W1_range[0], W1_range[1], n_points)
    W2_vals = np.linspace(W2_range[0], W2_range[1], n_points)

    # Store the number of fixed points at each grid node
    n_fp = np.zeros((n_points, n_points), dtype=int)

    total = n_points * n_points
    print(f"Scanning {total} grid points (W1 x W2) ...")

    for i, W1 in enumerate(W1_vals):
        for j, W2 in enumerate(W2_vals):
            fps = find_fixed_points(
                mu=params['mu'],
                beta=params['beta'],
                rho=params['rho'],
                W1=W1,
                W2=W2,
                n_guess=n_guess,
            )
            n_fp[j, i] = len(fps)

        if (i + 1) % 30 == 0 or i == n_points - 1:
            print(f"  Progress: {i + 1}/{n_points} rows done")

    # Discrete colormap: 1 = monostable, 2 = critical, 3 = bistable
    cmap = ListedColormap(['#6aa84f', '#f39c12', '#c0392b'])

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=100)

    # pcolormesh with discrete levels
    mesh = ax.pcolormesh(
        W1_vals,
        W2_vals,
        n_fp,
        shading='auto',
        cmap=cmap,
        vmin=1,
        vmax=3,
    )

    # Overlay contour line separating 1- and 3-equilibrium regions
    ax.contour(
        W1_vals,
        W2_vals,
        n_fp,
        levels=[2.0],
        colors='black',
        linewidths=1.5,
        linestyles='--',
    )

    # Mark the default parameter point
    ax.plot(
        DEFAULT_PARAMS['W1'],
        DEFAULT_PARAMS['W2'],
        'ko',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        label='Baseline $(W_1, W_2)$',
        zorder=5,
    )

    cbar = fig.colorbar(mesh, ax=ax, ticks=[1, 2, 3], shrink=0.8)
    cbar.set_label('Number of equilibria', fontsize=13)
    cbar.ax.set_yticklabels(['1 (monostable)', '2 (critical)', '3 (bistable)'])

    ax.set_xlabel(r'$W_1$', fontsize=13)
    ax.set_ylabel(r'$W_2$', fontsize=13)
    ax.set_title(
        r'Regime diagram in the $(W_1, W_2)$ plane' + '\n'
        r'$\mu={mu}, \beta={beta}, \rho={rho}$'.format(**params),
        fontsize=14,
    )
    ax.set_xlim(W1_range)
    ax.set_ylim(W2_range)
    ax.set_xticks(np.arange(-1.5, 2.01, 0.5))
    ax.set_yticks(np.arange(-1.5, 2.01, 0.5))
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
        )
        print(f"Saved: {output_path}")

    return fig, ax


if __name__ == '__main__':
    generate_diagram()
