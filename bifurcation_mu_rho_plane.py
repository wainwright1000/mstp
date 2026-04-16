"""
Two-parameter bifurcation diagram in the (mu, rho) plane.

Colours indicate the number of equilibria (fixed points) of f_2(x)
for each combination of mu and rho, with beta, W1, W2 held fixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
from matplotlib.colors import ListedColormap

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_PARAMS = {
    'mu': 0.4,
    'beta': 14,
    'rho': 1,
    'W1': -0.6,
    'W2': 0.3,
}


def response(x, mu, beta, rho, W1, W2):
    mu2 = 1 - mu
    exp1 = np.exp(beta * (W1 + rho * (1 - 2 * x)))
    exp2 = np.exp(beta * (W2 + rho * (1 - 2 * x)))
    return mu / (1 + exp1) + mu2 / (1 + exp2)


def fixed_point_equation(x, mu, beta, rho, W1, W2):
    return response(x, mu, beta, rho, W1, W2) - x


def find_fixed_points(mu, beta, rho, W1, W2, n_guess=100):
    x_grid = np.linspace(0, 1, n_guess)
    g_values = fixed_point_equation(x_grid, mu, beta, rho, W1, W2)
    fixed_points = []
    for i in range(len(x_grid) - 1):
        if g_values[i] * g_values[i + 1] <= 0:
            try:
                root = brentq(fixed_point_equation, x_grid[i], x_grid[i + 1],
                              args=(mu, beta, rho, W1, W2))
                if not any(abs(root - fp) < 1e-6 for fp in fixed_points):
                    fixed_points.append(root)
            except ValueError:
                pass
    return sorted(fixed_points)


def generate_diagram(
    output_path='figures/bifurcation_mu_rho_plane.png',
    mu_range=(0, 1),
    rho_range=(0.001, 3),
    n_points=250,
    n_guess=100,
):
    params = DEFAULT_PARAMS.copy()
    mu_vals = np.linspace(mu_range[0], mu_range[1], n_points)
    rho_vals = np.linspace(rho_range[0], rho_range[1], n_points)
    n_fp = np.zeros((n_points, n_points), dtype=int)

    print(f"Scanning {n_points*n_points} grid points (mu x rho) ...")
    for i, mu in enumerate(mu_vals):
        for j, rho in enumerate(rho_vals):
            fps = find_fixed_points(
                mu=mu, beta=params['beta'], rho=rho,
                W1=params['W1'], W2=params['W2'], n_guess=n_guess,
            )
            n_fp[j, i] = len(fps)
        if (i + 1) % 25 == 0 or i == n_points - 1:
            print(f"  Progress: {i + 1}/{n_points} rows done")

    cmap = ListedColormap(['#6aa84f', '#f39c12', '#c0392b'])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)
    mesh = ax.pcolormesh(mu_vals, rho_vals, n_fp, shading='auto',
                         cmap=cmap, vmin=1, vmax=3)
    ax.contour(mu_vals, rho_vals, n_fp, levels=[2.0], colors='black',
               linewidths=1.5, linestyles='--')
    ax.plot(DEFAULT_PARAMS['mu'], DEFAULT_PARAMS['rho'], 'ko',
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='Baseline $(\\mu, \\rho)$', zorder=5)

    cbar = fig.colorbar(mesh, ax=ax, ticks=[1, 2, 3], shrink=0.8)
    cbar.set_label('Number of equilibria', fontsize=13)
    cbar.ax.set_yticklabels(['1 (monostable)', '2 (critical)', '3 (bistable)'])

    ax.set_xlabel(r'$\mu$', fontsize=13)
    ax.set_ylabel(r'$\rho$', fontsize=13)
    ax.set_title(
        r'Regime diagram in the $(\mu, \rho)$ plane' + '\n'
        r'$\beta={beta}, W_1={W1}, W_2={W2}$'.format(**params),
        fontsize=14,
    )
    ax.set_xlim(mu_range)
    ax.set_ylim(rho_range)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 3.01, 0.5))
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_aspect('auto', adjustable='box')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig, ax


if __name__ == '__main__':
    generate_diagram()
