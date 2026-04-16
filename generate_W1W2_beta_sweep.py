"""
Generate (W1, W2) regime diagrams for a few different beta values
to show how the bistable region expands with steepness.
"""

from bifurcation_W1W2_plane import generate_diagram

betas = [6, 10, 20]

for beta in betas:
    print(f"\n=== Generating diagram for beta = {beta} ===")
    generate_diagram(
        output_path=f'figures/bifurcation_W1W2_plane_beta{beta}.png',
        mu_value=0.4,
        beta_value=beta,
        W1_range=(-1.5, 2),
        W2_range=(-1.5, 2),
        n_points=300,
        n_guess=100,
    )

print("\nAll diagrams generated!")
