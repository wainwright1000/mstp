# Bifurcation Diagram Logic

## Agent Guardrails — Do Not Change Without Explicit Confirmation

- **Maintain both script suites.** The `*_with_B.py` and `*.py` files are intentionally kept in parallel. Do not merge them or delete one suite.
- **No vertical reference line for β or ρ.** The threshold `1/(2βρ)` varies with the x-axis parameter in these plots. Adding a vertical line or a native-axis curve here is a known visual error.
- **Do not alter circle/X-mark coordinate logic without testing all four parameters.** The placement rules differ between constant-threshold parameters (μ, W₁) and varying-threshold parameters (β, ρ).
- **Keep Arial font and the current colour scheme.** Green = stable, red = unstable (class I STPs), purple = B(x*), black circle = B-curve/threshold intersection, black X = exact class II STP, grey dashed = horizontal connector (all params) or vertical threshold line (μ, W₁ only).
- **Top equilibrium line in the μ diagram must stay thicker.** It lies so close to x = 1 that it is nearly invisible at standard line weight, and unlike other parameters the segment does not appear elsewhere on the plot.
- **Use `scale_b_to_axis` for all purple overlays.** All eight scripts share the same helper: `B_val * range_width + range_min`. This keeps the four-panel figure visually consistent.
- **Do not simplify the branch-gap logic.** The `abs(y2 - y1) > 0.2` (or `abs(x2 - x1) > 0.2`) gap checks are necessary to prevent interpolating across disconnected equilibrium branches.

---

## File Map

| Script | Output | Threshold type | Reference line? | Notes |
| -------- | -------- | ---------------- | ----------------- | ------- |
| `bifurcation_mu.py` | `figures/bifurcation_mu.png` | Constant | Vertical grey line | Native axes, no B overlay |
| `bifurcation_mu_with_B.py` | `figures/bifurcation_mu_with_B.png` | Constant | Vertical grey line | With purple B(x*) overlay |
| `bifurcation_W1.py` | `figures/bifurcation_W1.png` | Constant | Vertical grey line | Native axes, no B overlay |
| `bifurcation_W1_with_B.py` | `figures/bifurcation_W1_with_B.png` | Constant | Vertical grey line | With purple B(x*) overlay |
| `bifurcation_beta.py` | `figures/bifurcation_beta.png` | Varying | None | Native axes, no B overlay |
| `bifurcation_beta_with_B.py` | `figures/bifurcation_beta_with_B.png` | Varying | None | With purple B(x*) overlay |
| `bifurcation_rho.py` | `figures/bifurcation_rho.png` | Varying | None | Native axes, no B overlay |
| `bifurcation_rho_with_B.py` | `figures/bifurcation_rho_with_B.png` | Varying | None | With purple B(x*) overlay |
| `combine_figures.py` | `figures/combined_bifurcation.png` | — | — | 2×2 panel of the without-B set |
| `combine_figures_with_B.py` | `figures/combined_bifurcation_with_B.png` | — | — | 2×2 panel of the with-B set |

All scripts must remain consistent with the class-II STP detection algorithm described below.

---

## Core Mathematics

### Model

- Response function: `f₂(x) = μ/(1+exp(β(W₁+ρ(1-2x)))) + (1-μ)/(1+exp(β(W₂+ρ(1-2x))))`
- Fixed points: solve `g(x) = f(x) - x = 0`
- Stability: stable if `|f'(x*)| < 1`, unstable if `> 1`

### B(x) (Proposition 2.13)

```text
Zⱼ = exp(β(Wⱼ + ρ(1-2x)))
Yⱼ = Zⱼ / (1+Zⱼ)²
B(x) = μ·Y₁ + (1-μ)·Y₂
```

### Class II STP Condition

`B(x*) = 1/(2βρ)` marks the degenerate stability transition.

---

## Algorithm Invariants (All Scripts)

For every parameter value in the swept range:

1. **Find all equilibria** `x*` of `g(x) = 0`.
2. **Classify stability** by evaluating `|f'(x*)|`.
3. **Compute `B(x*)`** at each equilibrium.
4. **Detect sign changes** in `B(x*) - 1/(2βρ)` along single branches (skip gaps > 0.2).
5. **Solve exactly** for the parameter value where `g(x*) = 0` and `B(x*) = 1/(2βρ)` using `minimize_scalar` with tight bounds around the interpolated candidate.
6. **Mark the result** with an X.

The with-B scripts add two extra steps:

- Plot the purple `B(x*)` curve via `scale_b_to_axis(B_val, param_range) = B_val * range_width + range_min`.
- Place a circle at the intersection of the purple curve with the threshold, and connect circle ↔ X with a horizontal grey dashed line at the same `y = x*`.

**Resolution exception:** The ρ scripts use `n_points = 5000` and `n_guess = 1000` (vs 1000/200 elsewhere) because the ρ diagram contains sparse regions that produce visible inaccuracies at standard resolution.

### Constant vs Varying Threshold

| Parameter | X-axis range | Threshold `1/(2βρ)` | Reference line | Scale factor for B-axis |
| ----------- | -------------- | --------------------- | ---------------- | ------------------------ |
| μ | [0, 1] | Constant | Vertical line at `1/(2βρ)` | `range_width = 1.0` |
| W₁ | [-1.5, 2] | Constant | Vertical line at `scale_b_to_axis(1/(2βρ), param_range)` | `range_width = 3.5` |
| β | [0, 14] | Varies with β | **None** | `range_width = 14.0` |
| ρ | [0.001, 3] | Varies with ρ | **None** | `range_width = 2.999` |

For μ and W₁ the threshold does not depend on the x-axis parameter, so a vertical reference line is correct. For β and ρ the threshold is a hyperbola in `(parameter, B)` space; because the purple overlay already lives in a different coordinate system (`B × scale_factor`), plotting an extra reference curve in native coordinates creates a misleading visual. The correct visual is the purple overlay plus the circle/X connector only.

---

## Decision Log

1. **Parallel suites (with-B and without-B)**
   - *Why:* The without-B scripts provide clean native-axis bifurcation diagrams suitable for publication or presentations where the B overlay would clutter the figure. The with-B scripts include the purple B(x*) overlay for detailed analysis.
   - *When settled:* After implementing and comparing both versions; both are now kept in sync.

2. **No reference curve for β and ρ**
   - *Why:* Early attempts plotted `B_crit = 1/(2βρ)` as a curve in native axes. This is visually wrong because the purple `B(x*)` overlay uses a scaled horizontal axis (`B × scale_factor`) while the native axis shows the parameter itself. The only correct markers are the circle (on the purple curve) and the X (on the native parameter axis), joined by a horizontal line.
   - *When settled:* After debugging misaligned markers in `bifurcation_beta_with_B.py`.

3. **Scaled B-axis via `scale_b_to_axis`**
   - *Why:* Raw `B(x*)` values are small (≈ 0–0.25). Scaling them to the native parameter range makes the curve visible and keeps the four-panel figure visually consistent. All eight scripts now use the same helper: `B_val * range_width + range_min`.
   - *When settled:* During the refactor to consolidate cross-script differences.

4. **Exact solving with `minimize_scalar`**
   - *Why:* Simple interpolation of the discrete equilibrium curve is not accurate enough for publication-quality figures. We first interpolate to get a candidate `x*`, then use `minimize_scalar` on `g(x*, param)² + (B(x*) - B_crit)²` with tight bounds to pin down the exact parameter value.

5. **Arial font throughout**
   - *Why:* Consistent journal-style formatting across all figures.

6. **Reference parameters updated to μ = 0.4**
   - *Why:* `mu = 0.54` was an older reference model. The current reference is `μ = 0.4, β = 14, ρ = 1, W₁ = -0.6, W₂ = 0.3`. All `DEFAULT_PARAMS` dictionaries and the `mu_value=0.4` signature defaults now match.
   - *Consequence:* There is no practical change to the generated figures, because the β/ρ/W₁ scripts already overrode `mu` to 0.4 via their signature defaults. The change only removes a source of confusion.
   - *When settled:* During the cross-script consistency check.

7. **`find_turning_points` used only for branch thickness in μ**
   - *Why:* The without-B `bifurcation_mu.py` was accidentally using class-II STP y-values to split branches for point-size styling, while the with-B version used class-I turning points. Both μ scripts now use `find_turning_points` (class-I) for the split, and the B-criterion exclusively for the X-marks.
   - *When settled:* During the cross-script consistency check.

8. **Higher resolution for ρ**
   - *Why:* The ρ diagram contains sparse regions that produced visible inaccuracies at the standard 1000 × 200 resolution.
   - *When settled:* After observing artefacts in early ρ plots.

---

## Common Agent Pitfalls

- **Confusing scaled and raw B values.** The purple curve plots `scale_b_to_axis(B_val, param_range)`, but the threshold condition `B(x*) = 1/(2βρ)` uses the raw `B(x*)`. Always compare raw values when detecting sign changes, then scale only for plotting.
- **Trying to add a vertical line to β or ρ diagrams.** The threshold varies with the x-axis parameter here; a vertical line would only be correct for a single parameter value.
- **Removing the `*_with_B.py` / `*.py` duplication.** Both suites are intentionally maintained. If one set breaks, fix it; do not consolidate.
- **Relaxing gap thresholds.** The `> 0.2` gap checks prevent the interpolator from stitching together unrelated equilibrium branches (e.g., a high-x stable branch and a low-x unstable branch). Removing or widening this causes spurious X-marks.
- **Using `find_turning_points` for class-II STP placement.** This function tracks branch appearances / disappearances (class-I). Class-II STPs must always be found via the B-criterion. In the μ scripts, `find_turning_points` is used only to decide which stable branch gets the thicker scatter points.
- **Changing marker styles.** The colour and marker conventions are fixed (green stable, red unstable, purple B, black circle, black X). Altering these breaks consistency with published/combined figures.

---

## Visual Conventions

| Element | Style |
| --------- | ------- |
| Stable equilibria | Green (`#6aa84f`) |
| Unstable equilibria (class I STPs) | Red |
| B(x*) curve | Purple |
| B/threshold intersection | Black circle (`ko`, open) |
| Class II STP exact location | Black X (`kx`) |
| Horizontal connector | Grey dashed line |
| Vertical threshold reference | Grey dashed line (μ, W₁ only) |
| Font | Arial |
| Title | 16 pt |
| Axis labels | 13 pt |
| Ticks | 12 pt |
| Legend | 13–14 pt |
