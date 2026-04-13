# Bifurcation Diagram Generation Logic

## Overview
This document describes the logic for generating bifurcation diagrams with B(x*) overlay for the f₂(x) model. The key challenge is correctly identifying and visualizing class II social tipping points (STPs) where B(x*) = 1/(2βρ).

## Core Mathematics

### The Model
- Response function: f₂(x) = μ/(1+exp(β(W₁+ρ(1-2x)))) + (1-μ)/(1+exp(β(W₂+ρ(1-2x))))
- Fixed points: g(x) = f(x) - x = 0
- Stability: |f'(x*)| < 1 for stable, > 1 for unstable

### B(x) Calculation (Proposition 2.13)
```
Zⱼ = exp(β(Wⱼ + ρ(1-2x)))
Yⱼ = Zⱼ/(1+Zⱼ)²
B(x) = μ·Y₁ + (1-μ)·Y₂
```

### Class II STP Condition
B(x*) = 1/(2βρ) is the threshold where equilibria transition stability in a degenerate way.

---

## Parameter-Specific Logic

### CRITICAL INSIGHT: The Threshold Depends on Which Parameter Varies

The condition B(x*) = 1/(2βρ) behaves differently depending on which parameter is on the x-axis:

#### 1. μ varies (x-axis = μ ∈ [0,1])
- **Threshold**: 1/(2βρ) is CONSTANT (doesn't depend on μ)
- **Visualization**: Vertical line at x = 1/(2βρ) ≈ 0.036
- **Circle placement**: Where purple B(x*) curve crosses the vertical line
- **Why it works**: μ doesn't appear in the threshold, so fixed vertical line is correct

#### 2. β varies (x-axis = β ∈ [0,14])
- **Threshold**: 1/(2βρ) varies with β
- **Visualization**: Purple B(x*) overlay uses horizontal axis `B(x*) × 14`. No reference curve is plotted in native coordinates because the threshold and the purple curve live in different coordinate systems.
- **Circle placement**: Where purple B(x*) curve satisfies `B(x*) = 1/(2βρ)`
- **Key insight**: Cannot use vertical line - the threshold itself changes with β

#### 3. ρ varies (x-axis = ρ ∈ [0.001, 3])
- **Threshold**: 1/(2βρ) varies with ρ
- **Visualization**: Purple B(x*) overlay uses horizontal axis `B(x*) × 3`. No reference curve is plotted in native coordinates.
- **Circle placement**: Where purple B(x*) curve satisfies `B(x*) = 1/(2βρ)`

#### 4. W₁ varies (x-axis = W₁ ∈ [-1.5, 2])
- **Threshold**: 1/(2βρ) is CONSTANT (doesn't depend on W₁)
- **Visualization**: Vertical line at scaled position
- **Similar to μ**: W₁ doesn't appear in threshold

---

## Working Algorithm (for μ and W₁ - CONSTANT threshold)

These work with a vertical reference line:

### Step 1: Generate Equilibrium Curves
```python
for param_val in param_range:
    fps = find_fixed_points(...)  # All equilibria at this parameter value
    classify as stable/unstable
    store (param_val, x*) pairs
```

### Step 2: Calculate B(x*) at Equilibrium Points
```python
for each (param_val, x_val) in equilibria:
    B_val = calculate_B(x_val, mu_at_this_point, beta, rho, W1, W2)
    B_x_vals.append(B_val)  # x-coordinate is B value (0-0.25 range)
    B_y_vals.append(x_val)  # y-coordinate is x* (0-1 range)
    # Note: For scaled axes, multiply B_val by scale_factor
```

### Step 3: Plot Reference Line
```python
# For μ or W1: vertical line at constant threshold
normalized_critical = 1 / (2 * beta * rho)
# Scale to axis range (e.g., ×1 for μ, ×3.5 for W1)
ax.axvline(x=normalized_critical * scale_factor, ...)
```

### Step 4: Find Intersections (Circles)
```python
target_B = normalized_critical * scale_factor
nearby_threshold = 0.1 * scale_factor  # Only consider B within this range

# Filter purple points near the target
purple_points = [(y, B) for y, B in zip(B_y_vals, B_x_vals) 
                 if abs(B - target_B) < nearby_threshold]
purple_points = sorted(set(purple_points))  # Sort by y

# Find sign changes in (B - target_B) = intersection
intersection_ys = []
for i in range(len(purple_points) - 1):
    y1, B1 = purple_points[i]
    y2, B2 = purple_points[i + 1]
    
    if abs(y2 - y1) > 0.2:  # Skip large gaps (different branches)
        continue
    
    if (B1 - target_B) * (B2 - target_B) < 0:  # Sign change
        t = (target_B - B1) / (B2 - B1)  # Linear interpolation
        y_interp = y1 + t * (y2 - y1)
        intersection_ys.append(y_interp)
```

### Step 5: Place Circles and Draw Lines to X Marks
```python
for y_val in intersection_ys:
    # Circle at intersection with vertical line
    ax.plot(mu_critical, y_val, 'ko', markerfacecolor='none', ...)
    
    # Find closest turning point at this y
    # Turning points are class I STPs (where branches appear/disappear)
    best_tx = None
    best_dist = float('inf')
    for tx, ty in zip(turning_x, turning_y):
        dist = abs(ty - y_val)
        if dist < best_dist:
            best_dist = dist
            best_tx = tx
    
    if best_tx is not None and best_dist < 0.15:  # Threshold for matching
        # Horizontal line from circle to turning point
        ax.plot([mu_critical, best_tx], [y_val, y_val], 'grey', linestyle='--')
        # X mark at turning point
        ax.plot(best_tx, y_val, 'kx', markersize=10)
```

---

## Algorithm for β and ρ (VARYING threshold)

### Key Difference
The threshold 1/(2βρ) changes with the x-axis parameter. The purple `B(x*)` overlay uses a different horizontal scale (`B × scale_factor`), so we do **not** plot a reference curve in the native axes.

### Step 1-2: Same as above (generate equilibria, calculate B(x*))

### Step 3: Build Equilibrium Data with B_crit
```python
# For each equilibrium, calculate both B(x*) and B_crit at that parameter
for beta_val, x_val in zip(all_beta_vals, all_x_vals):
    B_val = calculate_B(x_val, mu, beta_val, rho, W1, W2)
    B_scaled = B_val * 14
    B_crit = 1 / (2 * beta_val * rho) if beta_val > 0.1 else inf
    equilibrium_data.append((beta_val, x_val, B_scaled, B_val, B_crit))
```

### Step 4: Find Intersections (Where B(x*) = B_crit)
```python
# Sort by x* to follow branches
equilibrium_data.sort(key=lambda t: t[1])

intersection_ys = []
for i in range(len(equilibrium_data) - 1):
    beta1, x1, Bs1, Bv1, Bc1 = equilibrium_data[i]
    beta2, x2, Bs2, Bv2, Bc2 = equilibrium_data[i+1]
    
    # Only consider nearby points (same branch)
    if abs(x2 - x1) > 0.2 or abs(beta2 - beta1) > 1:
        continue
    
    diff1 = Bv1 - Bc1
    diff2 = Bv2 - Bc2
    
    if diff1 * diff2 < 0:  # Sign change = intersection
        t = abs(diff1) / (abs(diff1) + abs(diff2))
        y_interp = x1 + t * (x2 - x1)
        intersection_ys.append(y_interp)
```

### Step 5: Find Exact Intersection Points and Place Markers
```python
for y_val in intersection_ys:
    # Solve for exact beta where: g(y_val, beta) = 0 AND B(y_val, beta) = 1/(2*beta*rho)
    def intersection_condition(beta):
        g_val = fixed_point_equation(y_val, mu, beta, rho, W1, W2)
        B_val = calculate_B(y_val, mu, beta, rho, W1, W2)
        B_crit = 1 / (2 * beta * rho)
        return g_val**2 + (B_val - B_crit)**2
    
    result = minimize_scalar(intersection_condition, bounds=(0.5, 14), method='bounded')
    
    if result.fun < 0.001:  # Good solution found
        beta_at_int = result.x
        B_at_int = calculate_B(y_val, mu, beta_at_int, rho, W1, W2)
        B_scaled = B_at_int * 14
        
        # Circle at (B_scaled, y_val) - on purple curve at intersection
        ax.plot(B_scaled, y_val, 'ko', markerfacecolor='none', ...)
        
        # X mark at the exact beta where g(y_val, beta) = 0 and B = B_crit
        ax.plot(beta_at_int, y_val, 'kx', ...)
        
        # Horizontal dashed line connecting them
        ax.plot([B_scaled, beta_at_int], [y_val, y_val], 'grey', linestyle='--')
```

For ρ, the logic is identical but with scale factor 3 and appropriate bounds.

---

## Scaling Reference

| Parameter | Range | Scale Factor | Offset | Vertical Line Position |
|-----------|-------|--------------|--------|------------------------|
| μ | [0, 1] | 1 | 0 | 1/(2βρ) ≈ 0.036 |
| β | [0, 14] | 14 | 0 | Curve: B_crit(β) = 1/(2βρ) |
| ρ | [0.001, 3] | 3 | 0.001 | Curve: B_crit(ρ) = 1/(2βρ) |
| W₁ | [-1.5, 2] | 3.5 | -1.5 | 1/(2βρ) scaled to range |

Note: For β and ρ, the threshold varies, so we plot a curve, not a vertical line.

---

## Critical Lessons Learned

### 1. Threshold Constancy Matters
- **μ and W₁**: Threshold 1/(2βρ) is constant → vertical line works
- **β and ρ**: Threshold varies with parameter → needs hyperbolic curve

### 2. Intersection Detection
- Must sort points by y-value (x*) to follow branches
- Skip large gaps (>0.2 in y) to avoid interpolating across branches
- Use sign change detection: (B1 - target) × (B2 - target) < 0

### 3. Circle Placement
- Circles go at the B(x*) curve position: (B_scaled, y_int)

### 4. X Mark Placement
- X marks go at the exact parameter value where `g(x*) = 0` AND `B(x*) = 1/(2βρ)`
- Horizontal lines connect circles to X marks (same y-value)
- For μ and W₁ (constant threshold), X marks are found by minimizing `|g(y, param)|` along the horizontal line from the vertical reference line.
- For β and ρ (varying threshold), X marks are found by the same minimization after locating the intersection on the purple curve.

### 5. Avoiding False Positives
- For varying thresholds (β, ρ), must solve both g(x*)=0 AND B=B_crit simultaneously
- Use the interpolated parameter value from Step 4 to set tight bounds for `minimize_scalar`
- This prevents the optimizer from wandering to distant branches

### 6. The Beta/Rho Coordinate Confusion
The threshold `1/(2βρ)` is a function of the x-axis parameter. Because the purple `B(x*)` curve uses a different horizontal scale (`B × scale_factor`), plotting the threshold as a curve in native coordinates creates a misleading visual. The correct approach is to show only the purple overlay and the horizontal connectors; the circles themselves mark where the intersection condition is satisfied.

---

## File Status

| File | Status | Notes |
|------|--------|-------|
| bifurcation_mu_with_B.py | Working | Constant threshold, vertical line |
| bifurcation_W1_with_B.py | Working | Constant threshold, vertical line |
| bifurcation_beta_with_B.py | Working | Varying threshold, purple overlay only |
| bifurcation_rho_with_B.py | Working | Varying threshold, purple overlay only |

---

## Next Steps (post-merge)

1. **Back-port intersection logic** to legacy `bifurcation_*.py` scripts (without B overlay)
2. **Rename files** so the current `_with_B` versions become the canonical `bifurcation_*.py`
3. **Archive or delete** the pre-B legacy scripts
4. **Update this document** to reflect the final file names

---

## Visual Conventions

- **Green (#6aa84f)**: Stable equilibria
- **Red**: Unstable equilibria (class I STPs)
- **Purple**: B(x*) curve
- **Black circles (o)**: Intersection of B(x*) with threshold
- **Black X marks**: Class II STPs (at turning points)
- **Grey dashed lines**:
  - Vertical: Constant threshold reference (μ, W₁)
  - Horizontal: Connection from circle to X mark (all four parameters)

Font: Arial throughout
Title: 16pt
Axis labels: 13pt
Ticks: 12pt
Legend: 13-14pt
