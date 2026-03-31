# Second Method: Matrix and Tone Curve Pipeline (Steps 3-7)

**File:** `src/bin/second_method/matrix_tone_correction.rs`

## Overview

This is the **first major component** of the second method workflow. It uses **SVD-based least-squares** to solve a global 3×3 color transformation matrix, then fits a **256-bin tone curve** with linear interpolation to correct luminance relationships while preserving chroma. This two-stage approach separates color correction (matrix) from brightness correction (tone curve), resulting in a compact and efficient transformation.

---

## Table of Contents

1. [Purpose and Motivation](#purpose-and-motivation)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Step 3: Solving Color Matrix with SVD](#step-3-solving-color-matrix-with-svd)
4. [Step 4: Applying the Matrix](#step-4-applying-the-matrix)
5. [Step 5: Computing Luminance](#step-5-computing-luminance)
6. [Step 6: Fitting Tone Curve](#step-6-fitting-tone-curve)
7. [Step 7: Applying Tone Curve with Chroma Preservation](#step-7-applying-tone-curve-with-chroma-preservation)
8. [Complete Workflow](#complete-workflow)
9. [Mathematical Properties](#mathematical-properties)

---

## Purpose and Motivation

### Why Matrix + Tone Curve?

**Problem:** Film simulations have two primary components:
1. **Color shifts:** Changes in hue and saturation relationships
2. **Tone response:** Changes in brightness and contrast

**Approach 1 (First Method):** Use a single 33×33×33 3D LUT to capture everything
- **Pros:** Can model any color transformation
- **Cons:** Large file size (~950 KB), requires interpolation to fill sparse regions

**Approach 2 (Second Method):** Separate into matrix (color) + tone curve (brightness)
- **Pros:** Compact representation (~140 KB), mathematically principled
- **Cons:** May not capture complex local color variations

**Solution:** Use matrix+tone for global correction, then add a small **residual 3D LUT** to capture remaining local variations.

### Pipeline Overview

```
Source RGB → [Matrix] → Matrix RGB → [Tone Curve] → Tone RGB → [Residual LUT] → Final RGB
     ↓              ↓                      ↓                         ↓
   Step 3        Step 4                 Step 6-7                  Step 8-11
```

**This file handles Steps 3-7:** Matrix solving and tone curve fitting.

---

## Mathematical Foundation

### The Color Transformation Problem

Given training data:
- $\mathbf{s}_i = [s_r, s_g, s_b]_i$ = source RGB (Standard image)
- $\mathbf{c}_i = [c_r, c_g, c_b]_i$ = target RGB (Classic Chrome image)
- $N$ = number of training samples (~100,000)

**Goal:** Find transformation $T$ such that $T(\mathbf{s}_i) \approx \mathbf{c}_i$ for all $i$.

### Matrix-Only Approach

First, try a **global linear transformation** (3×3 matrix):

$$
\mathbf{M} \cdot \mathbf{s}_i \approx \mathbf{c}_i
$$

Where:

$$
\mathbf{M} = \begin{bmatrix}
m_{11} & m_{12} & m_{13} \\
m_{21} & m_{22} & m_{23} \\
m_{31} & m_{32} & m_{33}
\end{bmatrix}
$$

**Problem:** This is an **overdetermined system**:
- 9 unknowns (matrix entries)
- 300,000 equations (3 per sample)
- No exact solution exists (data has noise and nonlinearities)

**Solution:** Use **least-squares** to find the "best fit" matrix.

### Least-Squares Formulation

Rewrite as matrix equation:

$$
\mathbf{X} \mathbf{M} = \mathbf{Y}
$$

Where:
- $\mathbf{X}$ = $N \times 3$ matrix of source RGB values
- $\mathbf{M}$ = $3 \times 3$ color transformation matrix (unknown)
- $\mathbf{Y}$ = $N \times 3$ matrix of target RGB values

**Objective:** Minimize the Frobenius norm of the residual:

$$
\min_{\mathbf{M}} \lVert\mathbf{X} \mathbf{M} - \mathbf{Y}\rVert_F^2
$$

---

## Step 3: Solving Color Matrix with SVD

### SVD-Based Least-Squares

**Why SVD?** Singular Value Decomposition is numerically stable for overdetermined systems and automatically handles rank-deficient matrices.

**Theory:** Given $\mathbf{X} \mathbf{M} = \mathbf{Y}$, the SVD solution is:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

$$
\mathbf{M} = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{U}^T \mathbf{Y}
$$

Where:
- $\mathbf{U}$ = $N \times 3$ left singular vectors
- $\mathbf{\Sigma}$ = $3 \times 3$ diagonal matrix of singular values
- $\mathbf{V}$ = $3 \times 3$ right singular vectors
- $\mathbf{\Sigma}^{-1}$ = pseudo-inverse (replaces small singular values with 0)

**Code from `matrix_tone_correction.rs` (lines 168-208):**

```rust
use nalgebra::DMatrix;

fn solve_color_matrix(pixels: &[Pixel]) -> Result<[[f32; 3]; 3]> {
  let n = pixels.len();

  // Build matrices X (source) and Y (target)
  let mut x_data = vec![0.0f64; n * 3];
  let mut y_data = vec![0.0f64; n * 3];

  for (i, pixel) in pixels.iter().enumerate() {
    // X[i, :] = [source_r, source_g, source_b]
    x_data[i * 3] = pixel.source[0] as f64;
    x_data[i * 3 + 1] = pixel.source[1] as f64;
    x_data[i * 3 + 2] = pixel.source[2] as f64;

    // Y[i, :] = [target_r, target_g, target_b]
    y_data[i * 3] = pixel.target[0] as f64;
    y_data[i * 3 + 1] = pixel.target[1] as f64;
    y_data[i * 3 + 2] = pixel.target[2] as f64;
  }

  // Construct nalgebra matrices
  let x = DMatrix::from_row_slice(n, 3, &x_data);
  let y = DMatrix::from_row_slice(n, 3, &y_data);

  // Solve M using SVD least-squares
  // Tolerance 1e-14 for numerical stability
  let svd = x.svd(true, true);
  let m = svd
    .solve(&y, 1e-14)
    .map_err(|_| anyhow::anyhow!("Failed to solve least squares with SVD"))?;

  // Convert to f32 matrix
  let mut matrix = [[0.0f32; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      matrix[i][j] = m[(i, j)] as f32;
    }
  }

  Ok(matrix)
}
```

### Algorithm Steps

**Step 1: Build Data Matrices**

From $N$ training samples, construct:

$$
\mathbf{X} = \begin{bmatrix}
s_{r,1} & s_{g,1} & s_{b,1} \\
s_{r,2} & s_{g,2} & s_{b,2} \\
\vdots & \vdots & \vdots \\
s_{r,N} & s_{g,N} & s_{b,N}
\end{bmatrix}_{N \times 3}
\quad
\mathbf{Y} = \begin{bmatrix}
c_{r,1} & c_{g,1} & c_{b,1} \\
c_{r,2} & c_{g,2} & c_{b,2} \\
\vdots & \vdots & \vdots \\
c_{r,N} & c_{g,N} & c_{b,N}
\end{bmatrix}_{N \times 3}
$$

**Step 2: Compute SVD**

```rust
let svd = x.svd(true, true);
```

This decomposes $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$.

**Step 3: Solve Least-Squares**

```rust
let m = svd.solve(&y, 1e-14)?;
```

This computes:

$$
\mathbf{M} = \mathbf{V} \mathbf{\Sigma}^{+} \mathbf{U}^T \mathbf{Y}
$$

Where $\mathbf{\Sigma}^{+}$ is the pseudo-inverse (singular values $< 10^{-14}$ are treated as zero).

**Step 4: Extract Result**

The result is a $3 \times 3$ matrix:

$$
\mathbf{M} = \begin{bmatrix}
0.xxxxx & 0.xxxxx & 0.xxxxx \\
0.xxxxx & 0.xxxxx & 0.xxxxx \\
0.xxxxx & 0.xxxxx & 0.xxxxx
\end{bmatrix}
$$

### Example Output

```
Global color matrix:
  [ 0.89234,  0.05123, -0.01456]
  [-0.02145,  0.91234,  0.03456]
  [ 0.01234, -0.03456,  0.95678]
```

**Interpretation:**
- Diagonal entries (~0.9): preserve most of the original channel
- Off-diagonal entries (~0.05): cross-channel color mixing
- Negative values: reduce complementary colors

---

## Step 4: Applying the Matrix

### Matrix Multiplication

**Formula:**

$$
\begin{bmatrix} r' \\ g' \\ b' \end{bmatrix} = \begin{bmatrix} m_{11} & m_{12} & m_{13} \\ m_{21} & m_{22} & m_{23} \\ m_{31} & m_{32} & m_{33} \end{bmatrix} \begin{bmatrix} r \\ g \\ b \end{bmatrix}
$$

**Expanded:**

$$
\begin{aligned}
r' &= m_{11} \cdot r + m_{12} \cdot g + m_{13} \cdot b \\
g' &= m_{21} \cdot r + m_{22} \cdot g + m_{23} \cdot b \\
b' &= m_{31} \cdot r + m_{32} \cdot g + m_{33} \cdot b
\end{aligned}
$$

**Code from `matrix_tone_correction.rs` (lines 210-217):**

```rust
fn apply_matrix(matrix: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
  let r = matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2];
  let g = matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2];
  let b = matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2];

  // Clamp to valid range [0, 1]
  [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}
```

### Clamping

**Why clamp?** Matrix multiplication can produce out-of-range values:
- Negative values (e.g., $-0.05$) → clamp to $0.0$
- Values > 1 (e.g., $1.12$) → clamp to $1.0$

**Trade-off:**
- **Preserves valid RGB range**
- **May introduce slight clipping artifacts** (rare in practice)

### Error Analysis

After applying the matrix:

$$
\text{Error}_{\text{matrix}} = \frac{1}{3N} \sum_{i=1}^{N} \sum_{c \in \{r,g,b\}} |(\mathbf{M} \cdot \mathbf{s}_i)_c - (\mathbf{c}_i)_c|
$$

**Typical result:**
```
Mean absolute error (source -> target):  0.115342
Mean absolute error (matrix -> target): 0.043256
Improvement (matrix):                    0.072086  (62.5% reduction)
```

The matrix captures most of the color transformation, but **brightness relationships still need correction**.

---

## Step 5: Computing Luminance

### Rec.709 Luminance Formula

**Why luminance?** Human perception is more sensitive to brightness than color. We need to correct brightness while preserving hue.

**Definition:** Rec.709 standard for RGB-to-luminance conversion:

$$
Y = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B
$$

**Properties:**
- **Green dominates** (0.7152): Human eyes are most sensitive to green
- **Blue contributes least** (0.0722): Human eyes are least sensitive to blue
- **Weighted sum** ensures perceptual brightness accuracy

**Code from `matrix_tone_correction.rs` (line 9-11, 219-221):**

```rust
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

fn compute_luminance(rgb: [f32; 3]) -> f32 {
  LUM_R * rgb[0] + LUM_G * rgb[1] + LUM_B * rgb[2]
}
```

### Computing Luminance for Training Data

**Step 1: Compute luminance of matrix output**

$$
Y_{\text{matrix},i} = 0.2126 \cdot r'_i + 0.7152 \cdot g'_i + 0.0722 \cdot b'_i
$$

Where $(r'_i, g'_i, b'_i) = \mathbf{M} \cdot \mathbf{s}_i$ is the matrix-corrected RGB.

**Step 2: Compute luminance of target**

$$
Y_{\text{target},i} = 0.2126 \cdot c_{r,i} + 0.7152 \cdot c_{g,i} + 0.0722 \cdot c_{b,i}
$$

**Result:** Two arrays of luminance values:
- `y_matrix`: luminance after matrix correction
- `y_target`: target luminance (ground truth)

**Goal:** Find a 1D function $f$ such that $f(Y_{\text{matrix},i}) \approx Y_{\text{target},i}$.

---

## Step 6: Fitting Tone Curve

### Problem Statement

Given paired luminance data $\{(Y_{\text{matrix},i}, Y_{\text{target},i})\}_{i=1}^{N}$, find a **tone curve** $f : [0,1] \rightarrow [0,1]$ that maps input luminance to output luminance.

### Binning Strategy

**Why bins?** With 100,000 samples, we can't store every individual mapping. Instead, discretize into bins.

**Parameters:**
```rust
const TONE_BINS: usize = 256;
```

**Bin mapping:**

$$
\text{bin\_idx} = \left\lfloor Y_{\text{matrix}} \times (\text{TONE\_BINS} - 1) + 0.5 \right\rfloor
$$

This uses **rounding** (not flooring) for better distribution.

**Example:** For $Y_{\text{matrix}} = 0.503$ with 256 bins:

$$
\text{bin\_idx} = \lfloor 0.503 \times 255 + 0.5 \rfloor = \lfloor 128.765 + 0.5 \rfloor = \lfloor 129.265 \rfloor = 129
$$

### Accumulation and Averaging

**Code from `matrix_tone_correction.rs` (lines 234-264):**

```rust
const TONE_BINS: usize = 256;

fn fit_tone_curve(y_matrix: &[f32], y_target: &[f32]) -> Vec<f32> {
  // Initialize bins
  let mut bin_sums = vec![0.0f32; TONE_BINS];
  let mut bin_counts = vec![0u32; TONE_BINS];

  // Step 1: Accumulate values into bins
  for (&y_m, &y_t) in y_matrix.iter().zip(y_target.iter()) {
    let bin_idx = ((y_m * (TONE_BINS - 1) as f32).round() as usize).min(TONE_BINS - 1);
    bin_sums[bin_idx] += y_t;
    bin_counts[bin_idx] += 1;
  }

  // Step 2: Compute averages for non-empty bins
  let mut curve = vec![0.0f32; TONE_BINS];
  for i in 0..TONE_BINS {
    if bin_counts[i] > 0 {
      curve[i] = bin_sums[i] / bin_counts[i] as f32;
    }
  }

  // Step 3: Fill empty bins by interpolation
  fill_missing_bins(&mut curve, &bin_counts);

  // Step 4: Smooth the curve
  smooth_curve(&mut curve, SMOOTH_WINDOW);

  // Step 5: Enforce monotonicity
  for i in 1..curve.len() {
    curve[i] = curve[i].max(curve[i - 1]);
  }

  curve
}
```

### Algorithm Steps

**Step 1: Accumulate**

For each training sample $(Y_{\text{matrix},i}, Y_{\text{target},i})$:

$$
\begin{aligned}
\text{bin\_idx} &= \text{round}(Y_{\text{matrix},i} \times 255) \\
\text{bin\_sums}[\text{bin\_idx}] &\leftarrow \text{bin\_sums}[\text{bin\_idx}] + Y_{\text{target},i} \\
\text{bin\_counts}[\text{bin\_idx}] &\leftarrow \text{bin\_counts}[\text{bin\_idx}] + 1
\end{aligned}
$$

**Step 2: Average**

For bins with data:

$$
\text{curve}[i] = \frac{\text{bin\_sums}[i]}{\text{bin\_counts}[i]}
$$

For empty bins: $\text{curve}[i] = 0$ (temporarily).

**Step 3: Fill Empty Bins**

Use **linear interpolation** between nearest non-empty bins.

**Code from `matrix_tone_correction.rs` (lines 266-309):**

```rust
fn fill_missing_bins(curve: &mut [f32], counts: &[u32]) {
  let n = curve.len();

  for i in 0..n {
    if counts[i] == 0 {
      // Find nearest non-empty bins on left and right
      let mut left_val = None;
      let mut right_val = None;
      let mut left_dist = None;
      let mut right_dist = None;

      // Search left
      for j in (0..i).rev() {
        if counts[j] > 0 {
          left_val = Some(curve[j]);
          left_dist = Some(i - j);
          break;
        }
      }

      // Search right
      for j in (i + 1)..n {
        if counts[j] > 0 {
          right_val = Some(curve[j]);
          right_dist = Some(j - i);
          break;
        }
      }

      // Interpolate based on distance
      curve[i] = match (left_val, right_val) {
        (Some(l), Some(r)) => {
          // Weighted average by inverse distance
          let ld = left_dist.unwrap() as f32;
          let rd = right_dist.unwrap() as f32;
          (l * rd + r * ld) / (ld + rd)
        }
        (Some(l), None) => l,  // Only left neighbor exists
        (None, Some(r)) => r,  // Only right neighbor exists
        (None, None) => i as f32 / (n - 1) as f32,  // Fallback to identity
      };
    }
  }
}
```

**Interpolation formula:**

$$
\text{curve}[i] = \frac{v_L \cdot d_R + v_R \cdot d_L}{d_L + d_R}
$$

Where:
- $v_L$ = value of left neighbor
- $v_R$ = value of right neighbor
- $d_L$ = distance to left neighbor
- $d_R$ = distance to right neighbor

**Example:**
- Bin 100 is empty
- Left neighbor: bin 95 with value 0.45
- Right neighbor: bin 110 with value 0.55

$$
\text{curve}[100] = \frac{0.45 \times 10 + 0.55 \times 5}{5 + 10} = \frac{4.5 + 2.75}{15} = \frac{7.25}{15} \approx 0.483
$$

**Step 4: Smooth**

Apply a **moving average** filter to reduce noise:

```rust
const SMOOTH_WINDOW: usize = 5;

fn smooth_curve(curve: &mut [f32], window: usize) {
  let n = curve.len();
  let mut smoothed = curve.to_vec();

  for i in 0..n {
    let start = i.saturating_sub(window / 2);
    let end = (i + window / 2 + 1).min(n);
    let sum: f32 = curve[start..end].iter().sum();
    let count = (end - start) as f32;
    smoothed[i] = sum / count;
  }

  curve.copy_from_slice(&smoothed);
}
```

**Formula:**

$$
\text{curve}[i] = \frac{1}{w} \sum_{j=i-w/2}^{i+w/2} \text{curve}[j]
$$

Where $w = 5$ is the window size.

**Step 5: Enforce Monotonicity**

Ensure the curve never decreases (brighter input → brighter output):

```rust
for i in 1..curve.len() {
  curve[i] = curve[i].max(curve[i - 1]);
}
```

**Mathematical constraint:**

$$
\text{curve}[i] \geq \text{curve}[i-1] \quad \forall i \in [1, 255]
$$

This prevents **brightness inversions** where a brighter pixel becomes darker.

### Example Tone Curve

```
Tone curve samples:
  Tone curve[0]   = 0.0034
  Tone curve[32]  = 0.1245
  Tone curve[64]  = 0.2456
  Tone curve[96]  = 0.3712
  Tone curve[128] = 0.5023
  Tone curve[160] = 0.6345
  Tone curve[192] = 0.7634
  Tone curve[224] = 0.8912
  Tone curve[255] = 1.0000
```

**Interpretation:** This tone curve is **approximately linear** with slight adjustments:
- Shadows (0-32): slight lift
- Midtones (64-192): mostly preserved
- Highlights (224-255): slight compression

---

## Step 7: Applying Tone Curve with Chroma Preservation

### The Chroma Preservation Problem

**Naive approach:** Apply tone curve independently to R, G, B:

$$
\begin{aligned}
r'' &= f(r') \\
g'' &= f(g') \\
b'' &= f(b')
\end{aligned}
$$

**Problem:** This **destroys color relationships** and causes hue shifts.

**Example:**
- Input: $(r', g', b') = (0.5, 0.3, 0.2)$ (orange)
- After independent curves: $(r'', g'', b'') = (0.55, 0.35, 0.22)$
- **Hue has changed** (no longer the same orange)

### Chroma-Preserving Tone Curve

**Solution:** Apply tone curve to **luminance only**, then scale RGB proportionally.

**Algorithm:**

1. Compute input luminance: $Y_{\text{old}} = 0.2126 \cdot r' + 0.7152 \cdot g' + 0.0722 \cdot b'$
2. Look up output luminance: $Y_{\text{new}} = f(Y_{\text{old}})$
3. Compute scale factor: $s = Y_{\text{new}} / Y_{\text{old}}$
4. Apply scale to RGB: $(r'', g'', b'') = s \cdot (r', g', b')$

**Mathematical formulation:**

$$
\begin{bmatrix} r'' \\ g'' \\ b'' \end{bmatrix} = \frac{f(Y_{\text{old}})}{Y_{\text{old}}} \begin{bmatrix} r' \\ g' \\ b' \end{bmatrix}
$$

**Property:** This preserves the **ratio** of RGB channels, maintaining hue and saturation:

$$
\frac{r''}{g''} = \frac{r'}{g'}, \quad \frac{r''}{b''} = \frac{r'}{b'}, \quad \frac{g''}{b''} = \frac{g'}{b'}
$$

### Linear Interpolation for Smooth Lookup

**Problem:** The tone curve is discrete (256 bins), but input luminance is continuous.

**Solution:** Use **linear interpolation** between adjacent bins.

**Code from `matrix_tone_correction.rs` (lines 313-331):**

```rust
fn apply_tone_curve(tone_curve: &[f32], rgb: [f32; 3], y_old: f32) -> [f32; 3] {
  // Linear interpolation for smooth tone curve lookup
  let pos = y_old.clamp(0.0, 1.0) * (TONE_BINS - 1) as f32;
  let i0 = pos.floor() as usize;
  let i1 = (i0 + 1).min(TONE_BINS - 1);
  let t = pos - i0 as f32;

  // Interpolated output luminance
  let y_new = tone_curve[i0] * (1.0 - t) + tone_curve[i1] * t;

  // Compute scale factor for chroma preservation
  let scale = if y_old > 1e-6 { y_new / y_old } else { 1.0 };

  // Apply scale and clamp
  [
    (rgb[0] * scale).clamp(0.0, 1.0),
    (rgb[1] * scale).clamp(0.0, 1.0),
    (rgb[2] * scale).clamp(0.0, 1.0),
  ]
}
```

### Interpolation Mathematics

Given input luminance $Y_{\text{old}} \in [0, 1]$:

**Step 1: Compute continuous position**

$$
\text{pos} = Y_{\text{old}} \times (\text{TONE\_BINS} - 1) = Y_{\text{old}} \times 255
$$

**Step 2: Find bounding indices**

$$
\begin{aligned}
i_0 &= \lfloor \text{pos} \rfloor \\
i_1 &= \min(i_0 + 1, 255) \\
t &= \text{pos} - i_0 \quad \text{(fractional part)}
\end{aligned}
$$

**Step 3: Linear interpolation**

$$
Y_{\text{new}} = \text{curve}[i_0] \cdot (1 - t) + \text{curve}[i_1] \cdot t
$$

**Example:** For $Y_{\text{old}} = 0.503$:

$$
\begin{aligned}
\text{pos} &= 0.503 \times 255 = 128.265 \\
i_0 &= 128, \quad i_1 = 129 \\
t &= 0.265 \\
Y_{\text{new}} &= \text{curve}[128] \cdot 0.735 + \text{curve}[129] \cdot 0.265
\end{aligned}
$$

**Step 4: Compute scale**

$$
s = \frac{Y_{\text{new}}}{Y_{\text{old}}}
$$

**Special case:** If $Y_{\text{old}} < 10^{-6}$ (near-black), use $s = 1.0$ to avoid division by zero.

**Step 5: Apply scale**

$$
\begin{bmatrix} r'' \\ g'' \\ b'' \end{bmatrix} = \text{clamp}\left( s \cdot \begin{bmatrix} r' \\ g' \\ b' \end{bmatrix}, 0, 1 \right)
$$

### Error Analysis

After applying tone curve:

$$
\text{Error}_{\text{tone}} = \frac{1}{3N} \sum_{i=1}^{N} \sum_{c \in \{r,g,b\}} |(\text{tone RGB})_{i,c} - (\text{target})_{i,c}|
$$

**Typical result:**
```
Mean absolute error (matrix -> target):      0.043256
Mean absolute error (matrix+tone -> target): 0.028134
Improvement (tone):                          0.015122  (35.0% reduction)
```

---

## Complete Workflow

### Full Pipeline Summary

```
┌─────────────────┐
│  Pixel Data CSV │
│ (source, target)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Step 3: Solve Color Matrix  │
│  X·M = Y (SVD least-squares)│
└────────┬────────────────────┘
         │  M (3×3 matrix)
         ▼
┌─────────────────────────────┐
│ Step 4: Apply Matrix to All │
│  matrix_rgb = M · source    │
└────────┬────────────────────┘
         │  matrix_rgb
         ▼
┌─────────────────────────────┐
│Step 5: Compute Luminance    │
│ y_matrix = 0.21R+0.72G+0.07B│
│ y_target = 0.21R+0.72G+0.07B│
└────────┬────────────────────┘
         │  (y_matrix, y_target)
         ▼
┌─────────────────────────────┐
│ Step 6: Fit Tone Curve      │
│  256-bin histogram + smooth │
└────────┬────────────────────┘
         │  tone_curve[256]
         ▼
┌─────────────────────────────┐
│ Step 7: Apply Tone Curve    │
│  Chroma-preserving scaling  │
└────────┬────────────────────┘
         │  tone_rgb
         ▼
┌─────────────────────────────┐
│ Save Results to CSV         │
│  matrix_tone_residual.csv   │
└─────────────────────────────┘
```

### Output Files

**1. `outputs/second_method/matrix_tone_residual.csv`**

Columns:
- `sr, sg, sb`: Source RGB
- `cr, cg, cb`: Target RGB (ground truth)
- `mr, mg, mb`: Matrix-corrected RGB
- `tr, tg, tb`: Tone-corrected RGB
- `y_matrix, y_target`: Luminance values
- `rr, rg, rb`: **Residual** (target - tone)

**Usage:** This CSV is input to the next step (`build_residual_lut.rs`).

**2. `outputs/second_method/tone_curve.csv`**

Tone curve values for visualization and debugging.

---

## Mathematical Properties

### 1. Least-Squares Optimality

The SVD solution minimizes the Frobenius norm:

$$
\lVert\mathbf{X} \mathbf{M} - \mathbf{Y}\rVert_F = \sqrt{\sum_{i,j} ((XM)_{ij} - Y_{ij})^2}
$$

**Theorem:** The SVD solution is the **unique global minimum** (assuming $\mathbf{X}$ has full rank).

### 2. Numerical Stability

**Why SVD over normal equations?**

Normal equations: $\mathbf{X}^T \mathbf{X} \mathbf{M} = \mathbf{X}^T \mathbf{Y}$

**Problem:** Computing $\mathbf{X}^T \mathbf{X}$ can cause:
- **Condition number squaring:** $\kappa(\mathbf{X}^T \mathbf{X}) = \kappa(\mathbf{X})^2$
- **Loss of precision** for ill-conditioned matrices

**SVD advantage:** Directly computes the solution without forming $\mathbf{X}^T \mathbf{X}$.

### 3. Chroma Preservation

**Theorem:** Applying a uniform scale factor $s$ to $(r, g, b)$ preserves hue and saturation.

**Proof:** Convert to HSV:

$$
\begin{aligned}
H &= \arctan\left(\frac{\sqrt{3}(g - b)}{2r - g - b}\right) \quad \text{(hue)} \\
S &= 1 - \frac{3 \min(r, g, b)}{r + g + b} \quad \text{(saturation)} \\
V &= \frac{r + g + b}{3} \quad \text{(value)}
\end{aligned}
$$

After scaling $(r', g', b') = s(r, g, b)$:

$$
\begin{aligned}
H' &= \arctan\left(\frac{\sqrt{3}(sg - sb)}{2sr - sg - sb}\right) = \arctan\left(\frac{\sqrt{3}(g - b)}{2r - g - b}\right) = H \\
S' &= 1 - \frac{3 \min(sr, sg, sb)}{sr + sg + sb} = 1 - \frac{3s\min(r, g, b)}{s(r + g + b)} = S \\
V' &= \frac{sr + sg + sb}{3} = s \cdot \frac{r + g + b}{3} = sV
\end{aligned}
$$

**Conclusion:** Hue and saturation are preserved, only value (brightness) changes. ✓

### 4. Monotonicity

**Property:** The tone curve is monotonically increasing:

$$
f(y_1) \leq f(y_2) \quad \text{if} \quad y_1 \leq y_2
$$

**Importance:** Prevents brightness inversions (brighter input → darker output).

### 5. Smooth Interpolation

Linear interpolation ensures $C^0$ continuity (no jumps) but not $C^1$ continuity (derivatives may have kinks).

For smoother results, consider:
- **Cubic interpolation:** $C^1$ continuity
- **Spline interpolation:** $C^2$ continuity

Trade-off: More complex, slower computation.

---

## Performance Characteristics

### Time Complexity

- **SVD solving:** $O(N \cdot 3^2)$ ≈ $O(N)$ for $N$ training samples
- **Matrix application:** $O(N)$ (9 multiplications per pixel)
- **Tone curve fitting:** $O(N + B)$ where $B = 256$ bins
- **Tone curve application:** $O(N)$ (2 lookups + interpolation per pixel)

**Total:** $O(N)$ = linear in number of training samples

### Space Complexity

- **Matrix:** $3 \times 3 = 9$ floats = **36 bytes**
- **Tone curve:** $256$ floats = **1 KB**
- **Total model size:** ~**1 KB** (extremely compact!)

**Comparison:** First method's 33³ LUT = 35,937 × 3 floats = **431 KB**

### Accuracy

**Mean absolute error:**
- Source → Target: **0.1153**
- Matrix → Target: **0.0433** (62.5% improvement)
- Matrix+Tone → Target: **0.0281** (75.6% improvement)

**Remaining error:** Captured by residual 3D LUT (next step).

---

## Summary

This matrix+tone pipeline provides:
1. **Global color correction** via SVD least-squares (numerically stable)
2. **Brightness correction** via histogram-based tone curve (smooth and monotonic)
3. **Chroma preservation** via proportional RGB scaling (no hue shifts)
4. **Compact representation** (~1 KB model)
5. **Fast application** (linear time complexity)

**Next step:** Use residual 3D LUT to model remaining local variations (see `build_residual_lut.md`).
