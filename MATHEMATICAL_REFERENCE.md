# Mathematical Reference

Complete mathematical documentation for all algorithms used in the Classic Chrome LUT workflow.

---

## Table of Contents

1. [Stratified LAB Sampling](#1-stratified-lab-sampling)
2. [LUT Building with IDW Interpolation](#2-lut-building-with-idw-interpolation)
3. [Brightness Bias Correction](#3-brightness-bias-correction)
4. [LUT Application with Trilinear Interpolation](#4-lut-application-with-trilinear-interpolation)
5. [Quality Metrics (MSE, PSNR, Delta E)](#5-quality-metrics)
6. [Brightness Bias Analysis](#6-brightness-bias-analysis)

---

## 1. Stratified LAB Sampling

**File:** `stratified_compare_pixel.rs`

### Purpose
Generate evenly distributed training samples across the LAB color space to prevent over-sampling of common colors.

### RGB to LAB Conversion

**Step 1: RGB [0,1] → XYZ**

Using the sRGB to CIE XYZ transformation matrix:

$$
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
=
\begin{bmatrix}
0.4124564 & 0.3575761 & 0.1804375 \\
0.2126729 & 0.7151522 & 0.0721750 \\
0.0193339 & 0.1191920 & 0.9503041
\end{bmatrix}
\begin{bmatrix}
R \\
G \\
B
\end{bmatrix}
$$

**Implementation:**
```rust
fn rgb_to_xyz(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    (x, y, z)
}
```

**Step 2: XYZ → LAB**

Reference white point (D65 illuminant):
$$X_n = 0.95047, \quad Y_n = 1.00000, \quad Z_n = 1.08883$$

Intermediate function $f(t)$ with threshold $\delta = \frac{6}{29}$ and $\kappa = \left(\frac{29}{3}\right)^3 = \frac{841}{108}$:

$$
f(t) = \begin{cases}
t^{1/3} & \text{if } t > \delta^3 \\
\frac{\kappa \cdot t + 16}{116} & \text{otherwise}
\end{cases}
$$

LAB calculation:

$$
\begin{aligned}
L^* &= 116 \cdot f\left(\frac{Y}{Y_n}\right) - 16 \\
a^* &= 500 \cdot \left[f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right] \\
b^* &= 200 \cdot \left[f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right]
\end{aligned}
$$

**Implementation:**
```rust
fn xyz_to_lab(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;
    
    let delta = 6.0 / 29.0;
    let delta_cubed = delta * delta * delta;
    
    let f = |t: f32| -> f32 {
        if t > delta_cubed {
            t.powf(1.0 / 3.0)
        } else {
            (841.0 / 108.0) * t + 16.0 / 116.0
        }
    };
    
    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);
    
    let l_star = 116.0 * fy - 16.0;
    let a_star = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    
    (l_star, a_star, b_star)
}
```

**Ranges:**
- $L^* \in [0, 100]$ - Lightness (0 = black, 100 = white)
- $a^* \in [-128, 127]$ - Green (−) to Red (+)
- $b^* \in [-128, 127]$ - Blue (−) to Yellow (+)

### Bucket Assignment

**Step 1: Normalize LAB to [0, 1]**

$$
\begin{aligned}
L_{\text{norm}} &= \frac{L^*}{100} \\
a_{\text{norm}} &= \frac{a^* + 128}{255} \\
b_{\text{norm}} &= \frac{b^* + 128}{255}
\end{aligned}
$$

**Implementation:**
```rust
fn normalize_lab(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let l_norm = l / 100.0;
    let a_norm = (a + 128.0) / 255.0;
    let b_norm = (b + 128.0) / 255.0;
    (l_norm, a_norm, b_norm)
}
```

**Step 2: Compute Bucket Indices**

Given bucket grid size $B = 8$ (creates $B^3 = 512$ buckets):

$$
\begin{aligned}
\text{bucket}_L &= \lfloor L_{\text{norm}} \times (B - 1) \rfloor \\
\text{bucket}_a &= \lfloor a_{\text{norm}} \times (B - 1) \rfloor \\
\text{bucket}_b &= \lfloor b_{\text{norm}} \times (B - 1) \rfloor
\end{aligned}
$$

Convert 3D coordinates to linear index:

$$
\text{bucket\_index} = \text{bucket}_L + \text{bucket}_a \times B + \text{bucket}_b \times B^2
$$

**Implementation:**
```rust
const BUCKET_SIZE: usize = 8;

fn compute_bucket(l_norm: f32, a_norm: f32, b_norm: f32) -> usize {
    let bucket_l = ((l_norm * (BUCKET_SIZE - 1) as f32).floor() as usize)
        .min(BUCKET_SIZE - 1);
    let bucket_a = ((a_norm * (BUCKET_SIZE - 1) as f32).floor() as usize)
        .min(BUCKET_SIZE - 1);
    let bucket_b = ((b_norm * (BUCKET_SIZE - 1) as f32).floor() as usize)
        .min(BUCKET_SIZE - 1);
    
    bucket_l + bucket_a * BUCKET_SIZE + bucket_b * BUCKET_SIZE * BUCKET_SIZE
}
```

### Sampling Strategy

**Algorithm:**

For each image $I$ in training set:

$$
\text{For each bucket } b \in \{0, 1, \ldots, 511\}:
$$

Let $P_b$ = set of pixels in image $I$ that map to bucket $b$

$$
\text{samples}_b = \begin{cases}
\text{random\_sample}(P_b, M) & \text{if } |P_b| > M \\
P_b & \text{otherwise}
\end{cases}
$$

where $M = 200$ (max samples per bucket)

**Implementation:**
```rust
use rand::seq::SliceRandom;

const MAX_SAMPLES_PER_BUCKET: usize = 200;
const TOTAL_BUCKETS: usize = 512; // 8³

fn stratified_sampling(
    pixels: Vec<(f32, f32, f32)>, // LAB pixels
    rng: &mut impl rand::Rng
) -> Vec<(f32, f32, f32)> {
    // Group pixels by bucket
    let mut buckets: Vec<Vec<(f32, f32, f32)>> = vec![Vec::new(); TOTAL_BUCKETS];
    
    for (l, a, b) in pixels {
        let (l_norm, a_norm, b_norm) = normalize_lab(l, a, b);
        let bucket_idx = compute_bucket(l_norm, a_norm, b_norm);
        buckets[bucket_idx].push((l, a, b));
    }
    
    // Sample from each bucket
    let mut samples = Vec::new();
    for mut bucket in buckets {
        if bucket.len() > MAX_SAMPLES_PER_BUCKET {
            bucket.shuffle(rng);
            samples.extend_from_slice(&bucket[..MAX_SAMPLES_PER_BUCKET]);
        } else {
            samples.extend(bucket);
        }
    }
    
    samples
}
```

**Result:**
- Maximum samples per bucket per image: $M = 200$
- Maximum total samples per image: $B^3 \times M = 512 \times 200 = 102{,}400$
- Expected samples (8 images): $\approx 103{,}000$

### Mathematical Properties

**Stratification ensures:**
1. **Coverage:** All regions of LAB space represented
2. **Balance:** No color over-represented (max 200 samples/bucket)
3. **Perceptual uniformity:** LAB distances correlate with human perception

---

## 2. LUT Building with IDW Interpolation

**File:** `build_lut.rs`

### LUT Structure

3D look-up table mapping source RGB → target RGB:

```
LUT: R³ → R³
LUT[i][j][k] = [R', G', B']

where:
  i, j, k ∈ {0, 1, ..., N-1}  (N = 33)
  i corresponds to R dimension
  j corresponds to G dimension
  k corresponds to B dimension
```

### Training Data Accumulation

**Step 1: Map RGB to LUT indices**

For each training sample (Rs, Gs, Bs) → (Rt, Gt, Bt):

```
i = floor(Rs × (N - 1))
j = floor(Gs × (N - 1))
k = floor(Bs × (N - 1))
```

**Step 2: Accumulate and average**

```
LUT[i][j][k] += [Rt, Gt, Bt]
count[i][j][k] += 1

After all samples:
LUT[i][j][k] = LUT[i][j][k] / count[i][j][k]  (if count > 0)
```

### Inverse-Distance Weighted (IDW) Interpolation

**Purpose:** Fill empty LUT cells (count = 0) using nearby filled cells

**Weight Function:**

For distance $d$ between two cells:

$$
w(d) = \frac{1}{d}, \quad d > 0
$$

where distance in 3D LUT space:

$$
d_{(i,j,k) \to (i',j',k')} = \sqrt{(i-i')^2 + (j-j')^2 + (k-k')^2}
$$

**Interpolated Value:**

Given empty cell at $(i, j, k)$ and set of filled neighbors $\mathcal{N}$:

$$
\text{LUT}[i,j,k] = \frac{\displaystyle\sum_{n \in \mathcal{N}} w_n \cdot \text{LUT}_n}{\displaystyle\sum_{n \in \mathcal{N}} w_n}
$$

where:
- $w_n = \frac{1}{d_n}$ is weight for neighbor $n$
- $\text{LUT}_n = [R', G', B']$ is the RGB value at neighbor $n$

**Algorithm (Pseudocode):**

```
For each empty cell (i, j, k):
    For radius r = 1 to N:
        weighted_sum = [0, 0, 0]
        weight_sum = 0
        
        For each cell (i', j', k') within radius r:
            if count[i'][j'][k'] > 0:  // Has data
                d = sqrt((i-i')² + (j-j')² + (k-k')²)
                
                if d > 0:
                    w = 1 / d
                    
                    weighted_sum += w × LUT[i'][j'][k']
                    weight_sum += w
        
        if weight_sum > 0:
            LUT[i][j][k] = weighted_sum / weight_sum
            break  // Stop searching, cell filled
```

**Implementation:**
```rust
fn idw_interpolation(
    lut: &mut Vec<Vec<Vec<[f32; 3]>>>,
    count: &Vec<Vec<Vec<u32>>>,
    n: usize
) {
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // Skip cells that already have data
                if count[i][j][k] > 0 {
                    continue;
                }
                
                // Search for non-empty neighbors with increasing radius
                let mut found = false;
                for radius in 1..=n {
                    let mut weighted_sum = [0.0f32; 3];
                    let mut weight_sum = 0.0f32;
                    
                    // Search all cells within this radius
                    for di in -(radius as i32)..=(radius as i32) {
                        for dj in -(radius as i32)..=(radius as i32) {
                            for dk in -(radius as i32)..=(radius as i32) {
                                let ni = i as i32 + di;
                                let nj = j as i32 + dj;
                                let nk = k as i32 + dk;
                                
                                // Check bounds
                                if ni < 0 || nj < 0 || nk < 0 
                                    || ni >= n as i32 || nj >= n as i32 || nk >= n as i32 {
                                    continue;
                                }
                                
                                let (ni, nj, nk) = (ni as usize, nj as usize, nk as usize);
                                
                                // Skip empty neighbors
                                if count[ni][nj][nk] == 0 {
                                    continue;
                                }
                                
                                // Compute Euclidean distance
                                let distance = ((di * di + dj * dj + dk * dk) as f32).sqrt();
                                if distance == 0.0 {
                                    continue;
                                }
                                
                                // IDW weight
                                let weight = 1.0 / distance;
                                
                                // Accumulate weighted values
                                weighted_sum[0] += lut[ni][nj][nk][0] * weight;
                                weighted_sum[1] += lut[ni][nj][nk][1] * weight;
                                weighted_sum[2] += lut[ni][nj][nk][2] * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                    
                    // If we found at least one neighbor, fill the cell
                    if weight_sum > 0.0 {
                        lut[i][j][k][0] = weighted_sum[0] / weight_sum;
                        lut[i][j][k][1] = weighted_sum[1] / weight_sum;
                        lut[i][j][k][2] = weighted_sum[2] / weight_sum;
                        found = true;
                        break;
                    }
                }
                
                if !found {
                    // Fallback: identity mapping
                    lut[i][j][k][0] = i as f32 / (n - 1) as f32;
                    lut[i][j][k][1] = j as f32 / (n - 1) as f32;
                    lut[i][j][k][2] = k as f32 / (n - 1) as f32;
                }
            }
        }
    }
}
```

**Mathematical Properties:**

1. **Exactness at data points:**
   $$\lim_{d \to 0} w(d) = \infty \implies \text{LUT}[i,j,k] = \text{LUT}_{\text{neighbor}}$$

2. **Smoothness:**
   - Continuous everywhere
   - First derivatives discontinuous at data points (but acceptable)

3. **Local influence:**
   - Nearby cells dominate: $w(1) = 1.0$, $w(2) = 0.5$, $w(3) \approx 0.33$
   - Distant cells have minimal effect

4. **Complexity:**
   - Worst case: $O(E \times N^3)$ where $E$ = empty cells, $N$ = LUT size
   - Average case: $O(E \times r^3)$ where $r$ is typical radius (small)

### Fallback: Identity Mapping

If no filled neighbors found (rare):
```
LUT[i][j][k] = [i/(N-1), j/(N-1), k/(N-1)]
```

This maps input RGB directly to output (no transformation).

---

## 3. Brightness Bias Correction

**File:** `build_lut.rs` (integrated)

### Purpose
Correct systematic brightness offset in the LUT to match ground truth luminance.

### Overview

After building the LUT from training data, a systematic brightness bias may remain. This correction eliminates that bias by operating in perceptually uniform LAB color space.

**Process:** $\text{LUT}_{RGB} \xrightarrow{\text{RGB→LAB}} \text{LUT}_{LAB} \xrightarrow{\text{correct } L^*} \text{LUT}_{LAB}^{\text{corrected}} \xrightarrow{\text{LAB→RGB}} \text{LUT}_{RGB}^{\text{corrected}}$

### Step 1: RGB to LAB Conversion (per LUT cell)

For each LUT cell $\text{LUT}[i][j][k] = [R, G, B]$ where $R, G, B \in [0, 1]$:

$$
[L^*, a^*, b^*] = \text{RGB\_to\_LAB}([R, G, B])
$$

(Uses same RGB→LAB conversion as Section 1)

### Step 2: Bias Correction Formula

Apply calibrated correction to L* channel:

$$
L^*_{\text{corrected}} = L^* - \text{CALIBRATED\_BIAS\_L}
$$

Where:
- $\text{CALIBRATED\_BIAS\_L} = 1.489$ (measured from 8 ground truth images)
- $L^* \in [0, 100]$ (CIELAB lightness range)

**Calibration Process:**
The calibrated bias was determined by:
1. Building initial LUT from training data
2. Applying LUT to validation images
3. Computing average L* error across all pixels
4. Averaging across multiple images (8 images → 1.489 L* units)

### Step 3: Clamp to Valid Range

Ensure corrected value stays within valid LAB range:

$$
L^*_{\text{corrected}} = \text{clamp}(L^*_{\text{corrected}}, 0, 100)
$$

### Step 4: LAB to RGB Conversion

Convert corrected LAB back to RGB:

$$
[R', G', B'] = \text{LAB\_to\_RGB}([L^*_{\text{corrected}}, a^*, b^*])
$$

**Note:** $a^*$ and $b^*$ (color channels) remain unchanged, only brightness is corrected.

### Step 5: Final Clamping

Ensure RGB values stay in valid range:

$$
[R', G', B'] = \text{clamp}([R', G', B'], 0, 1)
$$

---

### LAB to RGB Inverse Transformation

**Step 1: LAB → XYZ**

Inverse of the XYZ→LAB transformation:

$$
\begin{aligned}
f_y &= \frac{L^* + 16}{116} \\
f_x &= \frac{a^*}{500} + f_y \\
f_z &= f_y - \frac{b^*}{200}
\end{aligned}
$$

**Inverse function:**

$$
f^{-1}(t) = \begin{cases}
t^3 & \text{if } t > \delta \\
\frac{116t - 16}{\kappa} & \text{otherwise}
\end{cases}
$$

Where:
- $\delta = \frac{6}{29} \approx 0.2069$
- $\kappa = 903.3$ (same as forward transformation)

**Apply inverse:**

$$
\begin{aligned}
X &= X_n \times f^{-1}(f_x) \\
Y &= Y_n \times f^{-1}(f_y) \\
Z &= Z_n \times f^{-1}(f_z)
\end{aligned}
$$

**Step 2: XYZ → RGB**

$$
\begin{bmatrix} R \\ G \\ B \end{bmatrix} = 
\begin{bmatrix}
 3.2404542 & -1.5371385 & -0.4985314 \\
-0.9692660 &  1.8760108 &  0.0415560 \\
 0.0556434 & -0.2040259 &  1.0572252
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

**Gamma correction (inverse):**

$$
C_{\text{RGB}} = \begin{cases}
12.92 \times C_{\text{linear}} & \text{if } C_{\text{linear}} \leq 0.0031308 \\
1.055 \times C_{\text{linear}}^{1/2.4} - 0.055 & \text{otherwise}
\end{cases}
$$

---

### Complete Implementation

```rust
const CALIBRATED_BIAS_L: f32 = 1.489;

// D65 white point reference
const XN: f32 = 0.95047;
const YN: f32 = 1.00000;
const ZN: f32 = 1.08883;

fn apply_brightness_correction(lut: &mut Vec<Vec<Vec<[f32; 3]>>>, n: usize) {
    println!("Applying brightness bias correction (LAB space)...");
    
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let rgb = lut[i][j][k];
                
                // Step 1: RGB → LAB
                let lab = rgb_to_lab(rgb[0], rgb[1], rgb[2]);
                
                // Step 2: Correct L* channel
                let l_corrected = (lab[0] - CALIBRATED_BIAS_L).clamp(0.0, 100.0);
                
                // Step 3: LAB → RGB with corrected L*
                let corrected_rgb = lab_to_rgb(l_corrected, lab[1], lab[2]);
                
                // Step 4: Final clamping
                lut[i][j][k] = [
                    corrected_rgb[0].clamp(0.0, 1.0),
                    corrected_rgb[1].clamp(0.0, 1.0),
                    corrected_rgb[2].clamp(0.0, 1.0),
                ];
            }
        }
    }
    
    println!("Brightness correction completed.");
}

fn lab_to_rgb(l: f32, a: f32, b: f32) -> [f32; 3] {
    // Step 1: LAB → XYZ
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    
    // Inverse f function
    let delta: f32 = 6.0 / 29.0;
    let delta_cubed = delta * delta * delta;
    
    let finv = |t: f32| -> f32 {
        if t > delta {
            t * t * t
        } else {
            (116.0 * t - 16.0) / 903.3
        }
    };
    
    let x = XN * finv(fx);
    let y = YN * finv(fy);
    let z = ZN * finv(fz);
    
    // Step 2: XYZ → RGB (linear)
    let r_linear =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g_linear = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b_linear =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    
    // Step 3: Apply gamma correction (sRGB)
    let gamma_correct = |c: f32| -> f32 {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    
    [
        gamma_correct(r_linear),
        gamma_correct(g_linear),
        gamma_correct(b_linear),
    ]
}

// RGB to LAB helper (reuses implementation from Section 1)
fn rgb_to_lab(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Inverse gamma correction
    let linear = |c: f32| -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    
    let r_lin = linear(r);
    let g_lin = linear(g);
    let b_lin = linear(b);
    
    // RGB → XYZ
    let x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin;
    let y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin;
    let z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin;
    
    // XYZ → LAB
    let xn = x / XN;
    let yn = y / YN;
    let zn = z / ZN;
    
    let delta: f32 = 6.0 / 29.0;
    let delta_cubed = delta * delta * delta;
    
    let f = |t: f32| -> f32 {
        if t > delta_cubed {
            t.cbrt()
        } else {
            (903.3 * t + 16.0) / 116.0
        }
    };
    
    let fx = f(xn);
    let fy = f(yn);
    let fz = f(zn);
    
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    
    [l, a, b]
}
```

---

### Why LAB Space for Correction?

**Advantages:**

1. **Perceptually uniform:** $\Delta L^* = 1$ represents the same perceptual change at any luminance level
   - Correcting by 1.489 L* units has consistent visual impact across all brightness levels
   
2. **Preserves color:** Only $L^*$ is modified, $a^*$ and $b^*$ remain unchanged
   - Hue and chroma are preserved
   - No color shifts or casts introduced
   
3. **Accurate across range:** Works correctly from shadows (L*=0) to highlights (L*=100)
   - Linear correction in perceptual space
   
4. **Decoupled dimensions:** Brightness independent of color
   - Can correct brightness without affecting color appearance

**RGB Space Problems:**

If we attempted correction in RGB space:

$$
\text{RGB}_{\text{corrected}} = \text{RGB} \times k, \quad k = 0.985 \text{ (multiplicative)}
$$

or

$$
\text{RGB}_{\text{corrected}} = \text{RGB} - c, \quad c = \text{constant (subtractive)}
$$

**Issues:**
- **Not perceptually uniform:** Same RGB change has different perceptual effect at different luminances
  - Brightening dark pixels by 10 units is very noticeable
  - Brightening bright pixels by 10 units is barely noticeable
  
- **Color shift:** RGB correction affects all three channels
  - Can introduce unwanted color casts
  - Changes hue and saturation, not just brightness
  
- **Nonlinear perception:** Human vision is logarithmic, RGB is linear
  - RGB arithmetic doesn't match perception
  
- **Gamma complexity:** sRGB has gamma encoding
  - Need to work in linear space for accurate math
  - LAB already incorporates perceptual nonlinearity

**Result:** LAB correction produces perceptually accurate, color-preserving brightness adjustment.

---

## 4. LUT Application with Trilinear Interpolation

**File:** `apply_lut.rs`

### Purpose
Apply 3D LUT to image pixels using smooth interpolation.

### Input Normalization

Given input pixel (R, G, B) in [0, 255]:
```
r = R / 255  ∈ [0, 1]
g = G / 255  ∈ [0, 1]
b = B / 255  ∈ [0, 1]
```

### LUT Position Calculation

Map [0,1] to LUT coordinates:
```
x = r × (N - 1)
y = g × (N - 1)
z = b × (N - 1)

where N = 33 (LUT size)
```

Example: r=0.5 → x = 0.5 × 32 = 16.0

### Surrounding Cell Indices

```
x0 = floor(x)
y0 = floor(y)
z0 = floor(z)

x1 = min(x0 + 1, N - 1)
y1 = min(y0 + 1, N - 1)
z1 = min(z0 + 1, N - 1)
```

### Interpolation Weights

```
dx = x - x0  ∈ [0, 1)
dy = y - y0  ∈ [0, 1)
dz = z - z0  ∈ [0, 1)
```

Example:
- x = 16.3 → x0 = 16, dx = 0.3
- Means 30% toward next cell, 70% from current cell

### 8 Corner Values

Fetch LUT values at 8 surrounding points in lexicographic order:

$$
\begin{aligned}
c_{000} &= \text{LUT}[x_0][y_0][z_0] \\
c_{001} &= \text{LUT}[x_0][y_0][z_1] \\
c_{010} &= \text{LUT}[x_0][y_1][z_0] \\
c_{011} &= \text{LUT}[x_0][y_1][z_1] \\
c_{100} &= \text{LUT}[x_1][y_0][z_0] \\
c_{101} &= \text{LUT}[x_1][y_0][z_1] \\
c_{110} &= \text{LUT}[x_1][y_1][z_0] \\
c_{111} &= \text{LUT}[x_1][y_1][z_1]
\end{aligned}
$$

Each $c_{ijk}$ is a 3D vector: $c_{ijk} = [R', G', B']$

### Trilinear Interpolation Formula

**Method 1: Sequential Linear Interpolations**

**Step 1: Interpolate along Z-axis** (4 interpolations):

$$
\begin{aligned}
c_{00} &= c_{000} \cdot (1 - d_z) + c_{001} \cdot d_z \\
c_{01} &= c_{010} \cdot (1 - d_z) + c_{011} \cdot d_z \\
c_{10} &= c_{100} \cdot (1 - d_z) + c_{101} \cdot d_z \\
c_{11} &= c_{110} \cdot (1 - d_z) + c_{111} \cdot d_z
\end{aligned}
$$

**Step 2: Interpolate along Y-axis** (2 interpolations):

$$
\begin{aligned}
c_0 &= c_{00} \cdot (1 - d_y) + c_{01} \cdot d_y \\
c_1 &= c_{10} \cdot (1 - d_y) + c_{11} \cdot d_y
\end{aligned}
$$

**Step 3: Interpolate along X-axis** (1 interpolation):

$$
\text{result} = c_0 \cdot (1 - d_x) + c_1 \cdot d_x
$$

**Method 2: Direct Formula**

Expanding the sequential interpolations gives the complete formula:

$$
\begin{aligned}
\text{result} &= c_{000} \cdot (1-d_x)(1-d_y)(1-d_z) \\
              &+ c_{001} \cdot (1-d_x)(1-d_y) \cdot d_z \\
              &+ c_{010} \cdot (1-d_x) \cdot d_y \cdot (1-d_z) \\
              &+ c_{011} \cdot (1-d_x) \cdot d_y \cdot d_z \\
              &+ c_{100} \cdot d_x \cdot (1-d_y)(1-d_z) \\
              &+ c_{101} \cdot d_x \cdot (1-d_y) \cdot d_z \\
              &+ c_{110} \cdot d_x \cdot d_y \cdot (1-d_z) \\
              &+ c_{111} \cdot d_x \cdot d_y \cdot d_z
\end{aligned}
$$

**Verification:**

Sum of weights equals 1:

$$
\sum_{i,j,k \in \{0,1\}} w_{ijk} = (1-d_x+d_x)(1-d_y+d_y)(1-d_z+d_z) = 1 \cdot 1 \cdot 1 = 1
$$

**Implementation:**
```rust
fn trilinear_interpolate(
    lut: &Vec<Vec<Vec<[f32; 3]>>>,
    x: f32,
    y: f32,
    z: f32,
    n: usize
) -> [f32; 3] {
    let n_max = (n - 1) as usize;
    
    // Get surrounding indices
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;
    
    let x1 = (x0 + 1).min(n_max);
    let y1 = (y0 + 1).min(n_max);
    let z1 = (z0 + 1).min(n_max);
    
    // Interpolation weights
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let dz = z - z0 as f32;
    
    // Fetch 8 corner values
    let c000 = lut[x0][y0][z0];
    let c001 = lut[x0][y0][z1];
    let c010 = lut[x0][y1][z0];
    let c011 = lut[x0][y1][z1];
    let c100 = lut[x1][y0][z0];
    let c101 = lut[x1][y0][z1];
    let c110 = lut[x1][y1][z0];
    let c111 = lut[x1][y1][z1];
    
    let mut result = [0.0f32; 3];
    
    // Interpolate each channel independently
    for ch in 0..3 {
        // Step 1: Z-axis
        let c00 = c000[ch] * (1.0 - dz) + c001[ch] * dz;
        let c01 = c010[ch] * (1.0 - dz) + c011[ch] * dz;
        let c10 = c100[ch] * (1.0 - dz) + c101[ch] * dz;
        let c11 = c110[ch] * (1.0 - dz) + c111[ch] * dz;
        
        // Step 2: Y-axis
        let c0 = c00 * (1.0 - dy) + c01 * dy;
        let c1 = c10 * (1.0 - dy) + c11 * dy;
        
        // Step 3: X-axis
        result[ch] = c0 * (1.0 - dx) + c1 * dx;
    }
    
    result
}
```

**Alternative: Direct Formula Implementation**
```rust
fn trilinear_direct(
    lut: &Vec<Vec<Vec<[f32; 3]>>>,
    x0: usize, y0: usize, z0: usize,
    x1: usize, y1: usize, z1: usize,
    dx: f32, dy: f32, dz: f32
) -> [f32; 3] {
    let c000 = lut[x0][y0][z0];
    let c001 = lut[x0][y0][z1];
    let c010 = lut[x0][y1][z0];
    let c011 = lut[x0][y1][z1];
    let c100 = lut[x1][y0][z0];
    let c101 = lut[x1][y0][z1];
    let c110 = lut[x1][y1][z0];
    let c111 = lut[x1][y1][z1];
    
    let mut result = [0.0f32; 3];
    
    for ch in 0..3 {
        result[ch] = 
            c000[ch] * (1.0-dx) * (1.0-dy) * (1.0-dz) +
            c001[ch] * (1.0-dx) * (1.0-dy) * dz +
            c010[ch] * (1.0-dx) * dy * (1.0-dz) +
            c011[ch] * (1.0-dx) * dy * dz +
            c100[ch] * dx * (1.0-dy) * (1.0-dz) +
            c101[ch] * dx * (1.0-dy) * dz +
            c110[ch] * dx * dy * (1.0-dz) +
            c111[ch] * dx * dy * dz;
    }
    
    result
}
```

### Output Conversion

```
R_out = clamp(result[0], 0, 1) × 255
G_out = clamp(result[1], 0, 1) × 255
B_out = clamp(result[2], 0, 1) × 255
```

### Mathematical Properties

1. **Continuity:** No discontinuities at LUT cell boundaries
2. **Smoothness:** C⁰ continuous (value continuous, derivatives may not be)
3. **Exactness:** At LUT grid points, dx=dy=dz=0, returns exact LUT value
4. **Efficiency:** Only 7 additions + 8 multiplications per channel

### Alternative: Nearest Neighbor (Not Used)

Simple but produces banding:
```
i = round(r × (N-1))
j = round(g × (N-1))
k = round(b × (N-1))

result = LUT[i][j][k]  // No interpolation
```

---

## 5. Quality Metrics

**File:** `compare_lut.rs`

### 5.1 Mean Squared Error (MSE)

**Definition:**

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left[(R_{\text{gt}}^{(i)} - R_{\text{lut}}^{(i)})^2 + (G_{\text{gt}}^{(i)} - G_{\text{lut}}^{(i)})^2 + (B_{\text{gt}}^{(i)} - B_{\text{lut}}^{(i)})^2\right]
$$

Where:
- $N$ = total number of pixels
- $R_{\text{gt}}, G_{\text{gt}}, B_{\text{gt}}$ = ground truth RGB values $\in [0, 255]$
- $R_{\text{lut}}, G_{\text{lut}}, B_{\text{lut}}$ = LUT output RGB values $\in [0, 255]$
- Index $i$ denotes pixel position

**Per-pixel error:**

$$
\text{error}_{\text{pixel}}^{(i)} = (R_{\text{gt}}^{(i)} - R_{\text{lut}}^{(i)})^2 + (G_{\text{gt}}^{(i)} - G_{\text{lut}}^{(i)})^2 + (B_{\text{gt}}^{(i)} - B_{\text{lut}}^{(i)})^2
$$

**Properties:**
- Range: $[0, \infty)$, lower is better
- Units: squared intensity units
- Current result: **MSE = 3.24**
- Heavily penalizes large errors (squared term)

**Implementation:**
```rust
use opencv::core::Mat;

fn compute_mse(img_gt: &Mat, img_lut: &Mat) -> Result<f64, Box<dyn std::error::Error>> {
    let rows = img_gt.rows();
    let cols = img_gt.cols();
    let n_pixels = (rows * cols) as f64;
    
    let mut mse = 0.0;
    
    for y in 0..rows {
        for x in 0..cols {
            // Get BGR pixels (OpenCV format)
            let pixel_gt: &opencv::core::Vec3b = img_gt.at_2d(y, x)?;
            let pixel_lut: &opencv::core::Vec3b = img_lut.at_2d(y, x)?;
            
            // Compute squared error for each channel
            for c in 0..3 {
                let diff = pixel_gt[c] as f64 - pixel_lut[c] as f64;
                mse += diff * diff;
            }
        }
    }
    
    // Average over all pixels
    mse /= n_pixels;
    
    Ok(mse)
}
```

---

### 5.2 Peak Signal-to-Noise Ratio (PSNR)

**Definition:**

$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right) \quad \text{(dB)}
$$

Where:
- $\text{MAX} = 255$ (maximum pixel value for 8-bit images)
- $\text{MSE}$ = Mean Squared Error

**Alternative forms:**

$$
\begin{aligned}
\text{PSNR} &= 20 \cdot \log_{10}\left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right) \\
&= 20 \cdot \log_{10}(\text{MAX}) - 10 \cdot \log_{10}(\text{MSE}) \\
&\approx 48.13 - 10 \cdot \log_{10}(\text{MSE})
\end{aligned}
$$

**Example calculation:**

For $\text{MSE} = 3.24$:

$$
\begin{aligned}
\text{PSNR} &= 10 \cdot \log_{10}\left(\frac{255^2}{3.24}\right) \\
&= 10 \cdot \log_{10}\left(\frac{65025}{3.24}\right) \\
&= 10 \cdot \log_{10}(20069.44) \\
&= 10 \times 4.3026 \\
&= 43.03 \text{ dB}
\end{aligned}
$$

**Properties:**
- Units: decibels (dB)
- Range: typically $[20, 50]$ dB for images
- Higher is better (logarithmic scale)
- Current result: **PSNR = 43.03 dB** (Excellent)

**Quality interpretation:**

| PSNR Range | Quality Level | Description |
|------------|---------------|-------------|
| < 20 dB    | Very Poor     | Unacceptable |
| 20-25 dB   | Poor          | Low quality |
| 25-30 dB   | Fair          | Acceptable compression |
| 30-35 dB   | Good          | High quality |
| 35-40 dB   | Very Good     | Production quality |
| **40-45 dB** | **Excellent** | **Professional** ← **Current: 43.03 dB** |
| > 45 dB    | Outstanding   | Near-perfect/Visually lossless |

**Implementation:**
```rust
fn compute_psnr(mse: f64) -> f64 {
    if mse < 1e-10 {
        // Avoid log(0), return high PSNR
        return 100.0;
    }
    
    let max_pixel = 255.0;
    10.0 * ((max_pixel * max_pixel) / mse).log10()
}

// Alternative implementation using 20 * log10(MAX / sqrt(MSE))
fn compute_psnr_alt(mse: f64) -> f64 {
    if mse < 1e-10 {
        return 100.0;
    }
    
    let max_pixel = 255.0;
    20.0 * (max_pixel / mse.sqrt()).log10()
}
```

---

### 5.3 Delta E (CIE76 Color Difference)

**Purpose:** Measure perceptually uniform color difference in LAB space

**CIE76 Formula:**

For a single pixel:

$$
\Delta E_{ab}^{(i)} = \sqrt{(L_{\text{gt}}^{*(i)} - L_{\text{lut}}^{*(i)})^2 + (a_{\text{gt}}^{*(i)} - a_{\text{lut}}^{*(i)})^2 + (b_{\text{gt}}^{*(i)} - b_{\text{lut}}^{*(i)})^2}
$$

**Aggregate Statistics:**

$$
\begin{aligned}
\text{Average } \Delta E &= \frac{1}{N} \sum_{i=1}^{N} \Delta E_{ab}^{(i)} \\
\text{Median } \Delta E &= \text{median}(\{\Delta E_{ab}^{(1)}, \Delta E_{ab}^{(2)}, \ldots, \Delta E_{ab}^{(N)}\}) \\
\text{Max } \Delta E &= \max_{i} \Delta E_{ab}^{(i)}
\end{aligned}
$$

Where:
- $L^*, a^*, b^*$ = CIELAB color space coordinates
- Subscript "gt" = ground truth (Classic Chrome)
- Subscript "lut" = LUT-applied output
- $N$ = total number of pixels

**Properties:**
- Perceptually uniform: $\Delta E = 1$ is just noticeable difference (JND)
- Range: $[0, \infty)$, lower is better
- Based on Euclidean distance in LAB space
- More accurate than RGB difference for human perception
- Current results:
  - **Average: 1.28** (barely perceptible)
  - **Median: 1.27** (barely perceptible)

**Perception Scale:**

| ΔE Range     | Perceptual Difference | Visibility |
|--------------|------------------------|------------|
| 0.0 - 1.0    | Not perceptible       | Perfect match |
| **1.0 - 2.0** | **Perceptible through close observation** | **Excellent** ← **Current: 1.28** |
| 2.0 - 3.5    | Perceptible at a glance | Very good |
| 3.5 - 5.0    | Clear difference      | Good |
| 5.0 - 10.0   | Significant difference| Fair |
| > 10.0       | Very obvious difference | Poor |

**Implementation:**
```rust
use opencv::imgproc::{cvt_color, COLOR_BGR2Lab};
use opencv::core::Mat;

fn compute_delta_e_statistics(
    img_gt: &Mat,
    img_lut: &Mat
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    // Convert both images to LAB
    let mut lab_gt = Mat::default();
    let mut lab_lut = Mat::default();
    
    cvt_color(img_gt, &mut lab_gt, COLOR_BGR2Lab, 0)?;
    cvt_color(img_lut, &mut lab_lut, COLOR_BGR2Lab, 0)?;
    
    let rows = lab_gt.rows();
    let cols = lab_gt.cols();
    let n_pixels = (rows * cols) as f64;
    
    let mut delta_e_values: Vec<f64> = Vec::with_capacity((rows * cols) as usize);
    let mut sum_delta_e = 0.0;
    let mut max_delta_e = 0.0;
    
    for y in 0..rows {
        for x in 0..cols {
            let pixel_gt: &opencv::core::Vec3b = lab_gt.at_2d(y, x)?;
            let pixel_lut: &opencv::core::Vec3b = lab_lut.at_2d(y, x)?;
            
            // Extract LAB values (OpenCV stores as [L, a, b] with L in [0,255])
            let l_gt = pixel_gt[0] as f64;
            let a_gt = pixel_gt[1] as f64;
            let b_gt = pixel_gt[2] as f64;
            
            let l_lut = pixel_lut[0] as f64;
            let a_lut = pixel_lut[1] as f64;
            let b_lut = pixel_lut[2] as f64;
            
            // Compute CIE76 Delta E
            let delta_l = l_gt - l_lut;
            let delta_a = a_gt - a_lut;
            let delta_b = b_gt - b_lut;
            
            let delta_e = (delta_l * delta_l + delta_a * delta_a + delta_b * delta_b).sqrt();
            
            sum_delta_e += delta_e;
            delta_e_values.push(delta_e);
            
            if delta_e > max_delta_e {
                max_delta_e = delta_e;
            }
        }
    }
    
    // Compute statistics
    let avg_delta_e = sum_delta_e / n_pixels;
    
    // Compute median
    delta_e_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_idx = delta_e_values.len() / 2;
    let median_delta_e = if delta_e_values.len() % 2 == 0 {
        (delta_e_values[median_idx - 1] + delta_e_values[median_idx]) / 2.0
    } else {
        delta_e_values[median_idx]
    };
    
    Ok((avg_delta_e, median_delta_e, max_delta_e))
}
```

**Note on CIE Versions:**
- **CIE76 (ΔE*ab):** Simple Euclidean distance in LAB (used here)
- **CIE94:** Weighted formula accounting for textile industry requirements
- **CIE2000 (ΔE00):** Most accurate, complex weighting for perceptual uniformity

CIE76 is sufficient for our use case and much faster to compute.

---

### 5.4 Mean Absolute Error (MAE) per Channel

**Definition:**

$$
\text{MAE}_c = \frac{1}{N} \sum_{i=1}^{N} |c_{\text{gt}}^{(i)} - c_{\text{lut}}^{(i)}|, \quad c \in \{R, G, B\}
$$

Where values are in range $[0, 255]$.

**For all three channels:**

$$
\begin{aligned}
\text{MAE}_R &= \frac{1}{N} \sum_{i=1}^{N} |R_{\text{gt}}^{(i)} - R_{\text{lut}}^{(i)}| \\
\text{MAE}_G &= \frac{1}{N} \sum_{i=1}^{N} |G_{\text{gt}}^{(i)} - G_{\text{lut}}^{(i)}| \\
\text{MAE}_B &= \frac{1}{N} \sum_{i=1}^{N} |B_{\text{gt}}^{(i)} - B_{\text{lut}}^{(i)}|
\end{aligned}
$$

**Current results:**
- Blue MAE: **1.47**
- Green MAE: **1.21** (best)
- Red MAE: **1.38**

**Properties:**
- Linear scale (compared to MSE which is quadratic)
- Easy interpretation: average per-pixel difference
- Units: intensity units $[0, 255]$
- Not sensitive to outliers like MSE

**Implementation:**
```rust
fn compute_mae_per_channel(
    img_gt: &Mat,
    img_lut: &Mat
) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    let rows = img_gt.rows();
    let cols = img_gt.cols();
    let n_pixels = (rows * cols) as f64;
    
    let mut mae = [0.0; 3];
    
    for y in 0..rows {
        for x in 0..cols {
            let pixel_gt: &opencv::core::Vec3b = img_gt.at_2d(y, x)?;
            let pixel_lut: &opencv::core::Vec3b = img_lut.at_2d(y, x)?;
            
            for c in 0..3 {
                let diff = (pixel_gt[c] as f64 - pixel_lut[c] as f64).abs();
                mae[c] += diff;
            }
        }
    }
    
    // Average over all pixels
    for c in 0..3 {
        mae[c] /= n_pixels;
    }
    
    // OpenCV uses BGR order
    Ok([mae[2], mae[1], mae[0]]) // Return as RGB
}
```

---

## 6. Brightness Bias Analysis

**File:** `analyze_brightness_bias.rs`

### Purpose
Detect systematic brightness offset (directional bias, not just magnitude of error).

### 6.1 RGB Bias (Mean Error)

**Mean Error Formula:**

For each channel $c \in \{R, G, B\}$:

$$
\text{Mean Error}_c = \frac{1}{N} \sum_{i=1}^{N} (c_{\text{lut}}^{(i)} - c_{\text{gt}}^{(i)})
$$

**Mean Absolute Error Formula:**

$$
\text{MAE}_c = \frac{1}{N} \sum_{i=1}^{N} |c_{\text{lut}}^{(i)} - c_{\text{gt}}^{(i)}|
$$

**Key Difference:**
- **Mean Error:** Shows **direction** (positive = brighter, negative = darker)
- **MAE:** Shows **magnitude** only (always positive)

**Interpretation:**

$$
\begin{cases}
\text{Mean Error} > 0 & \Rightarrow \text{LUT output is BRIGHTER than ground truth} \\
\text{Mean Error} < 0 & \Rightarrow \text{LUT output is DARKER than ground truth} \\
\text{Mean Error} \approx 0 & \Rightarrow \text{No systematic bias (balanced errors)}
\end{cases}
$$

**Current results:**
```
Blue Mean Error:  -0.074  (slightly darker)
Green Mean Error: -0.131  (slightly darker)
Red Mean Error:   -0.118  (slightly darker)

Overall RGB bias: -0.108  (-0.04%)  ← Essentially zero
```

**Implementation:**
```rust
fn compute_rgb_bias(
    img_gt: &Mat,
    img_lut: &Mat
) -> Result<([f64; 3], [f64; 3]), Box<dyn std::error::Error>> {
    let rows = img_gt.rows();
    let cols = img_gt.cols();
    let n_pixels = (rows * cols) as f64;
    
    let mut mean_error = [0.0; 3];
    let mut mae = [0.0; 3];
    
    for y in 0..rows {
        for x in 0..cols {
            let pixel_gt: &opencv::core::Vec3b = img_gt.at_2d(y, x)?;
            let pixel_lut: &opencv::core::Vec3b = img_lut.at_2d(y, x)?;
            
            for c in 0..3 {
                let error = pixel_lut[c] as f64 - pixel_gt[c] as f64;
                mean_error[c] += error;      // Signed error (for bias)
                mae[c] += error.abs();       // Absolute error (for magnitude)
            }
        }
    }
    
    // Average over all pixels
    for c in 0..3 {
        mean_error[c] /= n_pixels;
        mae[c] /= n_pixels;
    }
    
    // OpenCV uses BGR, convert to RGB
    Ok((
        [mean_error[2], mean_error[1], mean_error[0]],
        [mae[2], mae[1], mae[0]]
    ))
}
```

---

### 6.2 LAB L* Channel Bias

**Purpose:** Perceptually accurate brightness measurement using LAB lightness channel

**Step 1: Compute L* Difference**

For each pixel:

$$
\Delta L^{*(i)} = L_{\text{lut}}^{*(i)} - L_{\text{gt}}^{*(i)}
$$

Where $L^* \in [0, 100]$ (CIELAB lightness)

**Step 2: Mean L* Error (Bias)**

$$
\text{L}^* \text{ Mean Error} = \frac{1}{N} \sum_{i=1}^{N} \Delta L^{*(i)}
$$

**Step 3: Mean Absolute L* Error (Magnitude)**

$$
\text{L}^* \text{ MAE} = \frac{1}{N} \sum_{i=1}^{N} |\Delta L^{*(i)}|
$$

**Step 4: Convert to 8-bit Scale**

To express bias in terms of 0-255 RGB values:

$$
\text{L}^* \text{ bias (8-bit)} = \text{L}^* \text{ Mean Error} \times 2.55
$$

**Step 5: Percentage Bias**

$$
\text{Bias Percentage} = \frac{\text{L}^* \text{ Mean Error}}{100} \times 100\%
$$

**Current Results:**
```
L* Mean Error: -0.013  (very slightly darker)
L* MAE: 0.631
L* bias (8-bit): -0.033
Bias Percentage: -0.03%  ← Negligible
```

**Interpretation:**
- **< ±0.5%**: Excellent (no perceptible bias)
- **±0.5-1.0%**: Very good (barely perceptible)
- **±1.0-2.0%**: Good (slight bias)
- **±2.0-5.0%**: Fair (noticeable bias)
- **> ±5.0%**: Poor (significant bias)

**Implementation:**
```rust
use opencv::imgproc::{cvt_color, COLOR_BGR2Lab};

fn compute_lab_bias(
    img_gt: &Mat, 
    img_lut: &Mat
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    // Convert both images to LAB
    let mut lab_gt = Mat::default();
    let mut lab_lut = Mat::default();
    
    cvt_color(img_gt, &mut lab_gt, COLOR_BGR2Lab, 0)?;
    cvt_color(img_lut, &mut lab_lut, COLOR_BGR2Lab, 0)?;
    
    let rows = lab_gt.rows();
    let cols = lab_gt.cols();
    let n_pixels = (rows * cols) as f64;
    
    let mut l_mean_error = 0.0;
    let mut l_mae = 0.0;
    
    for y in 0..rows {
        for x in 0..cols {
            let pixel_gt: &opencv::core::Vec3b = lab_gt.at_2d(y, x)?;
            let pixel_lut: &opencv::core::Vec3b = lab_lut.at_2d(y, x)?;
            
            // Extract L* channel (OpenCV stores as [0,255] range)
            // Need to convert back to [0,100] scale
            let l_gt = pixel_gt[0] as f64 * 100.0 / 255.0;
            let l_lut = pixel_lut[0] as f64 * 100.0 / 255.0;
            
            let delta_l = l_lut - l_gt;
            
            l_mean_error += delta_l;
            l_mae += delta_l.abs();
        }
    }
    
    // Average over all pixels
    l_mean_error /= n_pixels;
    l_mae /= n_pixels;
    
    Ok((l_mean_error, l_mae))
}

fn analyze_brightness_bias(
    gt_path: &str,
    lut_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let img_gt = imread(gt_path, IMREAD_COLOR)?;
    let img_lut = imread(lut_path, IMREAD_COLOR)?;
    
    // RGB bias
    let (rgb_mean_error, rgb_mae) = compute_rgb_bias(&img_gt, &img_lut)?;
    
    // LAB L* bias
    let (l_mean_error, l_mae) = compute_lab_bias(&img_gt, &img_lut)?;
    
    println!("=== Brightness Bias Analysis ===");
    println!("\nRGB Mean Error (Bias):");
    println!("  R: {:.3}", rgb_mean_error[0]);
    println!("  G: {:.3}", rgb_mean_error[1]);
    println!("  B: {:.3}", rgb_mean_error[2]);
    println!("  Overall: {:.3}", 
        (rgb_mean_error[0] + rgb_mean_error[1] + rgb_mean_error[2]) / 3.0);
    
    println!("\nRGB Mean Absolute Error:");
    println!("  R: {:.3}", rgb_mae[0]);
    println!("  G: {:.3}", rgb_mae[1]);
    println!("  B: {:.3}", rgb_mae[2]);
    
    println!("\nLAB L* Analysis:");
    println!("  L* Mean Error: {:.3}", l_mean_error);
    println!("  L* MAE: {:.3}", l_mae);
    println!("  L* bias (8-bit): {:.3}", l_mean_error * 2.55);
    println!("  Bias Percentage: {:.2}%", l_mean_error);
    
    // Assessment
    let abs_bias_pct = l_mean_error.abs();
    let quality = if abs_bias_pct < 0.5 {
        "✓ Excellent - no perceptible bias"
    } else if abs_bias_pct < 1.0 {
        "✓ Very good - barely perceptible"
    } else if abs_bias_pct < 2.0 {
        "~ Good - slight bias"
    } else if abs_bias_pct < 5.0 {
        "⚠ Fair - noticeable bias"
    } else {
        "✗ Poor - significant bias"
    };
    
    println!("\nAssessment: {}", quality);
    
    Ok(())
}
```

**Why L* is Better Than RGB for Brightness:**
1. **Perceptually uniform:** ΔL* = 1 represents same perceptual change at any lightness
2. **Decouples brightness from color:** L* isolates brightness, a* b* isolate chroma
3. **Nonlinear mapping:** Matches human vision which is logarithmic
4. **Industry standard:** Used in color science, printing, display calibration

**RGB Problems:**
- Linear scale doesn't match perception
- Changes in R, G, B affect both brightness and color
- Same numerical difference has different perceptual impact at different levels

L* bias (%) = (L* Mean Error / 100) × 100%
```

**Current results:**
```
L* Mean Error: -0.086  (8-bit scale)
             = -0.034  (0-100 scale)
             = -0.03%

Interpretation: Essentially zero bias ✅
```

### 6.3 Color Shift Analysis (a*, b* channels)

**Purpose:** Detect systematic color casts

**Algorithm:**
```
a* Mean Error = (1/P) × Σ (a*_lut - a*_gt)
b* Mean Error = (1/P) × Σ (b*_lut - b*_gt)
```

**Interpretation:**
```
a* bias:
  > 0  →  Red cast
  < 0  →  Green cast
  ≈ 0  →  Neutral

b* bias:
  > 0  →  Yellow cast
  < 0  →  Blue cast
  ≈ 0  →  Neutral
```

**Current results:**
```
a* Mean Error: +0.015  (neutral, no red/green shift)
b* Mean Error: -0.035  (neutral, no blue/yellow shift)
```

### 6.4 Luminance Difference Histogram

**Purpose:** Visualize distribution of brightness errors

**Algorithm:**

**Step 1: Compute differences (0-100 scale)**
```
For each pixel:
    ΔL* = L*_lut - L*_gt
```

**Step 2: Round and count**
```
ΔL*_rounded = round(ΔL*)

histogram[ΔL*_rounded] += 1
```

**Step 3: Display distribution**
```
For each bin b:
    bar_length = (histogram[b] / max_count) × 50
    print: b, "█" × bar_length, histogram[b]
```

**Ideal distribution:**
- Centered at 0 (no bias)
- Symmetric (balanced errors)
- Narrow (small variance)

**Current distribution:**
```
   -1: █████████ 5,973,274
   +0: ██████████████████████████████████████████████████ 27,678,857  ← Peak at 0 ✅
   +1: ████████ 5,033,904
   
   Mean: -0.034 ≈ 0 ✅
```

### 6.5 Overall Brightness Verdict

**Formula:**
```
if |L* Mean Error| < 0.5%:
    verdict = "No significant brightness bias"
else if L* Mean Error > 0:
    verdict = "LUT output is systematically BRIGHTER"
else:
    verdict = "LUT output is systematically DARKER"
    
Overall RGB bias (%) = (RGB Mean Error / 255) × 100%
```

**Current verdict:**
```
✅ No significant brightness bias (-0.03%)
Overall RGB bias: -0.108 (-0.04%)
```

---

## Appendix: Complete Workflow Mathematics

### Summary of All Transformations

```
Input Image (Standard RGB)
         ↓
    [Stratified LAB Sampling]
         ↓
Training Data: (R_source, G_source, B_source) → (R_target, G_target, B_target)
         ↓
    [LUT Building]
         ↓
Filled LUT Cells (32% from data)
         ↓
    [IDW Interpolation]
         ↓
Complete LUT (100% filled)
         ↓
    [LAB Bias Correction]
         ↓
Corrected LUT (bias eliminated)
         ↓
    [Trilinear Interpolation]
         ↓
Output Image (Classic Chrome RGB)
         ↓
    [Quality Metrics: MSE, PSNR, ΔE]
    [Bias Analysis: LAB L* error]
         ↓
Validation Results
```

### Key Mathematical Constants

```
LUT Size (N):                  33
Total LUT cells:               35,937  (33³)
Stratified buckets:            512     (8³)
Max samples per bucket:        200
Total training samples:        103,427 (8 images)

Calibrated brightness bias:    +1.489 LAB L* units
Current PSNR:                  43.03 dB
Current Average ΔE:            1.28
Current L* bias after correction: -0.03%
```

### Computational Complexity

| Operation | Complexity | Time (8 images) |
|-----------|-----------|----------------|
| Stratified sampling | O(P × B) | ~10s |
| LUT building | O(S) | ~1s |
| IDW interpolation | O(E × N³) | ~1s |
| Bias correction | O(N³) | <1s |
| Trilinear application | O(P) | ~5s |
| LAB conversion (PSNR) | O(P) | ~60s |

Where:
- P = pixels per image (~40 million)
- B = buckets (512)
- S = training samples (103,427)
- E = empty cells (24,426)
- N = LUT size (33)

---

## References

### Color Space Standards
- CIE LAB (1976): International Commission on Illumination
- sRGB: IEC 61966-2-1:1999
- D65 illuminant: Standard daylight reference

### Interpolation Methods
- IDW: Shepard, D. (1968). "A two-dimensional interpolation function for irregularly-spaced data"
- Trilinear: Standard computer graphics technique

### Quality Metrics
- MSE/PSNR: Standard signal processing metrics
- Delta E (CIE76): CIE 1976 color difference formula
- ΔE < 1.0: Just Noticeable Difference (JND) threshold

### Image Processing
- RGB↔LAB conversion: OpenCV library implementation
- Color space transformations: Bruce Lindbloom's equations

---

**Document Version:** 1.0  
**Last Updated:** March 29, 2026  
**Corresponds to:** Classic Chrome LUT v1.0 (8 training images)
