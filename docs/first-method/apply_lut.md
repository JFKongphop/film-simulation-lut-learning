# Third Method: Applying 3D LUT with Trilinear Interpolation

**File:** `src/bin/apply_lut.rs`

## Overview

This is the **third step** in the Classic Chrome LUT workflow. It loads the 3D LUT from the `.cube` file (`outputs/lut_33.cube`) and applies it to images using **trilinear interpolation**. This transforms every pixel from the source color space to the target color space (Classic Chrome film simulation).

---

## Table of Contents

1. [Purpose and Motivation](#purpose-and-motivation)
2. [CUBE File Loading](#cube-file-loading)
3. [Trilinear Interpolation Mathematics](#trilinear-interpolation-mathematics)
4. [Image Processing Workflow](#image-processing-workflow)
5. [Complete Algorithm](#complete-algorithm)
6. [Mathematical Properties](#mathematical-properties)

---

## Purpose and Motivation

### What Does This Step Do?

Given:
- **Input:** A 33×33×33 LUT (35,937 discrete color mappings)
- **Image:** Arbitrary resolution with millions of pixels

**Problem:** Most pixel colors fall **between** LUT grid points, not exactly on them.

**Example:**
- LUT has values at $R = 0.0, 0.03125, 0.0625, \ldots, 1.0$ (33 discrete values)
- Pixel has $R = 0.5$ (exactly on grid) ✓
- Pixel has $R = 0.517$ (between grid points 0.5 and 0.53125) ✗

**Solution:** Use **trilinear interpolation** to smoothly estimate output colors for any input color by interpolating between surrounding LUT cells.

### Why Trilinear Interpolation?

**Alternatives:**

| Method | Accuracy | Speed | Smoothness | Artifacts |
|--------|----------|-------|------------|-----------|
| **Nearest neighbor** | Low | Fastest | No | Banding |
| **Linear (1D)** | Medium | Fast | Partial | Color shifts |
| **Bilinear (2D)** | Good | Fast | Good | Edge artifacts |
| **Trilinear (3D)** | Excellent | Fast | Excellent | None ✓ |
| **Tricubic (3D)** | Best | Slow | Best | None |

**Trilinear** is the optimal balance of accuracy, speed, and smoothness for real-time color grading.

---

## CUBE File Loading

### File Format Review

Standard `.cube` format from `build_lut.rs`:

```
# Comments
TITLE "Classic Chrome LUT - Corrected"
LUT_3D_SIZE 33

0.000000 0.000000 0.000000
0.000234 0.001123 0.002456
...
1.000000 1.000000 1.000000
```

**Data ordering:** Blue changes fastest (innermost loop):

```
for R = 0 to 32:
    for G = 0 to 32:
        for B = 0 to 32:
            write LUT[R][G][B]
```

### Loading Algorithm

**Code from `apply_lut.rs`:**

```rust
/// Represents a 3D LUT loaded from a .cube file
struct Lut3D {
  size: usize,
  data: Vec<Vec<Vec<[f32; 3]>>>, // [R][G][B] -> [R', G', B']
}

impl Lut3D {
  /// Load a 3D LUT from a .cube file
  fn from_cube_file<P: AsRef<Path>>(path: P) -> Result<Self> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut size = 0;
    let mut rgb_values = Vec::new();

    // Parse .cube file
    for line in reader.lines() {
      let line = line?;
      let line = line.trim();

      // Skip comments and empty lines
      if line.is_empty() || line.starts_with('#') {
        continue;
      }

      // Parse LUT_3D_SIZE
      if line.starts_with("LUT_3D_SIZE") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
          size = parts[1].parse()?;
        }
        continue;
      }

      // Skip TITLE and other metadata
      if line.starts_with("TITLE") || line.starts_with("DOMAIN_MIN") || line.starts_with("DOMAIN_MAX") {
        continue;
      }

      // Parse RGB values
      let parts: Vec<&str> = line.split_whitespace().collect();
      if parts.len() >= 3 {
        let r: f32 = parts[0].parse()?;
        let g: f32 = parts[1].parse()?;
        let b: f32 = parts[2].parse()?;
        rgb_values.push([r, g, b]);
      }
    }

    if size == 0 {
      anyhow::bail!("LUT_3D_SIZE not found in .cube file");
    }

    let expected_count = size * size * size;
    if rgb_values.len() != expected_count {
      anyhow::bail!(
        "Expected {} RGB values, found {}",
        expected_count,
        rgb_values.len()
      );
    }

    // Build 3D array from flat list
    // Standard .cube format: Blue changes fastest, then Green, then Red
    let mut data = vec![vec![vec![[0.0f32; 3]; size]; size]; size];
    let mut idx = 0;
    for r in 0..size {
      for g in 0..size {
        for b in 0..size {
          data[r][g][b] = rgb_values[idx];
          idx += 1;
        }
      }
    }

    Ok(Lut3D { size, data })
  }
}
```

### Parsing Steps

**Step 1: Read file line by line**

Skip:
- Empty lines
- Comments (starting with `#`)
- Metadata (`TITLE`, `DOMAIN_MIN`, `DOMAIN_MAX`)

**Step 2: Extract LUT size**

```rust
if line.starts_with("LUT_3D_SIZE") {
  size = parts[1].parse()?;  // Extract "33" from "LUT_3D_SIZE 33"
}
```

**Step 3: Parse RGB triplets**

Each data line: `"0.123456 0.234567 0.345678"`

```rust
let r: f32 = parts[0].parse()?;
let g: f32 = parts[1].parse()?;
let b: f32 = parts[2].parse()?;
rgb_values.push([r, g, b]);
```

**Step 4: Validate count**

$$
\text{Expected count} = N^3 = 33^3 = 35{,}937
$$

```rust
if rgb_values.len() != expected_count {
  anyhow::bail!("Wrong number of values");
}
```

**Step 5: Convert flat list to 3D array**

**Mapping formula:** Given line index $L$ (0-indexed) in `.cube` file:

$$
\begin{aligned}
B &= L \bmod N \\
G &= \left\lfloor \frac{L}{N} \right\rfloor \bmod N \\
R &= \left\lfloor \frac{L}{N^2} \right\rfloor
\end{aligned}
$$

**Code:**

```rust
let mut data = vec![vec![vec![[0.0f32; 3]; size]; size]; size];
let mut idx = 0;
for r in 0..size {
  for g in 0..size {
    for b in 0..size {
      data[r][g][b] = rgb_values[idx];
      idx += 1;
    }
  }
}
```

**Result:** 3D array `data[R][G][B]` where each cell contains `[R', G', B']` output color.

---

## Trilinear Interpolation Mathematics

### Overview

**Trilinear interpolation** extends linear interpolation to 3D space. It estimates the value at any point within a 3D grid by weighted averaging of the 8 surrounding corner points.

### Step 1: Compute LUT Position

Given input RGB color $(r, g, b) \in [0, 1]$, map to LUT coordinates:

$$
\begin{aligned}
x &= r \times (N - 1) \\
y &= g \times (N - 1) \\
z &= b \times (N - 1)
\end{aligned}
$$

Where $N = 33$ (LUT size).

**Example:** For $r = 0.517$:
$$
x = 0.517 \times 32 = 16.544
$$

This means the color falls between LUT indices 16 and 17.

**Code from `apply_lut.rs`:**

```rust
let n = self.size as f32;  // 33.0
let n_max = (self.size - 1) as usize;  // 32

// Step 1: Compute LUT position
let x = r * (n - 1.0);  // maps [0, 1] to [0, 32]
let y = g * (n - 1.0);
let z = b * (n - 1.0);
```

### Step 2: Get Surrounding Indices

Extract integer and fractional parts:

$$
\begin{aligned}
x_0 &= \lfloor x \rfloor, \quad x_1 = \min(x_0 + 1, N - 1) \\
y_0 &= \lfloor y \rfloor, \quad y_1 = \min(y_0 + 1, N - 1) \\
z_0 &= \lfloor z \rfloor, \quad z_1 = \min(z_0 + 1, N - 1)
\end{aligned}
$$

**Example:** For $x = 16.544$:
- $x_0 = 16$ (lower bound)
- $x_1 = 17$ (upper bound, clamped to max 32)

**Code from `apply_lut.rs`:**

```rust
// Step 2: Get surrounding indices
let x0 = x.floor() as usize;
let y0 = y.floor() as usize;
let z0 = z.floor() as usize;

let x1 = (x0 + 1).min(n_max);  // Clamp to prevent overflow at edges
let y1 = (y0 + 1).min(n_max);
let z1 = (z0 + 1).min(n_max);
```

**Clamping:** The `min(n_max)` prevents index overflow when input is exactly 1.0:
- Input $r = 1.0 \Rightarrow x = 32.0$
- $x_0 = 32$, $x_1 = \min(33, 32) = 32$ (same cell, no interpolation needed)

### Step 3: Compute Interpolation Weights

Extract fractional parts (distance from lower bound):

$$
\begin{aligned}
d_x &= x - x_0 \quad &\in [0, 1) \\
d_y &= y - y_0 \quad &\in [0, 1) \\
d_z &= z - z_0 \quad &\in [0, 1)
\end{aligned}
$$

**Example:** For $x = 16.544$:
$$
d_x = 16.544 - 16 = 0.544
$$

**Interpretation:** The point is 54.4% of the way from index 16 to index 17.

**Code from `apply_lut.rs`:**

```rust
// Step 3: Compute interpolation weights
let dx = x - x0 as f32;
let dy = y - y0 as f32;
let dz = z - z0 as f32;
```

### Step 4: Fetch 8 Corner Values

The input color $(x, y, z)$ lies within a cube defined by 8 corners:

**Corner positions:**
$$
\begin{aligned}
c_{000} &= \text{LUT}[x_0][y_0][z_0] \quad &\text{(lower-left-back)} \\
c_{001} &= \text{LUT}[x_0][y_0][z_1] \quad &\text{(lower-left-front)} \\
c_{010} &= \text{LUT}[x_0][y_1][z_0] \quad &\text{(lower-right-back)} \\
c_{011} &= \text{LUT}[x_0][y_1][z_1] \quad &\text{(lower-right-front)} \\
c_{100} &= \text{LUT}[x_1][y_0][z_0] \quad &\text{(upper-left-back)} \\
c_{101} &= \text{LUT}[x_1][y_0][z_1] \quad &\text{(upper-left-front)} \\
c_{110} &= \text{LUT}[x_1][y_1][z_0] \quad &\text{(upper-right-back)} \\
c_{111} &= \text{LUT}[x_1][y_1][z_1] \quad &\text{(upper-right-front)}
\end{aligned}
$$

**Subscript notation:** Binary digits (0 = lower, 1 = upper) for $(x, y, z)$ dimensions.

**Code from `apply_lut.rs`:**

```rust
// Step 4: Fetch 8 corner values
let c000 = self.data[x0][y0][z0];
let c001 = self.data[x0][y0][z1];
let c010 = self.data[x0][y1][z0];
let c011 = self.data[x0][y1][z1];
let c100 = self.data[x1][y0][z0];
let c101 = self.data[x1][y0][z1];
let c110 = self.data[x1][y1][z0];
let c111 = self.data[x1][y1][z1];
```

Each $c_{ijk}$ is a 3D vector: $c_{ijk} = [R', G', B']$

### Step 5: Trilinear Interpolation

**Sequential approach:** Interpolate in 3 stages (Z → Y → X).

#### Stage 1: Interpolate along Z-axis (4 interpolations)

Interpolate between pairs at same $(x, y)$ but different $z$:

$$
\begin{aligned}
c_{00} &= c_{000} \cdot (1 - d_z) + c_{001} \cdot d_z \\
c_{01} &= c_{010} \cdot (1 - d_z) + c_{011} \cdot d_z \\
c_{10} &= c_{100} \cdot (1 - d_z) + c_{101} \cdot d_z \\
c_{11} &= c_{110} \cdot (1 - d_z) + c_{111} \cdot d_z
\end{aligned}
$$

**Result:** 4 intermediate values on the faces of the cube.

#### Stage 2: Interpolate along Y-axis (2 interpolations)

Interpolate between pairs at same $x$ but different $y$:

$$
\begin{aligned}
c_0 &= c_{00} \cdot (1 - d_y) + c_{01} \cdot d_y \\
c_1 &= c_{10} \cdot (1 - d_y) + c_{11} \cdot d_y
\end{aligned}
$$

**Result:** 2 intermediate values on opposite edges of the cube.

#### Stage 3: Interpolate along X-axis (1 interpolation)

Final interpolation:

$$
\text{result} = c_0 \cdot (1 - d_x) + c_1 \cdot d_x
$$

**Result:** Final output color at position $(x, y, z)$.

### Complete Trilinear Formula

Expanding all stages gives the direct formula:

$$
\begin{aligned}
\text{result} &= c_{000} \cdot (1 - d_x)(1 - d_y)(1 - d_z) \\
              &+ c_{001} \cdot (1 - d_x)(1 - d_y) \cdot d_z \\
              &+ c_{010} \cdot (1 - d_x) \cdot d_y \cdot (1 - d_z) \\
              &+ c_{011} \cdot (1 - d_x) \cdot d_y \cdot d_z \\
              &+ c_{100} \cdot d_x \cdot (1 - d_y)(1 - d_z) \\
              &+ c_{101} \cdot d_x \cdot (1 - d_y) \cdot d_z \\
              &+ c_{110} \cdot d_x \cdot d_y \cdot (1 - d_z) \\
              &+ c_{111} \cdot d_x \cdot d_y \cdot d_z
\end{aligned}
$$

**Weight verification:** Sum of all weights equals 1:

$$
\sum_{\substack{i, j, k \in \{0, 1\}}} w_{ijk} = [(1-d_x) + d_x] \cdot [(1-d_y) + d_y] \cdot [(1-d_z) + d_z] = 1 \cdot 1 \cdot 1 = 1
$$

### Code Implementation: Sequential Approach

**Code from `apply_lut.rs`:**

```rust
/// Apply LUT to a single RGB value using trilinear interpolation
fn apply(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
  let n = self.size as f32;
  let n_max = (self.size - 1) as usize;

  // Step 1: Compute LUT position
  let x = r * (n - 1.0);
  let y = g * (n - 1.0);
  let z = b * (n - 1.0);

  // Step 2: Get surrounding indices
  let x0 = x.floor() as usize;
  let y0 = y.floor() as usize;
  let z0 = z.floor() as usize;

  let x1 = (x0 + 1).min(n_max);
  let y1 = (y0 + 1).min(n_max);
  let z1 = (z0 + 1).min(n_max);

  // Step 3: Compute interpolation weights
  let dx = x - x0 as f32;
  let dy = y - y0 as f32;
  let dz = z - z0 as f32;

  // Step 4: Fetch 8 corner values
  let c000 = self.data[x0][y0][z0];
  let c001 = self.data[x0][y0][z1];
  let c010 = self.data[x0][y1][z0];
  let c011 = self.data[x0][y1][z1];
  let c100 = self.data[x1][y0][z0];
  let c101 = self.data[x1][y0][z1];
  let c110 = self.data[x1][y1][z0];
  let c111 = self.data[x1][y1][z1];

  // Step 5: Trilinear interpolation (sequential: Z -> Y -> X)
  let mut result = [0.0f32; 3];
  for channel in 0..3 {
    // Stage 1: Interpolate along Z-axis
    let c00 = c000[channel] * (1.0 - dz) + c001[channel] * dz;
    let c01 = c010[channel] * (1.0 - dz) + c011[channel] * dz;
    let c10 = c100[channel] * (1.0 - dz) + c101[channel] * dz;
    let c11 = c110[channel] * (1.0 - dz) + c111[channel] * dz;

    // Stage 2: Interpolate along Y-axis
    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    // Stage 3: Interpolate along X-axis
    result[channel] = c0 * (1.0 - dx) + c1 * dx;
  }

  result
}
```

**Per-channel processing:** R, G, B channels are interpolated independently (same weights, different corner values).

### Why Sequential, Not Direct?

**Sequential approach (used here):**
- 7 interpolations per channel (4 + 2 + 1)
- Clear, readable code
- Easier to debug
- Intermediate values can be inspected

**Direct approach:**
- 8 multiplications, 7 additions per channel
- More compact
- Slightly faster (fewer intermediate variables)
- Less readable

For our use case, **sequential is preferred** for clarity and maintainability.

---

## Image Processing Workflow

### Apply LUT to Entire Image

**Code from `apply_lut.rs`:**

```rust
/// Apply LUT to an entire image
fn apply_to_image(&self, input: &Mat) -> Result<Mat> {
  let rows = input.rows();
  let cols = input.cols();

  let mut output = input.clone();

  // Process each pixel
  for y in 0..rows {
    for x in 0..cols {
      // Get BGR pixel (OpenCV uses BGR order)
      let pixel = input.at_2d::<core::Vec3b>(y, x)?;
      
      // Convert to [0, 1] range
      let b = pixel[0] as f32 / 255.0;
      let g = pixel[1] as f32 / 255.0;
      let r = pixel[2] as f32 / 255.0;

      // Apply LUT (with RGB order)
      let transformed = self.apply(r, g, b);

      // Clamp and convert back to [0, 255]
      let r_out = (transformed[0].clamp(0.0, 1.0) * 255.0) as u8;
      let g_out = (transformed[1].clamp(0.0, 1.0) * 255.0) as u8;
      let b_out = (transformed[2].clamp(0.0, 1.0) * 255.0) as u8;

      // Set output pixel (BGR order)
      let output_pixel = output.at_2d_mut::<core::Vec3b>(y, x)?;
      output_pixel[0] = b_out;
      output_pixel[1] = g_out;
      output_pixel[2] = r_out;
    }
  }

  Ok(output)
}
```

### Processing Steps (Per Pixel)

**Step 1: Read input pixel (BGR order)**

```rust
let pixel = input.at_2d::<core::Vec3b>(y, x)?;
```

OpenCV uses **BGR** channel ordering (historical reasons from early bitmap formats).

**Step 2: Normalize to [0, 1]**

$$
\begin{aligned}
r &= \frac{\text{pixel}[2]}{255}, \quad r \in [0, 1] \\
g &= \frac{\text{pixel}[1]}{255}, \quad g \in [0, 1] \\
b &= \frac{\text{pixel}[0]}{255}, \quad b \in [0, 1]
\end{aligned}
$$

**Step 3: Apply LUT with trilinear interpolation**

```rust
let transformed = self.apply(r, g, b);
```

Returns `[r', g', b']` in [0, 1] range (note: RGB order).

**Step 4: Denormalize to [0, 255]**

$$
\begin{aligned}
r_{\text{out}} &= \lfloor \text{clamp}(r', 0, 1) \times 255 \rfloor \\
g_{\text{out}} &= \lfloor \text{clamp}(g', 0, 1) \times 255 \rfloor \\
b_{\text{out}} &= \lfloor \text{clamp}(b', 0, 1) \times 255 \rfloor
\end{aligned}
$$

**Clamping** ensures out-of-range values (from interpolation or correction) are mapped to valid 8-bit range.

**Step 5: Write output pixel (BGR order)**

```rust
output_pixel[0] = b_out;  // Blue
output_pixel[1] = g_out;  // Green
output_pixel[2] = r_out;  // Red
```

### Main Function Workflow

**Code from `apply_lut.rs`:**

```rust
fn main() -> Result<()> {
  println!("🎨 Applying 3D LUT to Image");

  let lut_path = "outputs/lut_33.cube";
  let input_path = "source/compare/standard/9.JPG";
  let output_path = "outputs/lut_33.jpg";

  // Step 1: Load LUT
  println!("📖 Loading LUT from: {}", lut_path);
  let lut = Lut3D::from_cube_file(lut_path)?;
  println!("✅ Loaded {}x{}x{} LUT", lut.size, lut.size, lut.size);

  // Show sample LUT values
  let black_out = lut.apply(0.0, 0.0, 0.0);
  println!("   Black [0,0,0] -> [{:.4}, {:.4}, {:.4}]", black_out[0], black_out[1], black_out[2]);
  
  let white_out = lut.apply(1.0, 1.0, 1.0);
  println!("   White [1,1,1] -> [{:.4}, {:.4}, {:.4}]", white_out[0], white_out[1], white_out[2]);
  
  let mid_out = lut.apply(0.5, 0.5, 0.5);
  println!("   Mid [0.5,0.5,0.5] -> [{:.4}, {:.4}, {:.4}]", mid_out[0], mid_out[1], mid_out[2]);

  // Step 2: Load input image
  println!("\n📷 Loading input image: {}", input_path);
  let input = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;
  println!("✅ Loaded image: {}x{}", input.cols(), input.rows());

  // Step 3: Apply LUT
  println!("\n⚙️  Applying LUT with trilinear interpolation...");
  let output = lut.apply_to_image(&input)?;
  println!("✅ LUT applied successfully");

  // Step 4: Save output
  println!("\n💾 Saving output to: {}", output_path);
  imgcodecs::imwrite(output_path, &output, &core::Vector::new())?;
  println!("✅ Output saved successfully");

  println!("\n🎉 Done!");

  Ok(())
}
```

---

## Complete Algorithm

### High-Level Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  Input: outputs/lut_33.cube (35,937 RGB mappings)           │
│         source/compare/standard/9.JPG (test image)           │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 1: Load and parse .cube file                          │
│  - Read LUT_3D_SIZE (33)                                     │
│  - Parse 35,937 RGB triplets                                 │
│  - Build 3D array data[R][G][B]                              │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 2: Load input image with OpenCV                       │
│  - imread() returns Mat (BGR format)                         │
│  - Get dimensions: rows × cols                               │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 3: Process each pixel                                  │
│  For each pixel (x, y) in image:                             │
│    1. Read BGR values [0, 255]                               │
│    2. Normalize to RGB [0, 1]                                │
│    3. Compute LUT position (x, y, z) = RGB × 32              │
│    4. Get 8 corner indices (x0, x1, y0, y1, z0, z1)          │
│    5. Compute weights (dx, dy, dz)                           │
│    6. Fetch 8 corner colors from LUT                         │
│    7. Trilinear interpolate: Z -> Y -> X                     │
│    8. Clamp result to [0, 1]                                 │
│    9. Denormalize to BGR [0, 255]                            │
│   10. Write to output image                                  │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 4: Save output image                                   │
│  - imwrite() saves as JPEG                                   │
│  - Output: outputs/lut_33.jpg                                │
└──────────────────────────────────────────────────────────────┘
```

### Pseudocode with Mathematics

```
function apply_lut_to_image(lut, input_image):
    output_image = clone(input_image)
    
    for each pixel at (x, y):
        // Read and normalize
        BGR = input_image[y, x]  // [0, 255]
        R = BGR[2] / 255         // [0, 1]
        G = BGR[1] / 255
        B = BGR[0] / 255
        
        // Map to LUT coordinates
        lut_x = R × (N - 1)      // [0, 32]
        lut_y = G × (N - 1)
        lut_z = B × (N - 1)
        
        // Get surrounding cell indices
        x0 = floor(lut_x)
        x1 = min(x0 + 1, N - 1)
        y0 = floor(lut_y)
        y1 = min(y0 + 1, N - 1)
        z0 = floor(lut_z)
        z1 = min(z0 + 1, N - 1)
        
        // Compute interpolation weights
        dx = lut_x - x0
        dy = lut_y - y0
        dz = lut_z - z0
        
        // Fetch 8 corner colors
        c000 = lut[x0, y0, z0]
        c001 = lut[x0, y0, z1]
        c010 = lut[x0, y1, z0]
        c011 = lut[x0, y1, z1]
        c100 = lut[x1, y0, z0]
        c101 = lut[x1, y0, z1]
        c110 = lut[x1, y1, z0]
        c111 = lut[x1, y1, z1]
        
        // Trilinear interpolation (sequential)
        for each channel in [R, G, B]:
            // Stage 1: Z-axis (4 interpolations)
            c00 = c000[ch] × (1 - dz) + c001[ch] × dz
            c01 = c010[ch] × (1 - dz) + c011[ch] × dz
            c10 = c100[ch] × (1 - dz) + c101[ch] × dz
            c11 = c110[ch] × (1 - dz) + c111[ch] × dz
            
            // Stage 2: Y-axis (2 interpolations)
            c0 = c00 × (1 - dy) + c01 × dy
            c1 = c10 × (1 - dy) + c11 × dy
            
            // Stage 3: X-axis (1 interpolation)
            result[ch] = c0 × (1 - dx) + c1 × dx
        
        // Clamp and denormalize
        R_out = clamp(result[R], 0, 1) × 255
        G_out = clamp(result[G], 0, 1) × 255
        B_out = clamp(result[B], 0, 1) × 255
        
        // Write output (BGR order)
        output_image[y, x] = [B_out, G_out, R_out]
    
    return output_image
```

---

## Mathematical Properties

### 1. Continuity

**Property:** Output color changes smoothly as input color changes.

**Proof:** Trilinear interpolation uses continuous functions:
- $f(d) = (1 - d) \cdot a + d \cdot b$ is continuous for $d \in [0, 1]$
- Composition of continuous functions is continuous
- No discontinuities at cell boundaries

**Result:** No visible banding or color breaks in output images.

### 2. Exactness at Grid Points

**Property:** When input color falls exactly on a LUT grid point, output matches LUT value exactly.

**Proof:** If $(x, y, z) = (i, j, k)$ where $i, j, k$ are integers:
- $d_x = 0, d_y = 0, d_z = 0$
- All weights for non-corner cells become 0
- Only $c_{000}$ has weight $(1-0)(1-0)(1-0) = 1$
- $\text{result} = c_{000} \times 1 = \text{LUT}[i][j][k]$

**Result:** Training data colors are reproduced exactly.

### 3. Bounded Output

**Property:** Output values stay within [0, 1] if all LUT values are in [0, 1].

**Proof:** Trilinear interpolation is a **convex combination**:
- All weights $w_{ijk} \geq 0$
- $\sum w_{ijk} = 1$
- Result is weighted average: $\text{result} = \sum w_{ijk} \cdot c_{ijk}$
- If all $c_{ijk} \in [0, 1]$, then $\text{result} \in [0, 1]$

**Additional safety:** Code includes explicit clamping: `clamp(result, 0.0, 1.0)`

### 4. Smoothness Order

**Property:** Trilinear interpolation is C⁰ continuous (value continuous, derivatives not continuous).

**Derivative behavior:**
- **At grid points:** First derivatives may be discontinuous
- **Between grid points:** Smooth (piecewise linear)

**Visual impact:** Not noticeable at 33³ resolution for typical images.

**Better alternatives (not used here):**
- **Tricubic:** C¹ continuous (smooth derivatives), but much slower
- **Higher resolution LUT:** 65³ or 129³ makes derivative discontinuities imperceptible

### 5. Computational Complexity

**Per-pixel operations:**
- Normalization: 3 divisions
- Position calculation: 3 multiplications
- Floor operations: 3 floor()
- Index clamping: 3 min()
- Weight computation: 3 subtractions
- LUT lookups: 8 array accesses (cached)
- Interpolation: 7 lerps × 3 channels = 21 operations (mults + adds)
- Denormalization: 3 multiplications, 3 clamping, 3 casts

**Total per pixel:** ~50-60 operations

**For 4K image (3840 × 2160 = 8,294,400 pixels):**
$$
8{,}294{,}400 \times 60 \approx 497{,}664{,}000 \text{ operations} \approx 0.5 \text{ billion operations}
$$

**Modern CPU:** ~1-2 seconds for 4K image (highly parallelizable with SIMD or GPU).

### 6. Memory Access Pattern

**Cache efficiency:**
- 8 corner lookups are spatially local in memory
- Good cache hit rate (adjacent cells often accessed together)
- 3D array layout: `data[R][G][B]` → varies B fastest (memory-contiguous)

**Memory bandwidth:** Reading LUT (~140 KB) is negligible compared to image I/O.

---

## Performance Considerations

### Optimization Opportunities

1. **SIMD (Single Instruction, Multiple Data):**
   - Process 4 or 8 pixels simultaneously
   - 4× to 8× speedup possible

2. **GPU acceleration:**
   - Massively parallel (millions of pixels)
   - 100× to 1000× speedup for large images

3. **Lookup table caching:**
   - LUT fits entirely in L3 cache (~140 KB)
   - Preload to avoid cache misses

4. **16-bit LUT:**
   - Use 16-bit integers instead of 32-bit floats
   - 2× memory bandwidth reduction
   - Slightly lower precision (acceptable)

**Current implementation:** Single-threaded, no SIMD, but sufficient for ~1-2 second processing on typical images.

---

## Output Quality

### Expected Results

**Quality metrics** (from validation):
- **MSE:** ~3.24 (very low)
- **PSNR:** ~43 dB (excellent, professional quality)
- **Delta E (avg):** ~1.28 (barely perceptible)
- **Brightness bias:** -0.03% (negligible)

**Visual characteristics:**
- Smooth color transitions (no banding)
- Accurate color reproduction
- No artifacts at edges or gradients
- Film-like Classic Chrome aesthetic

---

## Summary

**Applying the 3D LUT** transforms images from Standard to Classic Chrome color space:

1. ✅ **Loads 33³ LUT from .cube file** (35,937 color mappings)
2. ✅ **Parses and validates LUT structure** (error checking)
3. ✅ **Processes every pixel with trilinear interpolation** (smooth, accurate)
4. ✅ **Sequential interpolation** (Z→Y→X, 7 lerps per channel)
5. ✅ **Handles edge cases** (clamping at boundaries)
6. ✅ **Outputs high-quality JPEG** (professional color grading)

**Trilinear interpolation** is the key algorithm:
- **Smooth:** C⁰ continuous, no visible artifacts
- **Accurate:** Exact at grid points, convex combination between
- **Fast:** ~50-60 operations per pixel, real-time capable
- **Standard:** Used in all professional color grading software

**Next step:** Validate output quality with comparisons and metrics (`compare_lut.rs`)
