# First Method: Stratified LAB Sampling

**File:** `src/bin/stratified_compare_pixel.rs`

## Overview

This is the **first step** in the Classic Chrome LUT workflow. It generates training samples by comparing pairs of images (Standard vs Classic Chrome) and collecting pixel correspondences. To ensure even coverage across the color space, it uses **stratified sampling in LAB space** rather than uniform random sampling in RGB space.

---

## Table of Contents

1. [Purpose and Motivation](#purpose-and-motivation)
2. [RGB to LAB Conversion](#rgb-to-lab-conversion)
3. [LAB Bucket Computation](#lab-bucket-computation)
4. [Stratified Sampling Algorithm](#stratified-sampling-algorithm)
5. [Complete Workflow](#complete-workflow)
6. [Mathematical Properties](#mathematical-properties)

---

## Purpose and Motivation

### Problem: Naive Pixel Sampling

If we randomly sample pixels from images, we get **biased training data**:
- Common colors (skin tones, sky, grass) are over-represented
- Rare colors (saturated reds, deep blues, bright yellows) are under-represented
- Flat regions (walls, sky) contribute thousands of redundant samples
- Result: LUT performs poorly on uncommon colors

### Solution: Stratified LAB Sampling

**Stratify** the LAB color space into buckets and sample evenly from each:

$$
\text{Sampling Goal: } \max_{b \in \text{Buckets}} |\text{samples}_b| = 200
$$

Where:
- Each bucket $b$ corresponds to a region of LAB space
- At most 200 samples per bucket per image
- All buckets are represented equally, regardless of frequency

**Benefits:**
1. **Even coverage:** Rare colors get same representation as common colors
2. **Reduced redundancy:** Large flat areas don't dominate training data
3. **Better generalization:** LUT works well across full color spectrum

---

## RGB to LAB Conversion

### Theory: Why LAB?

**CIELAB (LAB) color space** is designed to be perceptually uniform:
- **L\*:** Lightness from 0 (black) to 100 (white)
- **a\*:** Green (-128) to Red (+127)
- **b\*:** Blue (-128) to Yellow (+127)

$$
\Delta E = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}
$$

Equal distances in LAB space correspond to equal perceptual differences. This makes LAB ideal for bucketing colors by visual similarity.

### Implementation: OpenCV Conversion

**Code from `stratified_compare_pixel.rs`:**

```rust
/// Convert RGB (0-255) to LAB color space
fn rgb_to_lab(r: u8, g: u8, b: u8) -> Result<(f32, f32, f32)> {
  // Create a 1x1 BGR image (OpenCV uses BGR)
  let mut bgr_mat = unsafe {
    Mat::new_rows_cols(1, 1, core::CV_8UC3)?
  };
  
  let pixel = bgr_mat.at_2d_mut::<core::Vec3b>(0, 0)?;
  pixel[0] = b;
  pixel[1] = g;
  pixel[2] = r;
  
  // Convert to LAB
  let mut lab_mat = Mat::default();
  imgproc::cvt_color(&bgr_mat, &mut lab_mat, imgproc::COLOR_BGR2Lab, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
  
  let lab_pixel = lab_mat.at_2d::<core::Vec3b>(0, 0)?;
  
  // OpenCV LAB values are scaled: L: [0, 255] -> [0, 100], a/b: [0, 255] -> [-128, 127]
  let l = lab_pixel[0] as f32 * 100.0 / 255.0;
  let a = lab_pixel[1] as f32 - 128.0;
  let b = lab_pixel[2] as f32 - 128.0;
  
  Ok((l, a, b))
}
```

**Mathematical Steps:**

1. **Create OpenCV Mat:** 1×1 image with `CV_8UC3` (8-bit unsigned, 3 channels)
2. **RGB → BGR ordering:** OpenCV uses BGR channel order
3. **Color conversion:** `cvt_color` with `COLOR_BGR2Lab` flag
   - Internally: RGB → XYZ → LAB using D65 white point
   - Applies gamma correction and matrix transforms
4. **Scale LAB values:**

$$
\begin{aligned}
L^* &= \frac{\text{pixel}[0] \times 100}{255} \quad &\in [0, 100] \\
a^* &= \text{pixel}[1] - 128 \quad &\in [-128, 127] \\
b^* &= \text{pixel}[2] - 128 \quad &\in [-128, 127]
\end{aligned}
$$

OpenCV stores LAB in 8-bit format [0, 255], so we scale it back to standard LAB ranges.

---

## LAB Bucket Computation

### Bucket Grid: 8×8×8 = 512 Buckets

Divide LAB space into **512 discrete buckets** (8 bins per dimension):

**Code from `stratified_compare_pixel.rs`:**

```rust
/// Compute LAB bucket (8x8x8 grid)
fn compute_bucket(l: f32, a: f32, b: f32) -> (usize, usize, usize) {
  // L bucket: 0-100 split into 8 equal ranges
  let l_bin = ((l / 12.5).floor() as usize).min(7);
  
  // a bucket: -128 to 127 split into 8 equal ranges (32 units each)
  let a_bin = (((a + 128.0) / 32.0).floor() as usize).min(7);
  
  // b bucket: -128 to 127 split into 8 equal ranges (32 units each)
  let b_bin = (((b + 128.0) / 32.0).floor() as usize).min(7);
  
  (l_bin, a_bin, b_bin)
}
```

### Mathematical Formulas

**L\* dimension (Lightness):**

$$
\text{l\_bin} = \min\left(\left\lfloor \frac{L^*}{12.5} \right\rfloor, 7\right), \quad L^* \in [0, 100]
$$

- Range: 0 to 100
- Bin width: $\frac{100}{8} = 12.5$
- Bins: 0→[0, 12.5), 1→[12.5, 25), ..., 7→[87.5, 100]

**a\* dimension (Green to Red):**

$$
\text{a\_bin} = \min\left(\left\lfloor \frac{a^* + 128}{32} \right\rfloor, 7\right), \quad a^* \in [-128, 127]
$$

- Range: -128 to 127 (256 units)
- Shift to [0, 255]: $a^* + 128$
- Bin width: $\frac{256}{8} = 32$
- Bins: 0→[-128, -96), 1→[-96, -64), ..., 7→[96, 127]

**b\* dimension (Blue to Yellow):**

$$
\text{b\_bin} = \min\left(\left\lfloor \frac{b^* + 128}{32} \right\rfloor, 7\right), \quad b^* \in [-128, 127]
$$

- Same logic as a\* dimension

### Bucket Key

Each pixel maps to a 3D bucket key:

$$
\text{bucket\_key} = (\text{l\_bin}, \text{a\_bin}, \text{b\_bin}), \quad \text{where each } \in \{0, 1, 2, 3, 4, 5, 6, 7\}
$$

**Total buckets:** $8 \times 8 \times 8 = 512$ possible buckets

**Example:**
- $L^* = 45.2$, $a^* = -10.5$, $b^* = 30.8$
- l\_bin = $\lfloor 45.2 / 12.5 \rfloor = 3$
- a\_bin = $\lfloor (-10.5 + 128) / 32 \rfloor = \lfloor 117.5 / 32 \rfloor = 3$
- b\_bin = $\lfloor (30.8 + 128) / 32 \rfloor = \lfloor 158.8 / 32 \rfloor = 4$
- **bucket\_key = (3, 3, 4)**

---

## Stratified Sampling Algorithm

### Algorithm Overview

**Goal:** Prevent any single bucket from dominating the training dataset

**Strategy:** Cap samples per bucket at 200 pixels

### Step 1: Create Buckets

**Code from `stratified_compare_pixel.rs`:**

```rust
// Use fixed seed for reproducibility
let mut rng = StdRng::seed_from_u64(42);

// Create buckets: (l_bin, a_bin, b_bin) -> Vec<PixelData>
let mut buckets: HashMap<(usize, usize, usize), Vec<PixelData>> = HashMap::new();

// Process all pixels and assign to buckets
for pixel_idx in 0..total_pixels {
  let idx = pixel_idx * 3; // Convert pixel index to byte index (3 bytes per pixel)

  // Get BGR values (OpenCV uses BGR format)
  let b1 = img1.data[idx];
  let g1 = img1.data[idx + 1];
  let r1 = img1.data[idx + 2];

  let b2 = img2.data[idx];
  let g2 = img2.data[idx + 1];
  let r2 = img2.data[idx + 2];

  // Convert source RGB to LAB for bucketing
  let (l, a, b) = rgb_to_lab(r1, g1, b1)?;
  let bucket = compute_bucket(l, a, b);
  
  // Normalize RGB values to [0, 1]
  let sr = r1 as f32 / 255.0;
  let sg = g1 as f32 / 255.0;
  let sb = b1 as f32 / 255.0;
  let cr = r2 as f32 / 255.0;
  let cg = g2 as f32 / 255.0;
  let cb = b2 as f32 / 255.0;
  
  // Store pixel data in bucket
  let pixel_data = PixelData {
    index: pixel_idx,
    sr, sg, sb,  // Source RGB (Standard)
    cr, cg, cb,  // Chrome RGB (Classic Chrome)
    dr: sr - cr, // Difference (not used in LUT building)
    dg: sg - cg,
    db: sb - cb,
  };
  
  buckets.entry(bucket).or_insert_with(Vec::new).push(pixel_data);
}
```

**Data structure:**

$$
\text{buckets} : (\text{l\_bin}, \text{a\_bin}, \text{b\_bin}) \rightarrow \text{Vec}\langle\text{PixelData}\rangle
$$

- **Key:** 3D tuple (l\_bin, a\_bin, b\_bin)
- **Value:** Vector of all pixels that fall in this bucket
- **HashMap:** Efficient lookup and insertion

### Step 2: Sample from Each Bucket

**Code from `stratified_compare_pixel.rs`:**

```rust
// Sample from each bucket
let mut selected_pixels = 0;

for (_bucket_key, mut pixels) in buckets {
  let original_count = pixels.len();
  
  let sampled = if pixels.len() <= 200 {
    // Keep all pixels if 200 or fewer
    pixels
  } else {
    // Randomly sample exactly 200 pixels
    pixels.shuffle(&mut rng);
    pixels.into_iter().take(200).collect()
  };
  
  let sampled_count = sampled.len();
  selected_pixels += sampled_count;
  
  all_pixel_data.extend(sampled);
}
```

**Sampling rule:**

$$
\text{samples}_b = \begin{cases}
\text{all pixels in bucket } b & \text{if } |\text{pixels}_b| \leq 200 \\
\text{random sample of 200} & \text{if } |\text{pixels}_b| > 200
\end{cases}
$$

**Mathematical properties:**
- **Maximum per bucket:** 200 samples
- **Randomization:** Fisher-Yates shuffle with seed 42 (reproducible)
- **Take:** First 200 elements after shuffle

### Step 3: Aggregate Across Images

**Code from `stratified_compare_pixel.rs`:**

```rust
// Process all image pairs (1.JPG through 8.JPG)
for img_num in 1..=8 {
  let filename = format!("{}.JPG", img_num);
  let standard_path = format!("source/compare/standard/{}", filename);
  let chrome_path = format!("source/compare/classic-chrome/{}", filename);
  
  // ... (bucket creation and sampling for this image pair)
  
  all_pixel_data.extend(sampled); // Accumulate samples
}
```

**Total samples formula:**

$$
\text{Total samples} = \sum_{i=1}^{8} \sum_{b \in \text{Buckets}_i} \min(|\text{pixels}_{b,i}|, 200)
$$

Where:
- $i$ = image index (1 to 8)
- $b$ = bucket key
- $\text{Buckets}_i$ = set of non-empty buckets in image $i$

**Expected samples:**
- Maximum per image: $512 \times 200 = 102{,}400$ (if all buckets filled)
- Typical result: ~103,000 samples (across 8 images)

### Step 4: Write to CSV

**Code from `stratified_compare_pixel.rs`:**

```rust
// Write all pixel data to CSV file
let file = File::create("outputs/pixel_comparison.csv")?;
let mut wtr = csv::Writer::from_writer(file);

// Write all pixel data
for pixel_data in &all_pixel_data {
  wtr.serialize(pixel_data)?;
}

wtr.flush()?;
```

**Output format:**

```csv
index,sr,sg,sb,cr,cg,cb,dr,dg,db
12345,0.5294,0.4235,0.3176,0.5098,0.4000,0.2980,0.0196,0.0235,0.0196
...
```

Where:
- `sr, sg, sb` = Source RGB (Standard image) in [0, 1]
- `cr, cg, cb` = Chrome RGB (Classic Chrome image) in [0, 1]
- `dr, dg, db` = Difference (stored but not used in LUT building)

---

## Complete Workflow

### Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  For each of 8 image pairs:                                     │
│  1. Load Standard and Classic Chrome images                     │
│  2. Convert each pixel to LAB space                             │
│  3. Assign pixels to 8×8×8 LAB buckets                          │
│  4. Sample max 200 pixels per bucket (random shuffle)           │
│  5. Accumulate selected pixels                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Write all ~103,000 samples to CSV:                             │
│  outputs/pixel_comparison.csv                                   │
│  Each row: (source_RGB, target_RGB) correspondence              │
└─────────────────────────────────────────────────────────────────┘
```

### Fixed Seed for Reproducibility

**Constant:**

```rust
let mut rng = StdRng::seed_from_u64(42);
```

**Why it matters:**
- Same input images → same CSV output (deterministic)
- Allows exact reproduction of results
- Critical for debugging and validation
- Seed 42 chosen arbitrarily (Douglas Adams reference)

---

## Mathematical Properties

### 1. Coverage Uniformity

**Definition:** Each region of LAB space has equal representation in training data, regardless of natural frequency.

**Proof sketch:**
- Let $p_b$ = probability of pixel falling in bucket $b$ in natural images
- Without stratification: $\text{samples}_b \propto p_b \times N$ (frequent colors dominate)
- With stratification: $\text{samples}_b \leq 200$ (bounded, independent of $p_b$)

### 2. Perceptual Uniformity

**Property:** Buckets correspond to perceptually similar colors

LAB is designed so Euclidean distance approximates perceived color difference:

$$
\Delta E_{ab} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2} \approx \text{perceived difference}
$$

Pixels in the same bucket have:

$$
\begin{aligned}
\Delta L^* &< 12.5 \\
\Delta a^* &< 32 \\
\Delta b^* &< 32 \\
\Rightarrow \Delta E_{ab} &< \sqrt{12.5^2 + 32^2 + 32^2} \approx 45.8
\end{aligned}
$$

This is a reasonable grouping for "similar" colors.

### 3. Sample Efficiency

**Comparison with naive sampling:**

| Method | Sky pixels | Rare color pixels | Problem |
|--------|-----------|-------------------|---------|
| Naive (uniform random) | ~30,000 | ~10 | Redundant sky, missing rare colors |
| **Stratified LAB** | ~200 | ~200 | **Balanced coverage** ✓ |

**Efficiency metric:**

$$
\text{Effective coverage} = \frac{\text{Number of unique buckets used}}{\text{Total possible buckets (512)}}
$$

Stratified sampling maximizes coverage with minimal redundancy.

### 4. Bias Reduction

**Goal:** Training data should represent all colors equally

**Entropy of bucket distribution:**

$$
H = -\sum_{b=1}^{B} p_b \log_2 p_b
$$

Where $p_b = \frac{\text{samples}_b}{\sum_b \text{samples}_b}$

- **Uniform distribution** (stratified): $H \approx \log_2(B)$ (maximum entropy)
- **Skewed distribution** (naive): $H \ll \log_2(B)$ (low entropy, biased)

Higher entropy → better color space coverage → more robust LUT

---

## Implementation Notes

### PixelData Structure

```rust
#[derive(Serialize)]
struct PixelData {
  index: usize,     // Pixel index in original image
  sr: f32,          // Source Red [0, 1]
  sg: f32,          // Source Green [0, 1]
  sb: f32,          // Source Blue [0, 1]
  cr: f32,          // Chrome Red [0, 1]
  cg: f32,          // Chrome Green [0, 1]
  cb: f32,          // Chrome Blue [0, 1]
  dr: f32,          // Difference Red (not used)
  dg: f32,          // Difference Green (not used)
  db: f32,          // Difference Blue (not used)
}
```

**Note:** The difference fields (`dr`, `dg`, `db`) are stored but not used in LUT building. They could be useful for analysis or alternative training methods.

### RGB Normalization

**Formula:**

$$
\text{normalized\_value} = \frac{\text{pixel\_value}}{255}, \quad \text{pixel\_value} \in [0, 255]
$$

**Result:** RGB values in [0, 1] range

This matches the LUT's expected input/output range (32-bit float [0, 1]).

---

## Summary

**Stratified LAB Sampling** is the foundation of the Classic Chrome LUT workflow:

1. ✅ **Converts RGB to perceptually uniform LAB space**
2. ✅ **Divides LAB into 512 buckets (8×8×8 grid)**
3. ✅ **Samples max 200 pixels per bucket (prevents oversampling common colors)**
4. ✅ **Uses fixed seed (42) for reproducibility**
5. ✅ **Processes 8 image pairs → ~103,000 balanced samples**
6. ✅ **Outputs CSV with source→target RGB correspondences**

This ensures the LUT will generalize well across the entire color spectrum, not just the most frequent colors in the training images.

**Next step:** Use `outputs/pixel_comparison.csv` to build the 3D LUT with IDW interpolation (`build_lut.rs`)
