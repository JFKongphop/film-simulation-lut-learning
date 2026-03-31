# Error Map Generation

This document explains the mathematical foundation and implementation of the error visualization tool for comparing film simulation methods.

## Overview

The `generate_error_maps.rs` tool generates three types of visualizations to analyze the spatial distribution of color errors between ground truth and reconstructed images:

1. **Jet Colormap** - Continuous color scale for scientific analysis
2. **Custom Colormap** - Categorical zones for interpretability
3. **Amplified Difference** - Raw RGB differences magnified 10×

## Mathematical Foundation

### 1. Delta E (ΔE) - CIE76 Formula

The color difference is calculated in CIELAB color space using the Euclidean distance:

```
ΔE = √[(L₁ - L₂)² + (a₁ - a₂)² + (b₁ - b₂)²]
```

Where:
- `L` = Lightness (0-100)
- `a` = Green-Red axis (-128 to +127)
- `b` = Blue-Yellow axis (-128 to +127)

**Perceptual Meaning:**
- ΔE < 1.0: Imperceptible difference (not noticeable to human eye)
- 1.0 ≤ ΔE < 2.0: Barely perceptible (only noticeable upon close inspection)
- 2.0 ≤ ΔE < 5.0: Noticeable difference (visible but acceptable)
- ΔE ≥ 5.0: Obvious difference (unacceptable for color matching)

### 2. Color Space Conversion: BGR → LAB

OpenCV uses BGR (Blue-Green-Red) format internally, which must be converted to CIELAB for perceptual color difference calculation:

```
BGR → XYZ → LAB
```

The conversion accounts for:
- **Gamma correction** (sRGB → linear RGB)
- **Illuminant D65** (standard daylight)
- **2° standard observer**

### 3. Statistical Metrics

For a set of ΔE values across all pixels:

**Mean:**
```
μ = (1/N) Σ ΔEᵢ
```

**Median:**
```
median = ΔE at position ⌊N/2⌋ in sorted array
```

**Standard Deviation:**
```
σ = √[(1/N) Σ (ΔEᵢ - μ)²]
```

**Maximum:**
```
max = max(ΔE₁, ΔE₂, ..., ΔEₙ)
```

## Implementation Details

### Color Difference Calculation

```rust
fn calculate_delta_e(lab1: &core::Mat, lab2: &core::Mat) -> Result<core::Mat> {
  let size = lab1.size()?;
  let mut delta_e = core::Mat::new_rows_cols_with_default(
    size.height,
    size.width,
    core::CV_32F,  // 32-bit float for precision
    core::Scalar::all(0.0),
  )?;

  for y in 0..size.height {
    for x in 0..size.width {
      let pixel1: core::Vec3b = *lab1.at_2d(y, x)?;
      let pixel2: core::Vec3b = *lab2.at_2d(y, x)?;

      // Calculate differences in each channel
      let dl = pixel1[0] as f32 - pixel2[0] as f32;  // ΔL
      let da = pixel1[1] as f32 - pixel2[1] as f32;  // Δa
      let db = pixel1[2] as f32 - pixel2[2] as f32;  // Δb

      // Euclidean distance in LAB space
      let de = (dl * dl + da * da + db * db).sqrt();
      *delta_e.at_2d_mut(y, x)? = de;
    }
  }

  Ok(delta_e)
}
```

**Key Points:**
- Operates pixel-by-pixel across entire image (7728×5152 = 39.8 megapixels)
- Stores result as 32-bit float to preserve sub-unit precision
- Uses squared differences to avoid negative values

### Visualization 1: Jet Colormap

Maps ΔE values [0, 5] to a continuous rainbow color scale.

**Jet Color Function:**
```rust
fn jet_color(value: f32) -> (u8, u8, u8) {
  // value normalized to [0, 1]
  let r = ((1.5 - 4.0 * (value - 0.75).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  let g = ((1.5 - 4.0 * (value - 0.5).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  let b = ((1.5 - 4.0 * (value - 0.25).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  (r, g, b)
}
```

**Mathematical Mapping:**

The piecewise function creates smooth transitions:

- **Blue (v=0.0):** R=0, G=0, B=255
- **Cyan (v=0.25):** R=0, G=255, B=255 (blue peak at 0.25)
- **Green (v=0.5):** R=0, G=255, B=0 (green peak at 0.5)
- **Yellow (v=0.75):** R=255, G=255, B=0 (red peak at 0.75)
- **Red (v=1.0):** R=255, G=0, B=0

Each channel peaks at its center value and falls off linearly:
```
channel_intensity = max(0, 1.5 - 4|value - center|)
```

**Normalization:**
```rust
let normalized = ((val - min_val) / (max_val - min_val)).clamp(0.0, 1.0);
```
Maps ΔE ∈ [0, 5] → normalized ∈ [0, 1]

### Visualization 2: Custom Colormap

Categorical color zones based on perceptual thresholds.

```rust
fn create_custom_colormap(delta_e: &core::Mat) -> Result<core::Mat> {
  // ...
  let color = if val < 1.0 {
    [255, 100, 0]     // Blue (BGR): Imperceptible
  } else if val < 2.0 {
    [100, 255, 0]     // Green: Barely perceptible
  } else if val < 5.0 {
    [0, 255, 255]     // Yellow: Noticeable
  } else {
    [50, 50, 255]     // Red: Obvious
  };
  // ...
}
```

**Color Encoding (BGR format):**

| ΔE Range | Color | BGR Values | Meaning |
|----------|-------|------------|---------|
| [0, 1) | Blue | (255, 100, 0) | Imperceptible, high quality |
| [1, 2) | Green | (100, 255, 0) | Barely perceptible, acceptable |
| [2, 5) | Yellow | (0, 255, 255) | Noticeable, may need improvement |
| [5, ∞) | Red | (50, 50, 255) | Obvious error, unacceptable |

**Why These Thresholds?**
- Based on JND (Just Noticeable Difference) research
- ΔE=1 is the standard threshold for "same color" in industry
- ΔE=2.3 is ISO standard for acceptable color difference

### Visualization 3: Amplified Difference

Shows raw RGB differences magnified for visibility.

```rust
fn create_amplified_diff(img1: &core::Mat, img2: &core::Mat, amplify: f32) -> Result<core::Mat> {
  let mut diff = core::Mat::default();
  
  // Calculate absolute difference per channel
  core::absdiff(img1, img2, &mut diff)?;
  
  // Amplify and convert to 8-bit
  let mut amplified = core::Mat::default();
  diff.convert_to(&mut amplified, core::CV_8UC3, amplify as f64, 0.0)?;
  
  Ok(amplified)
}
```

**Formula:**
```
diff_amplified = min(255, |img1 - img2| × 10)
```

For each RGB channel independently:
- Original difference might be 1-10 (nearly invisible)
- Multiplied by 10× makes it 10-100 (clearly visible)
- Clamped to [0, 255] to prevent overflow

**Use Cases:**
- Detecting subtle structural patterns
- Finding systematic bias in specific color channels
- Debugging spatial artifacts

### Statistical Analysis

```rust
fn calculate_statistics(delta_e: &core::Mat) -> Result<Statistics> {
  let size = delta_e.size()?;
  let total_pixels = (size.height * size.width) as f64;

  let mut sum = 0.0;
  let mut max_val = 0.0;
  let mut values = Vec::new();

  // Single pass: collect all values
  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)? as f64;
      sum += val;
      if val > max_val {
        max_val = val;
      }
      values.push(val);
    }
  }

  let mean = sum / total_pixels;

  // Sort for median (O(n log n))
  values.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median = values[values.len() / 2];

  // Second pass: variance
  let variance = values.iter()
    .map(|v| (v - mean).powi(2))
    .sum::<f64>() / total_pixels;
  let std_dev = variance.sqrt();

  Ok(Statistics { mean, median, max: max_val, std_dev })
}
```

**Complexity:**
- Time: O(n log n) due to sorting (n = 39.8M pixels)
- Space: O(n) to store all values for median calculation

### Distribution Analysis

Calculates percentage of pixels in each error category:

```rust
fn calculate_distribution(delta_e: &core::Mat) -> Result<(f64, f64, f64, f64)> {
  // ...
  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)?;
      if val < 1.0 {
        count1 += 1;        // Imperceptible
      } else if val < 2.0 {
        count2 += 1;        // Barely perceptible
      } else if val < 5.0 {
        count3 += 1;        // Noticeable
      } else {
        count4 += 1;        // Obvious
      }
    }
  }

  Ok((
    count1 as f64 / total * 100.0,
    count2 as f64 / total * 100.0,
    count3 as f64 / total * 100.0,
    count4 as f64 / total * 100.0,
  ))
}
```

**Output Format:**
```
Method 1 - Pixel distribution:
  ΔE < 1: 33.31%        (imperceptible)
  1 ≤ ΔE < 2: 18.59%    (barely perceptible)
  2 ≤ ΔE < 5: 47.13%    (noticeable)
  ΔE ≥ 5: 0.97%         (obvious)
```

**Quality Interpretation:**
- High % in first two categories → excellent reconstruction
- Most pixels in ΔE < 2 → perceptually lossless
- High % in ΔE ≥ 5 → systematic problems

## Workflow

```
1. Load Images
   ├─ Ground Truth: source/compare/classic-chrome/9.JPG
   ├─ Method 1 Output: outputs/first_method/lut_33.jpg
   └─ Method 2 Output: outputs/second_method/final_clone.jpg

2. Color Space Conversion
   └─ BGR → LAB (OpenCV cvtColor)

3. Calculate ΔE Per-Pixel
   └─ Euclidean distance in LAB space

4. Generate Visualizations
   ├─ Jet colormap (continuous scale)
   ├─ Custom colormap (categorical zones)
   └─ Amplified difference (10× RGB)

5. Compute Statistics
   ├─ Mean, Median, Max, StdDev
   └─ Distribution across thresholds

6. Save Outputs
   └─ outputs/error/*.png (6 files)
```

## Usage

```bash
# Compile and run
cargo run --bin generate_error_maps --release

# Output files:
# - outputs/error/error_map_method1_jet.png
# - outputs/error/error_map_method2_jet.png
# - outputs/error/error_map_method1_custom.png
# - outputs/error/error_map_method2_custom.png
# - outputs/error/amplified_diff_method1.png
# - outputs/error/amplified_diff_method2.png
```

## Results Interpretation

### Statistical Output Example

```
ERROR STATISTICS
============================================================
Method 1 (Direct LUT):
  Mean ΔE: 1.8417
  Median ΔE: 1.7321
  Max ΔE: 47.0000
  Std Dev: 1.1391

Method 2 (Pipeline):
  Mean ΔE: 1.8487
  Median ΔE: 1.7321
  Max ΔE: 40.0125
  Std Dev: 1.1563
============================================================
```

**Analysis:**
- **Mean ≈ 1.84**: Average error is barely perceptible
- **Median < Mean**: Most pixels have lower error (skewed distribution)
- **Max = 47**: Outliers exist but are rare (edge artifacts)
- **StdDev ≈ 1.14**: Errors are tightly clustered around mean

### Visual Patterns to Look For

**Good Reconstruction:**
- Predominantly blue/green in custom colormap
- Uniform distribution (no spatial clustering)
- Low amplified difference intensity

**Poor Reconstruction:**
- Yellow/red regions in custom colormap
- Clustering in specific areas (e.g., shadows, highlights)
- High-contrast patterns in amplified difference

## Performance Considerations

**Image Size:** 7728 × 5152 = 39,829,056 pixels

**Processing Time (Release Build):**
- LAB conversion: ~0.5s
- ΔE calculation: ~2.0s (nested loops, 39M iterations)
- Statistics + sorting: ~1.5s
- Colormap generation: ~6.0s (3 maps × 2 methods)
- **Total: ~10-15 seconds**

**Memory Usage:**
- Input images: 3 × 39.8M × 3 bytes ≈ 360 MB
- LAB images: 3 × 39.8M × 3 bytes ≈ 360 MB
- ΔE maps: 2 × 39.8M × 4 bytes ≈ 320 MB
- Output images: 6 × 39.8M × 3 bytes ≈ 720 MB
- **Peak: ~1.8 GB RAM**

**Optimization Opportunities:**
- Parallel processing with rayon crate
- SIMD vectorization for ΔE calculation
- Streaming processing to reduce memory footprint

## References

1. **CIE76 Delta E Formula:**
   - CIE. (1976). *Colorimetry*. CIE Publication 15.

2. **Perceptual Thresholds:**
   - Mahy, M., Van Eycken, L., & Oosterlinck, A. (1994). "Evaluation of Uniform Color Spaces Developed after the Adoption of CIELAB and CIELUV." *Color Research & Application*, 19(2), 105-121.

3. **Just Noticeable Difference (JND):**
   - MacAdam, D. L. (1942). "Visual Sensitivities to Color Differences in Daylight." *Journal of the Optical Society of America*, 32(5), 247-274.

4. **ISO Standards:**
   - ISO 12647-2:2013 - Graphic technology — Process control for the production of half-tone colour separations
