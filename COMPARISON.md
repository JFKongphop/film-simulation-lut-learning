# Classic Chrome Film Simulation Methods Comparison

**Test Image**: `9.JPG` (7728×5152, ~39.8 MP)  
**Date**: March 29, 2026

## Overview

Two distinct approaches were evaluated for reverse-engineering Classic Chrome film simulation:

### Method 1: Direct 3D LUT (First Method)
- **Approach**: Build a single 33×33×33 3D LUT by directly mapping stratified samples from standard to Classic Chrome
- **File**: `outputs/lut_33.cube` (35,937 entries)
- **Technique**: Stratified random sampling with trilinear interpolation

### Method 2: Matrix + Tone + Residual Pipeline (Second Method)
- **Approach**: Decompose the transformation into three stages:
  1. Global 3×3 color matrix
  2. 1D tone curve (256 bins)
  3. Residual 3D LUT (17×17×17)
- **Files**: 
  - Color matrix (hardcoded)
  - `outputs/tone_curve.csv` (256 entries)
  - `outputs/residual_lut.cube` (4,913 entries)
- **Technique**: SVD least-squares + luminance tone mapping + residual correction

---

## Quantitative Results

### Image Quality Metrics

| Metric | First Method (LUT) | Second Method (Pipeline) | Winner |
|--------|-------------------|-------------------------|--------|
| **PSNR** | 43.0578 dB | 43.0616 dB | Pipeline (+0.0038) |
| **MSE** | 3.215851 | 3.213105 | Pipeline (-0.002746) |
| **Avg ΔE** | 1.2660 | 1.2073 | Pipeline (-0.0587) |
| **Median ΔE** | 1.2709 | 1.1765 | Pipeline (-0.0944) |
| **Max ΔE** | 18.4314 | 15.7181 | Pipeline (-2.7133) |

### Per-Channel Mean Absolute Error (8-bit scale)

| Channel | First Method | Second Method | Difference |
|---------|-------------|---------------|------------|
| **Blue** | 1.4782 | 1.4982 | +0.0200 |
| **Green** | 1.1880 | 1.2248 | +0.0368 |
| **Red** | 1.3632 | 1.3379 | -0.0253 |

---

## Bias Analysis

### Brightness Bias (Luminance L*)

| Metric | First Method | Second Method |
|--------|-------------|---------------|
| **Mean L* Error** | -0.004 (0-100 scale) | +0.145 (0-100 scale) |
| **L* Bias %** | -0.00% | +0.15% |
| **L* MAE** | 1.136 (8-bit) | 1.184 (8-bit) |
| **Verdict** | ✅ No bias | ✅ No significant bias |

### RGB Channel Bias (Mean Error, 8-bit scale)

| Channel | First Method | Second Method | Interpretation |
|---------|-------------|---------------|----------------|
| **Blue** | -0.056 | +0.329 | LUT slightly darker, Pipeline brighter |
| **Green** | -0.042 | +0.421 | LUT slightly darker, Pipeline brighter |
| **Red** | -0.085 | +0.185 | LUT slightly darker, Pipeline brighter |
| **Overall** | -0.061 (-0.02%) | +0.312 (+0.12%) | Minimal bias in both |

### Color Shift Analysis

| Method | a* (Green-Red) | b* (Blue-Yellow) | Assessment |
|--------|----------------|------------------|------------|
| **First Method** | -0.026 | -0.004 | Neutral (no shift) |
| **Second Method** | -0.133 | +0.022 | Neutral (no shift) |

---

## Luminance Distribution

Both methods show excellent luminance matching with most pixels within ±1 unit difference (0-100 scale).

### First Method Distribution Peak:
- **-1**: 5,447,640 pixels
- **0**: 27,946,195 pixels (peak)
- **+1**: 5,345,273 pixels

### Second Method Distribution Peak:
- **-1**: 3,744,328 pixels
- **0**: 27,158,928 pixels (peak)
- **+1**: 7,567,138 pixels

---

## Quality Assessment

### PSNR Interpretation
Both methods: **Excellent (nearly identical)**
- PSNR ≥ 40 dB indicates excellent quality with only minor perceptible differences

### Delta E (Color Difference) Interpretation
Both methods: **Perceptible through close observation**
- ΔE 1.0-2.0: Differences detectable only under careful scrutiny
- Human eye typically cannot distinguish ΔE < 1.0

---

## Technical Comparison

### Model Size

| Method | Total Parameters | Storage Size |
|--------|-----------------|--------------|
| **First Method** | ~107,811 values (33³×3) | 948 KB |
| **Second Method** | ~4,922 values (9 + 256 + 17³×3) | 138 KB |

**Compression ratio**: 6.9:1 (85% reduction)

### Computational Complexity (per pixel)

| Method | Operations |
|--------|------------|
| **First Method** | 1× trilinear interpolation (8 lookups, 7 lerps) |
| **Second Method** | 1× matrix multiply + 1× linear interpolation + 1× trilinear interpolation |

### Training Data Coverage

| Method | LUT Cells | Occupied Cells | Coverage |
|--------|-----------|----------------|----------|
| **First Method** | 35,937 (33³) | 11,754 | 32.7% |
| **Second Method** | 4,913 (17³) | 1,006 | 20.5% |

---

## Advantages & Disadvantages

### First Method: Direct 3D LUT

**Advantages:**
✅ Simpler pipeline (single transformation)  
✅ Industry-standard format (.cube)  
✅ Compatible with most color grading software  
✅ No systematic brightness bias (-0.00%)  
✅ Slightly lower per-channel errors

**Disadvantages:**
❌ Larger file size (948 KB)  
❌ Less interpretable (black box)  
❌ Cannot separately adjust matrix, tone, or residual  
❌ Requires more training samples for good coverage

### Second Method: Matrix + Tone + Residual

**Advantages:**
✅ **6.9× smaller** model size (138 KB)  
✅ Interpretable components (color matrix, tone curve, residual)  
✅ Mathematically principled (SVD least-squares)  
✅ Editable stages for artistic control  
✅ Better max ΔE (15.7 vs 18.4)  
✅ Slightly better PSNR and average ΔE

**Disadvantages:**
❌ More complex pipeline (3 stages)  
❌ Slight brightness bias (+0.15%, still negligible)  
❌ Per-channel errors marginally higher  
❌ Requires careful implementation of all stages

---

## Conclusions

### Overall Winner: **Second Method (Matrix + Tone + Residual)** 🏆

**Key Findings:**

1. **Quality**: Both methods achieve **excellent results** (PSNR > 43 dB, ΔE ≈ 1.2)
   - Differences are imperceptible to most viewers
   - Second method marginally better in most metrics

2. **Efficiency**: Second method is **85% smaller** while maintaining quality
   - 138 KB vs 948 KB
   - Better for deployment and distribution

3. **Interpretability**: Second method provides insight into the transformation
   - Color matrix reveals cross-channel relationships
   - Tone curve shows luminance mapping
   - Residual LUT captures remaining non-linearities

4. **Bias**: Both methods exhibit minimal bias
   - First method: essentially zero bias
   - Second method: +0.15% brightness bias (negligible)

### Recommendations

**Use First Method when:**
- Working with color grading software (DaVinci Resolve, Adobe)
- Need absolute color accuracy (minimal bias)
- File size is not a concern
- Want maximum compatibility

**Use Second Method when:**
- Need compact model for deployment
- Want artistic control over color, tone, and detail separately
- Building custom color pipeline
- Optimizing for storage/bandwidth
- Need to understand the transformation

### Future Improvements

Both methods could benefit from:
- Training on more diverse images (currently 103,427 pixels from 9 images)
- Higher-order interpolation (cubic/quintic vs linear)
- Exposure-conditioned transformations
- HDR support
- Gamut mapping for different color spaces

---

## Performance Summary

| Aspect | Winner | Margin |
|--------|--------|--------|
| **Image Quality** | Pipeline | Marginal (+0.09%) |
| **Model Size** | Pipeline | Significant (-85%) |
| **Brightness Accuracy** | LUT | Negligible (0.15% diff) |
| **Color Accuracy** | Tie | Both excellent |
| **Interpretability** | Pipeline | Clear winner |
| **Simplicity** | LUT | Fewer steps |
| **Compatibility** | LUT | Industry standard |

**Final Verdict**: The **Matrix + Tone + Residual Pipeline** (Second Method) is the superior approach for most applications, offering comparable quality with dramatically reduced model size and enhanced interpretability.
