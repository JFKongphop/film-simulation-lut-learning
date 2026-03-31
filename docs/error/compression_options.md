# Image Compression Options

## Option 1: PNG Compression (Implemented) ✓

**Status:** Already added to the code  
**Compression:** Lossless, level 9 (maximum)  
**File size reduction:** ~20-40%  
**Quality:** No loss  

```rust
let mut png_params = core::Vector::new();
png_params.push(imgcodecs::IMWRITE_PNG_COMPRESSION);
png_params.push(9); // 0-9, higher = more compression
```

---

## Option 2: Resize Images (Lossy)

Add this function and resize before saving:

```rust
fn resize_for_paper(img: &core::Mat, scale: f64) -> Result<core::Mat> {
  let size = img.size()?;
  let new_size = core::Size::new(
    (size.width as f64 * scale) as i32,
    (size.height as f64 * scale) as i32,
  );
  
  let mut resized = core::Mat::default();
  imgproc::resize(
    img,
    &mut resized,
    new_size,
    0.0,
    0.0,
    imgproc::INTER_AREA, // Best for downscaling
  )?;
  
  Ok(resized)
}
```

**Usage:**
```rust
// Resize to 50% before saving (4× smaller file)
let error_map_m1_small = resize_for_paper(&error_map_m1, 0.5)?;
imgcodecs::imwrite("outputs/error/error_map_method1_jet.png", &error_map_m1_small, &png_params)?;
```

**File size reduction:** 75% (at 50% scale)  
**Best for:** Papers (don't need full 39MP resolution)

---

## Option 3: Save as JPEG (Lossy)

Change file extension to `.jpg` and use JPEG quality:

```rust
// JPEG with 85% quality (good balance)
let mut jpeg_params = core::Vector::new();
jpeg_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
jpeg_params.push(85); // 0-100, higher = better quality

imgcodecs::imwrite(
  "outputs/error/error_map_method1_jet.jpg",  // .jpg instead of .png
  &error_map_m1,
  &jpeg_params,
)?;
```

**File size reduction:** 80-90%  
**Quality:** Slight artifacts (acceptable for papers)  
**Recommended quality:** 85 (good balance), 95 (high quality)

---

## Option 4: Combined (Resize + JPEG)

Best compression for papers:

```rust
// Resize to 50% and save as JPEG 85%
let error_map_small = resize_for_paper(&error_map_m1, 0.5)?;

let mut jpeg_params = core::Vector::new();
jpeg_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
jpeg_params.push(85);

imgcodecs::imwrite(
  "outputs/error/error_map_method1_jet.jpg",
  &error_map_small,
  &jpeg_params,
)?;
```

**File size reduction:** 95% (from 70MB → 3-4MB)  
**Best for:** Paper submissions with file size limits

---

## Comparison Table

| Method | Original | PNG-9 | 50% Resize | JPEG-85 | Resize+JPEG |
|--------|----------|-------|------------|---------|-------------|
| **Size** | 70 MB | ~40 MB | ~17 MB | ~7 MB | ~3 MB |
| **Quality** | Perfect | Perfect | Slight blur | Slight artifacts | More blur |
| **Lossless** | Yes | Yes | No | No | No |
| **Use for** | Archive | Archive | Papers | Papers | Web/preview |

---

## Recommendation by Use Case

**For paper submission (1/4 page figure):**
- Use **Option 4** (Resize 50% + JPEG 85%)
- 3-4 MB per image is acceptable
- Readers won't notice difference at print resolution

**For supplementary materials:**
- Use **Option 1** (PNG-9) - already implemented
- Keep high quality for detailed analysis

**For repository/GitHub:**
- Add `.gitignore` entry: `outputs/error/*.png`
- Only commit the code, not the large images
- Let users regenerate locally

---

## Implementation: Full Example

Add this to your `main()` function for multiple outputs:

```rust
fn main() -> Result<()> {
  // ... existing code ...
  
  // Compression parameters
  let mut png_params = core::Vector::new();
  png_params.push(imgcodecs::IMWRITE_PNG_COMPRESSION);
  png_params.push(9);
  
  let mut jpeg_params = core::Vector::new();
  jpeg_params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
  jpeg_params.push(85);
  
  // Save full-res PNG (for archive)
  imgcodecs::imwrite("outputs/error/error_map_method1_jet_full.png", &error_map_m1, &png_params)?;
  
  // Save paper-ready version (50% size, JPEG)
  let error_map_paper = resize_for_paper(&error_map_m1, 0.5)?;
  imgcodecs::imwrite("outputs/error/error_map_method1_jet_paper.jpg", &error_map_paper, &jpeg_params)?;
  
  println!("✓ Full resolution: error_map_method1_jet_full.png");
  println!("✓ Paper version: error_map_method1_jet_paper.jpg");
  
  Ok(())
}
```

---

## External Post-Processing (Alternative)

If you don't want to modify the Rust code, use ImageMagick after generation:

```bash
# Resize to 50% and compress
convert outputs/error/error_map_method1_jet.png -resize 50% -quality 85 outputs/error/error_map_method1_jet_small.jpg

# Batch process all images
for f in outputs/error/*.png; do
  convert "$f" -resize 50% -quality 85 "${f%.png}_small.jpg"
done
```

Or use `optipng` for lossless PNG optimization:
```bash
optipng -o7 outputs/error/*.png
```
