use anyhow::Result;
use csv::Reader;
use opencv::prelude::*;
use opencv::{core, imgcodecs};
use std::fs::File;
use std::io::{BufRead, BufReader};

// Hardcoded color matrix from Step 3 (90-image training)
const COLOR_MATRIX: [[f32; 3]; 3] = [
  [0.90001, 0.06628, 0.05371],
  [0.15345, 0.90884, 0.22182],
  [-0.05292, 0.03634, 0.72899],
];

// Rec.709 luminance coefficients
const LUM_R: f32 = 0.2126;
const LUM_G: f32 = 0.7152;
const LUM_B: f32 = 0.0722;

// Tone curve size
const TONE_BINS: usize = 256;

// LUT size
const LUT_SIZE: usize = 33;

fn main() -> Result<()> {
  println!("=== Step 10-11: Apply Full Pipeline (33³ Residual LUT) ===\n");

  // Define all LUT files to process
  let lut_methods = vec![
    ("33", "residual_lut_33.cube", "final_clone_33.jpg"),
  ];

  // Load tone curve (shared by all methods)
  println!("Loading tone curve from outputs/second_method/tone_curve.csv...");
  let tone_curve = load_tone_curve("outputs/second_method/tone_curve.csv")?;
  println!("Loaded {} tone curve bins", tone_curve.len());

  // Print pipeline parameters
  println!("\n=== Pipeline Parameters ===");
  println!("Color Matrix:");
  for row in &COLOR_MATRIX {
    println!("  [{:8.5}, {:8.5}, {:8.5}]", row[0], row[1], row[2]);
  }

  println!("\nTone curve (first 5 entries):");
  for i in 0..5.min(tone_curve.len()) {
    println!("  [{}] = {:.6}", i, tone_curve[i]);
  }

  // Process test images 91-94
  for img_num in 91..=94 {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📸 Processing Test Image: {}.JPG", img_num);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Load input image
    let input_path = format!("source/compare/standard/{}.JPG", img_num);
    println!("Loading input image from {}...", input_path);
    let input_img = imgcodecs::imread(&input_path, imgcodecs::IMREAD_COLOR)?;
    println!("Image size: {}x{}", input_img.cols(), input_img.rows());

    // Process each method
    println!("\n=== Processing All Methods ===");
    for (method_name, lut_file, _) in &lut_methods {
      println!("\n--- Method: {} ---", method_name.to_uppercase());

      // Load residual LUT
      let lut_path = format!("outputs/second_method/{}", lut_file);
      println!("Loading {}...", lut_path);
      let residual_lut = load_cube_lut(&lut_path)?;
      println!("Loaded LUT with {} entries ({}^3)", residual_lut.len(), LUT_SIZE);

      // Process image
      println!("Processing image...");
      let output_img = process_image(&input_img, &tone_curve, &residual_lut)?;

      // Save output with image number and method name
      let output_path = format!("outputs/second_method/final_clone_{}_{}.jpg", img_num, method_name);
      println!("Saving to {}...", output_path);
      imgcodecs::imwrite(&output_path, &output_img, &core::Vector::new())?;
      println!("✓ Saved {}", output_path);
    }
  }

  println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  println!("=== Complete ===");
  println!("Processed 4 test images × {} interpolation methods = {} outputs", lut_methods.len(), 4 * lut_methods.len());
  println!("All outputs saved to outputs/second_method/");

  Ok(())
}

fn load_tone_curve(path: &str) -> Result<Vec<f32>> {
  let mut reader = Reader::from_path(path)?;
  let mut curve = Vec::with_capacity(TONE_BINS);

  for result in reader.records() {
    let record = result?;
    let value: f32 = record[1].parse()?;
    curve.push(value);
  }

  Ok(curve)
}

fn load_cube_lut(path: &str) -> Result<Vec<[f32; 3]>> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let mut lut = Vec::new();

  for line in reader.lines() {
    let line = line?;
    let line = line.trim();

    // Skip header lines
    if line.starts_with("TITLE")
      || line.starts_with("LUT_3D_SIZE")
      || line.starts_with("DOMAIN_MIN")
      || line.starts_with("DOMAIN_MAX")
      || line.is_empty()
    {
      continue;
    }

    // Parse RGB triplet
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 3 {
      let r: f32 = parts[0].parse()?;
      let g: f32 = parts[1].parse()?;
      let b: f32 = parts[2].parse()?;
      lut.push([r, g, b]);
    }
  }

  Ok(lut)
}

fn process_image(input: &Mat, tone_curve: &[f32], residual_lut: &[[f32; 3]]) -> Result<Mat> {
  let rows = input.rows();
  let cols = input.cols();
  let mut output = input.clone();

  for y in 0..rows {
    for x in 0..cols {
      // Get BGR pixel (OpenCV uses BGR order)
      let pixel = input.at_2d::<core::Vec3b>(y, x)?;

      // Convert to [0, 1] range and RGB order
      let rgb_f32 = [
        pixel[2] as f32 / 255.0, // R (from BGR[2])
        pixel[1] as f32 / 255.0, // G
        pixel[0] as f32 / 255.0, // B (from BGR[0])
      ];

      // Apply full pipeline (RGB order)
      let final_rgb = apply_pipeline(rgb_f32, tone_curve, residual_lut);

      // Convert back to u8 and BGR order
      let output_pixel = output.at_2d_mut::<core::Vec3b>(y, x)?;
      output_pixel[2] = (final_rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8; // R -> BGR[2]
      output_pixel[1] = (final_rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8; // G
      output_pixel[0] = (final_rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8; // B -> BGR[0]
    }
  }

  Ok(output)
}

fn apply_pipeline(rgb: [f32; 3], tone_curve: &[f32], residual_lut: &[[f32; 3]]) -> [f32; 3] {
  // Step 1: Apply color matrix
  let matrix_rgb = apply_matrix(&COLOR_MATRIX, rgb);

  // Step 2: Apply tone curve
  let tone_rgb = apply_tone_curve(matrix_rgb, tone_curve);

  // Step 3: Apply residual LUT
  let final_rgb = apply_residual_lut(tone_rgb, residual_lut);

  final_rgb
}

fn apply_matrix(matrix: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
  let r = matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2];
  let g = matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2];
  let b = matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2];

  [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

fn apply_tone_curve(rgb: [f32; 3], tone_curve: &[f32]) -> [f32; 3] {
  // Compute luminance of matrix output
  let y_old = LUM_R * rgb[0] + LUM_G * rgb[1] + LUM_B * rgb[2];

  // Lookup corrected luminance using linear interpolation
  let pos = y_old.clamp(0.0, 1.0) * (TONE_BINS - 1) as f32;
  let i0 = pos.floor() as usize;
  let i1 = (i0 + 1).min(TONE_BINS - 1);
  let t = pos - i0 as f32;

  let y_new = tone_curve[i0] * (1.0 - t) + tone_curve[i1] * t;

  // Preserve chroma by scaling
  let scale = if y_old > 1e-6 { y_new / y_old } else { 1.0 };

  [
    (rgb[0] * scale).clamp(0.0, 1.0),
    (rgb[1] * scale).clamp(0.0, 1.0),
    (rgb[2] * scale).clamp(0.0, 1.0),
  ]
}

fn apply_residual_lut(rgb: [f32; 3], lut: &[[f32; 3]]) -> [f32; 3] {
  // Sample residual using trilinear interpolation
  let residual = sample_lut_trilinear(rgb, lut);

  // Add residual to RGB
  [
    (rgb[0] + residual[0]).clamp(0.0, 1.0),
    (rgb[1] + residual[1]).clamp(0.0, 1.0),
    (rgb[2] + residual[2]).clamp(0.0, 1.0),
  ]
}

fn sample_lut_trilinear(rgb: [f32; 3], lut: &[[f32; 3]]) -> [f32; 3] {
  // Convert RGB to LUT coordinates
  let x = rgb[0].clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
  let y = rgb[1].clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
  let z = rgb[2].clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;

  // Get surrounding cube corners
  let x0 = x.floor() as usize;
  let y0 = y.floor() as usize;
  let z0 = z.floor() as usize;

  let x1 = (x0 + 1).min(LUT_SIZE - 1);
  let y1 = (y0 + 1).min(LUT_SIZE - 1);
  let z1 = (z0 + 1).min(LUT_SIZE - 1);

  // Get fractional parts
  let xd = x - x0 as f32;
  let yd = y - y0 as f32;
  let zd = z - z0 as f32;

  // Sample the 8 corners
  let c000 = lut[get_lut_index(x0, y0, z0)];
  let c001 = lut[get_lut_index(x0, y0, z1)];
  let c010 = lut[get_lut_index(x0, y1, z0)];
  let c011 = lut[get_lut_index(x0, y1, z1)];
  let c100 = lut[get_lut_index(x1, y0, z0)];
  let c101 = lut[get_lut_index(x1, y0, z1)];
  let c110 = lut[get_lut_index(x1, y1, z0)];
  let c111 = lut[get_lut_index(x1, y1, z1)];

  // Trilinear interpolation
  let mut result = [0.0f32; 3];
  for i in 0..3 {
    let c00 = c000[i] * (1.0 - xd) + c100[i] * xd;
    let c01 = c001[i] * (1.0 - xd) + c101[i] * xd;
    let c10 = c010[i] * (1.0 - xd) + c110[i] * xd;
    let c11 = c011[i] * (1.0 - xd) + c111[i] * xd;

    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    result[i] = c0 * (1.0 - zd) + c1 * zd;
  }

  result
}

fn get_lut_index(r: usize, g: usize, b: usize) -> usize {
  b * LUT_SIZE * LUT_SIZE + g * LUT_SIZE + r
}
