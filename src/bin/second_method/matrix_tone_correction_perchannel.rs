use anyhow::Result;
use csv::{Reader, Writer};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::fs::File;

// Tone curve parameters
const TONE_BINS: usize = 256;
const SMOOTH_WINDOW: usize = 5;

#[derive(Debug, Deserialize)]
struct InputRow {
  #[allow(dead_code)]
  index: u32,
  sr: f32,
  sg: f32,
  sb: f32,
  cr: f32,
  cg: f32,
  cb: f32,
  #[allow(dead_code)]
  dr: f32,
  #[allow(dead_code)]
  dg: f32,
  #[allow(dead_code)]
  db: f32,
}


#[derive(Debug, Serialize)]
#[allow(dead_code)]
struct OutputRow {
  sr: f32,
  sg: f32,
  sb: f32,
  cr: f32,
  cg: f32,
  cb: f32,
  mr: f32,
  mg: f32,
  mb: f32,
  tr: f32,
  tg: f32,
  tb: f32,
  rr: f32,
  rg: f32,
  rb: f32,
}

#[derive(Debug, Clone)]
struct Pixel {
  source: [f32; 3],
  target: [f32; 3],
}

fn main() -> Result<()> {
  println!("=== Step 3-7: Matrix + Per-Channel Tone Curves Pipeline ===\n");

  // Load pixel data
  let pixels = load_pixel_data("outputs/pixel_comparison.csv")?;
  println!("Loaded {} pixels\n", pixels.len());

  // Step 3: Solve global 3×3 color matrix (same as before)
  println!("Step 3: Solving global color matrix...");
  let matrix = solve_color_matrix(&pixels)?;
  print_matrix(&matrix);

  // Step 4: Apply matrix to all pixels
  println!("\nStep 4: Applying matrix to all pixels...");
  let matrix_rgb: Vec<[f32; 3]> = pixels
    .iter()
    .map(|p| apply_matrix(&matrix, p.source))
    .collect();

  // Compute error before matrix (source -> target)
  let error_before = compute_mean_error(
    &pixels.iter().map(|p| p.source).collect::<Vec<_>>(),
    &pixels.iter().map(|p| p.target).collect::<Vec<_>>(),
  );
  println!(
    "Mean absolute error (source -> target): {:.6}",
    error_before
  );

  // Compute error after matrix
  let error_after_matrix = compute_mean_error(
    &matrix_rgb,
    &pixels.iter().map(|p| p.target).collect::<Vec<_>>(),
  );
  println!(
    "Mean absolute error (matrix -> target): {:.6}",
    error_after_matrix
  );

  // Step 6: Fit per-channel tone curves
  println!("\nStep 6: Fitting per-channel tone curves (R, G, B)...");
  
  // Extract per-channel data
  let matrix_r: Vec<f32> = matrix_rgb.iter().map(|rgb| rgb[0]).collect();
  let matrix_g: Vec<f32> = matrix_rgb.iter().map(|rgb| rgb[1]).collect();
  let matrix_b: Vec<f32> = matrix_rgb.iter().map(|rgb| rgb[2]).collect();
  
  let target_r: Vec<f32> = pixels.iter().map(|p| p.target[0]).collect();
  let target_g: Vec<f32> = pixels.iter().map(|p| p.target[1]).collect();
  let target_b: Vec<f32> = pixels.iter().map(|p| p.target[2]).collect();
  
  let tone_curve_r = fit_tone_curve(&matrix_r, &target_r, "R");
  let tone_curve_g = fit_tone_curve(&matrix_g, &target_g, "G");
  let tone_curve_b = fit_tone_curve(&matrix_b, &target_b, "B");
  println!("Per-channel tone curves fitted with {} bins each", TONE_BINS);

  // Save tone curves for later use
  save_tone_curves(&tone_curve_r, &tone_curve_g, &tone_curve_b, "outputs/second_method/tone_curve_perchannel.csv")?;

  // Step 7: Apply per-channel tone curves
  println!("\nStep 7: Applying per-channel tone curves...");
  let tone_rgb: Vec<[f32; 3]> = matrix_rgb
    .iter()
    .map(|&rgb| apply_perchannel_tone_curves(&tone_curve_r, &tone_curve_g, &tone_curve_b, rgb))
    .collect();

  // Compute error after tone
  let error_after_tone = compute_mean_error(
    &tone_rgb,
    &pixels.iter().map(|p| p.target).collect::<Vec<_>>(),
  );
  println!(
    "Mean absolute error (matrix+tone -> target): {:.6}",
    error_after_tone
  );

  // Save results
  println!("\nSaving results to outputs/second_method/matrix_tone_residual_perchannel.csv...");
  save_results(&pixels, &matrix_rgb, &tone_rgb)?;

  println!("\n=== Summary ===");
  println!("Error before matrix:      {:.6}", error_before);
  println!("Error after matrix:       {:.6}", error_after_matrix);
  println!("Error after matrix+tone:  {:.6}", error_after_tone);
  println!(
    "Improvement (matrix):     {:.6}",
    error_before - error_after_matrix
  );
  println!(
    "Improvement (tone):       {:.6}",
    error_after_matrix - error_after_tone
  );
  println!(
    "Total improvement:        {:.6}",
    error_before - error_after_tone
  );

  Ok(())
}

fn load_pixel_data(path: &str) -> Result<Vec<Pixel>> {
  let mut reader = Reader::from_path(path)?;
  let mut pixels = Vec::new();

  for result in reader.deserialize() {
    let row: InputRow = result?;
    pixels.push(Pixel {
      source: [row.sr, row.sg, row.sb],
      target: [row.cr, row.cg, row.cb],
    });
  }

  Ok(pixels)
}

fn solve_color_matrix(pixels: &[Pixel]) -> Result<[[f32; 3]; 3]> {
  let n = pixels.len();

  // Build matrices X (source) and Y (target)
  let mut x_data = vec![0.0f64; n * 3];
  let mut y_data = vec![0.0f64; n * 3];

  for (i, pixel) in pixels.iter().enumerate() {
    x_data[i * 3] = pixel.source[0] as f64;
    x_data[i * 3 + 1] = pixel.source[1] as f64;
    x_data[i * 3 + 2] = pixel.source[2] as f64;

    y_data[i * 3] = pixel.target[0] as f64;
    y_data[i * 3 + 1] = pixel.target[1] as f64;
    y_data[i * 3 + 2] = pixel.target[2] as f64;
  }

  let x = DMatrix::from_row_slice(n, 3, &x_data);
  let y = DMatrix::from_row_slice(n, 3, &y_data);

  // Solve M using SVD least-squares
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

fn apply_matrix(matrix: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
  let r = matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2];
  let g = matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2];
  let b = matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2];

  [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}

fn fit_tone_curve(channel_matrix: &[f32], channel_target: &[f32], channel_name: &str) -> Vec<f32> {
  // Initialize bins
  let mut bin_sums = vec![0.0f32; TONE_BINS];
  let mut bin_counts = vec![0u32; TONE_BINS];

  // Accumulate values into bins
  for (&c_m, &c_t) in channel_matrix.iter().zip(channel_target.iter()) {
    let bin_idx = ((c_m * (TONE_BINS - 1) as f32).round() as usize).min(TONE_BINS - 1);
    bin_sums[bin_idx] += c_t;
    bin_counts[bin_idx] += 1;
  }

  // Compute averages for non-empty bins
  let mut curve = vec![0.0f32; TONE_BINS];
  for i in 0..TONE_BINS {
    if bin_counts[i] > 0 {
      curve[i] = bin_sums[i] / bin_counts[i] as f32;
    }
  }

  // Fill empty bins by interpolation
  fill_missing_bins(&mut curve, &bin_counts);

  // Smooth the curve
  smooth_curve(&mut curve, SMOOTH_WINDOW);

  // Enforce monotonicity
  for i in 1..curve.len() {
    curve[i] = curve[i].max(curve[i - 1]);
  }

  // Print representative tone curve values
  println!("{} channel tone curve samples:", channel_name);
  for i in (0..TONE_BINS).step_by(64) {
    println!("  Tone curve_{}[{}] = {:.4}", channel_name, i, curve[i]);
  }

  curve
}

fn fill_missing_bins(curve: &mut [f32], counts: &[u32]) {
  let n = curve.len();

  for i in 0..n {
    if counts[i] == 0 {
      // Find nearest non-empty bins on both sides
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

      // Interpolate
      curve[i] = match (left_val, right_val) {
        (Some(l), Some(r)) => {
          let ld = left_dist.unwrap() as f32;
          let rd = right_dist.unwrap() as f32;
          (l * rd + r * ld) / (ld + rd)
        }
        (Some(l), None) => l,
        (None, Some(r)) => r,
        (None, None) => i as f32 / (n - 1) as f32,
      };
    }
  }
}

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

fn apply_perchannel_tone_curves(
  tone_r: &[f32],
  tone_g: &[f32],
  tone_b: &[f32],
  rgb: [f32; 3],
) -> [f32; 3] {
  // Apply each channel's tone curve independently (no luminance scaling)
  let apply_curve = |value: f32, curve: &[f32]| -> f32 {
    let pos = value.clamp(0.0, 1.0) * (TONE_BINS - 1) as f32;
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(TONE_BINS - 1);
    let t = pos - i0 as f32;
    
    (curve[i0] * (1.0 - t) + curve[i1] * t).clamp(0.0, 1.0)
  };

  [
    apply_curve(rgb[0], tone_r),
    apply_curve(rgb[1], tone_g),
    apply_curve(rgb[2], tone_b),
  ]
}

fn compute_mean_error(predicted: &[[f32; 3]], target: &[[f32; 3]]) -> f32 {
  let n = predicted.len() as f32;
  let sum: f32 = predicted
    .iter()
    .zip(target.iter())
    .map(|(p, t)| (p[0] - t[0]).abs() + (p[1] - t[1]).abs() + (p[2] - t[2]).abs())
    .sum();

  sum / (n * 3.0)
}

fn print_matrix(matrix: &[[f32; 3]; 3]) {
  println!("Color matrix:");
  for row in matrix {
    println!("  [{:8.5}, {:8.5}, {:8.5}]", row[0], row[1], row[2]);
  }
}

fn save_tone_curves(
  tone_r: &[f32],
  tone_g: &[f32],
  tone_b: &[f32],
  path: &str,
) -> Result<()> {
  let file = File::create(path)?;
  let mut writer = Writer::from_writer(file);

  // Write header
  writer.write_record(&["bin", "r", "g", "b"])?;

  // Write data
  for i in 0..TONE_BINS {
    writer.write_record(&[
      i.to_string(),
      tone_r[i].to_string(),
      tone_g[i].to_string(),
      tone_b[i].to_string(),
    ])?;
  }

  writer.flush()?;
  println!("Saved per-channel tone curves to {}", path);
  Ok(())
}

fn save_results(
  pixels: &[Pixel],
  matrix_rgb: &[[f32; 3]],
  tone_rgb: &[[f32; 3]],
) -> Result<()> {
  let file = File::create("outputs/second_method/matrix_tone_residual_perchannel.csv")?;
  let mut writer = Writer::from_writer(file);

  // Write header
  writer.write_record(&["sr", "sg", "sb", "cr", "cg", "cb", "mr", "mg", "mb", "tr", "tg", "tb", "rr", "rg", "rb"])?;

  // Write data
  for i in 0..pixels.len() {
    let p = &pixels[i];
    let m = matrix_rgb[i];
    let t = tone_rgb[i];
    let residual = [p.target[0] - t[0], p.target[1] - t[1], p.target[2] - t[2]];

    writer.write_record(&[
      p.source[0].to_string(),
      p.source[1].to_string(),
      p.source[2].to_string(),
      p.target[0].to_string(),
      p.target[1].to_string(),
      p.target[2].to_string(),
      m[0].to_string(),
      m[1].to_string(),
      m[2].to_string(),
      t[0].to_string(),
      t[1].to_string(),
      t[2].to_string(),
      residual[0].to_string(),
      residual[1].to_string(),
      residual[2].to_string(),
    ])?;
  }

  writer.flush()?;
  Ok(())
}
