use anyhow::Result;
use csv::Reader;
use opencv::prelude::*;
use opencv::{core, imgcodecs};
use std::fs::File;
use std::io::{BufRead, BufReader};

// Color matrix from 90-image training
const COLOR_MATRIX: [[f32; 3]; 3] = [
  [0.90001, 0.06628, 0.05371],
  [0.15345, 0.90884, 0.22182],
  [-0.05292, 0.03634, 0.72899],
];

const TONE_BINS: usize = 256;
const LUT_SIZE: usize = 17;

fn main() -> Result<()> {
  println!("=== Applying Per-Channel Pipeline ===\n");

  // Load per-channel tone curves
  println!("Loading per-channel tone curves...");
  let (tone_r, tone_g, tone_b) = load_perchannel_tone_curves("outputs/second_method/tone_curve_perchannel.csv")?;
  println!("Loaded {} tone curve bins per channel", tone_r.len());

  // Load residual LUT
  println!("Loading residual LUT...");
  let residual_lut = load_cube_lut("outputs/second_method/residual_lut_perchannel.cube")?;
  println!("Loaded LUT with {} entries ({}^3)", residual_lut.len(), LUT_SIZE);

  // Process test images
  for img_num in 91..=94 {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📸 Processing Image {}.JPG", img_num);

    let input_path = format!("source/compare/standard/{}.JPG", img_num);
    println!("Loading input image from {}...", input_path);
    let input_img = imgcodecs::imread(&input_path, imgcodecs::IMREAD_COLOR)?;
    println!("Image size: {}x{}", input_img.cols(), input_img.rows());

    println!("Processing...");
    let output_img = process_image(&input_img, &tone_r, &tone_g, &tone_b, &residual_lut)?;

    let output_path = format!("outputs/second_method/final_clone_{}_perchannel.jpg", img_num);
    println!("Saving to {}...", output_path);
    imgcodecs::imwrite(&output_path, &output_img, &core::Vector::new())?;
    println!("✓ Saved {}", output_path);
  }

  println!("\n🎉 All images processed!");
  Ok(())
}

fn load_perchannel_tone_curves(path: &str) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
  let mut reader = Reader::from_path(path)?;
  let mut tone_r = Vec::with_capacity(TONE_BINS);
  let mut tone_g = Vec::with_capacity(TONE_BINS);
  let mut tone_b = Vec::with_capacity(TONE_BINS);

  for result in reader.records() {
    let record = result?;
    tone_r.push(record[1].parse()?);
    tone_g.push(record[2].parse()?);
    tone_b.push(record[3].parse()?);
  }

  Ok((tone_r, tone_g, tone_b))
}

fn load_cube_lut(path: &str) -> Result<Vec<[f32; 3]>> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let mut lut = Vec::new();

  for line in reader.lines() {
    let line = line?;
    let line = line.trim();

    if line.starts_with("TITLE")
      || line.starts_with("LUT_3D_SIZE")
      || line.starts_with("DOMAIN_MIN")
      || line.starts_with("DOMAIN_MAX")
      || line.is_empty()
    {
      continue;
    }

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

fn process_image(
  input: &Mat,
  tone_r: &[f32],
  tone_g: &[f32],
  tone_b: &[f32],
  residual_lut: &[[f32; 3]],
) -> Result<Mat> {
  let rows = input.rows();
  let cols = input.cols();
  let mut output = input.clone();

  for y in 0..rows {
    for x in 0..cols {
      let pixel = input.at_2d::<core::Vec3b>(y, x)?;

      // Step 1: Convert BGR to RGB [0, 1]
      let r = pixel[2] as f32 / 255.0;
      let g = pixel[1] as f32 / 255.0;
      let b = pixel[0] as f32 / 255.0;

      // Step 2: Apply color matrix
      let mr = (COLOR_MATRIX[0][0] * r + COLOR_MATRIX[0][1] * g + COLOR_MATRIX[0][2] * b).clamp(0.0, 1.0);
      let mg = (COLOR_MATRIX[1][0] * r + COLOR_MATRIX[1][1] * g + COLOR_MATRIX[1][2] * b).clamp(0.0, 1.0);
      let mb = (COLOR_MATRIX[2][0] * r + COLOR_MATRIX[2][1] * g + COLOR_MATRIX[2][2] * b).clamp(0.0, 1.0);

      // Step 3: Apply per-channel tone curves
      let tr = apply_curve(mr, tone_r);
      let tg = apply_curve(mg, tone_g);
      let tb = apply_curve(mb, tone_b);

      // Step 4: Lookup residual from 3D LUT with trilinear interpolation
      let residual = lookup_lut_trilinear(tr, tg, tb, residual_lut);

      // Step 5: Add residual
      let fr = (tr + residual[0]).clamp(0.0, 1.0);
      let fg = (tg + residual[1]).clamp(0.0, 1.0);
      let fb = (tb + residual[2]).clamp(0.0, 1.0);

      // Convert back to BGR [0, 255]
      let out_pixel = output.at_2d_mut::<core::Vec3b>(y, x)?;
      out_pixel[2] = (fr * 255.0).round() as u8;
      out_pixel[1] = (fg * 255.0).round() as u8;
      out_pixel[0] = (fb * 255.0).round() as u8;
    }
  }

  Ok(output)
}

fn apply_curve(value: f32, curve: &[f32]) -> f32 {
  let pos = value.clamp(0.0, 1.0) * (TONE_BINS - 1) as f32;
  let i0 = pos.floor() as usize;
  let i1 = (i0 + 1).min(TONE_BINS - 1);
  let t = pos - i0 as f32;
  
  (curve[i0] * (1.0 - t) + curve[i1] * t).clamp(0.0, 1.0)
}

fn lookup_lut_trilinear(r: f32, g: f32, b: f32, lut: &[[f32; 3]]) -> [f32; 3] {
  let r_pos = r.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
  let g_pos = g.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
  let b_pos = b.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;

  let r0 = r_pos.floor() as usize;
  let g0 = g_pos.floor() as usize;
  let b0 = b_pos.floor() as usize;

  let r1 = (r0 + 1).min(LUT_SIZE - 1);
  let g1 = (g0 + 1).min(LUT_SIZE - 1);
  let b1 = (b0 + 1).min(LUT_SIZE - 1);

  let dr = r_pos - r0 as f32;
  let dg = g_pos - g0 as f32;
  let db = b_pos - b0 as f32;

  // Get 8 corner values
  let c000 = lut[r0 + g0 * LUT_SIZE + b0 * LUT_SIZE * LUT_SIZE];
  let c001 = lut[r0 + g0 * LUT_SIZE + b1 * LUT_SIZE * LUT_SIZE];
  let c010 = lut[r0 + g1 * LUT_SIZE + b0 * LUT_SIZE * LUT_SIZE];
  let c011 = lut[r0 + g1 * LUT_SIZE + b1 * LUT_SIZE * LUT_SIZE];
  let c100 = lut[r1 + g0 * LUT_SIZE + b0 * LUT_SIZE * LUT_SIZE];
  let c101 = lut[r1 + g0 * LUT_SIZE + b1 * LUT_SIZE * LUT_SIZE];
  let c110 = lut[r1 + g1 * LUT_SIZE + b0 * LUT_SIZE * LUT_SIZE];
  let c111 = lut[r1 + g1 * LUT_SIZE + b1 * LUT_SIZE * LUT_SIZE];

  // Trilinear interpolation
  let mut result = [0.0f32; 3];
  for i in 0..3 {
    let c00 = c000[i] * (1.0 - dr) + c100[i] * dr;
    let c01 = c001[i] * (1.0 - dr) + c101[i] * dr;
    let c10 = c010[i] * (1.0 - dr) + c110[i] * dr;
    let c11 = c011[i] * (1.0 - dr) + c111[i] * dr;

    let c0 = c00 * (1.0 - dg) + c10 * dg;
    let c1 = c01 * (1.0 - dg) + c11 * dg;

    result[i] = c0 * (1.0 - db) + c1 * db;
  }

  result
}
