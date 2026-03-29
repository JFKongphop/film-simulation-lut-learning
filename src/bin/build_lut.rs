use anyhow::Result;
use csv::ReaderBuilder;
use opencv::prelude::*;
use opencv::{core, imgproc};
use serde::Deserialize;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct PixelData {
  sr: f32,
  sg: f32,
  sb: f32,
  cr: f32,
  cg: f32,
  cb: f32,
}

const N: usize = 33;

// Calibrated brightness bias correction (LAB L* units)
// This value should be determined from multi-image calibration
// Current value: based on 8-image analysis showing +1.489 bias
// TODO: Update after 100-image calibration
const CALIBRATED_BIAS_L: f32 = 1.489;

fn main() -> Result<()> {
  println!("🎨 Building 3D LUT from CSV data");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  // Step 1: Initialize LUT and COUNT arrays
  let mut lut = vec![vec![vec![[0.0f32; 3]; N]; N]; N];
  let mut count = vec![vec![vec![0u32; N]; N]; N];

  // Step 2: Read CSV and accumulate values
  println!("📖 Reading CSV file...");
  let file = File::open("outputs/pixel_comparison.csv")?;
  let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

  let mut row_count = 0;

  for result in rdr.deserialize() {
    let record: PixelData = result?;
    row_count += 1;

    // Convert source RGB to LUT indices
    let i = ((record.sr * (N - 1) as f32).floor() as usize).min(N - 1);
    let j = ((record.sg * (N - 1) as f32).floor() as usize).min(N - 1);
    let k = ((record.sb * (N - 1) as f32).floor() as usize).min(N - 1);

    // Accumulate target RGB values
    lut[i][j][k][0] += record.cr;
    lut[i][j][k][1] += record.cg;
    lut[i][j][k][2] += record.cb;

    count[i][j][k] += 1;
  }

  println!("✅ Processed {} rows", row_count);

  // Step 3: Average accumulated values
  println!("🧮 Computing averages...");
  let mut filled_cells = 0;
  let mut empty_cells = 0;

  for i in 0..N {
    for j in 0..N {
      for k in 0..N {
        if count[i][j][k] > 0 {
          lut[i][j][k][0] /= count[i][j][k] as f32;
          lut[i][j][k][1] /= count[i][j][k] as f32;
          lut[i][j][k][2] /= count[i][j][k] as f32;
          filled_cells += 1;
        } else {
          empty_cells += 1;
        }
      }
    }
  }

  let total_cells = N * N * N;
  println!("📊 LUT Statistics:");
  println!("   Total cells: {}", total_cells);
  println!("   Filled cells: {} ({:.2}%)", filled_cells, (filled_cells as f64 / total_cells as f64) * 100.0);
  println!("   Empty cells: {} ({:.2}%)", empty_cells, (empty_cells as f64 / total_cells as f64) * 100.0);

  // Show sample counts
  println!("\n📈 Sample distribution:");
  let mut max_samples = 0;
  let mut min_samples = u32::MAX;
  let mut total_samples = 0u64;

  for i in 0..N {
    for j in 0..N {
      for k in 0..N {
        let c = count[i][j][k];
        if c > 0 {
          max_samples = max_samples.max(c);
          min_samples = min_samples.min(c);
          total_samples += c as u64;
        }
      }
    }
  }

  if filled_cells > 0 {
    println!("   Max samples per cell: {}", max_samples);
    println!("   Min samples per cell: {}", min_samples);
    println!("   Avg samples per filled cell: {:.2}", total_samples as f64 / filled_cells as f64);
  }

  // Step 4: Fill empty cells using inverse-distance weighted interpolation
  if empty_cells > 0 {
    println!("\n🔧 Filling empty cells with inverse-distance weighted interpolation...");
    let mut filled_count = 0;
    
    for i in 0..N {
      for j in 0..N {
        for k in 0..N {
          // Skip cells that already have data
          if count[i][j][k] > 0 {
            continue;
          }
          
          // Search for non-empty neighbors with increasing radius
          let mut found = false;
          for radius in 1..=N {
            let mut weighted_sum = [0.0f32; 3];
            let mut weight_sum = 0.0f32;
            
            // Search all cells within this radius
            for di in -(radius as i32)..=(radius as i32) {
              for dj in -(radius as i32)..=(radius as i32) {
                for dk in -(radius as i32)..=(radius as i32) {
                  let ni = i as i32 + di;
                  let nj = j as i32 + dj;
                  let nk = k as i32 + dk;
                  
                  // Skip out-of-bounds
                  if ni < 0 || nj < 0 || nk < 0 || ni >= N as i32 || nj >= N as i32 || nk >= N as i32 {
                    continue;
                  }
                  
                  let ni = ni as usize;
                  let nj = nj as usize;
                  let nk = nk as usize;
                  
                  // Skip empty neighbors
                  if count[ni][nj][nk] == 0 {
                    continue;
                  }
                  
                  // Compute distance and weight
                  let distance = ((di * di + dj * dj + dk * dk) as f32).sqrt();
                  if distance == 0.0 {
                    continue;
                  }
                  
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
              filled_count += 1;
              found = true;
              break;
            }
          }
          
          if !found {
            // Fallback: use identity mapping if no neighbors found
            lut[i][j][k][0] = i as f32 / (N - 1) as f32;
            lut[i][j][k][1] = j as f32 / (N - 1) as f32;
            lut[i][j][k][2] = k as f32 / (N - 1) as f32;
            filled_count += 1;
          }
        }
      }
    }
    
    println!("   ✅ Filled {} empty cells", filled_count);
  }
  
  // Final LUT composition statistics
  println!("\n📊 Final LUT Composition:");
  let cells_from_data = filled_cells;
  let cells_interpolated = if empty_cells > 0 { empty_cells } else { 0 };
  
  println!("   From training data: {} ({:.2}%)", 
    cells_from_data, 
    (cells_from_data as f64 / total_cells as f64) * 100.0);
  println!("   From interpolation: {} ({:.2}%)", 
    cells_interpolated, 
    (cells_interpolated as f64 / total_cells as f64) * 100.0);
  println!("   ─────────────────────────────────");
  println!("   Total completion:   {} (100.00%)", total_cells);

  // Step 5: Apply brightness bias correction
  println!("\n🔧 Applying brightness bias correction...");
  println!("   Correction: {:+.3} LAB L* units", CALIBRATED_BIAS_L);
  apply_brightness_correction(&mut lut)?;
  println!("   ✅ Correction applied to all {} cells", total_cells);

  // Save LUT to file
  println!("\n💾 Writing corrected LUT to file...");
  let output_file = File::create("outputs/lut_33.cube")?;
  write_cube_file(output_file, &lut)?;

  println!("✅ Corrected LUT saved to: outputs/lut_33.cube");

  // Show some sample LUT values
  println!("\n🔍 Sample LUT values:");
  println!("   Black [0,0,0] -> [{:.4}, {:.4}, {:.4}] (count: {})",
    lut[0][0][0][0], lut[0][0][0][1], lut[0][0][0][2], count[0][0][0]);
  println!("   White [{},{},{}] -> [{:.4}, {:.4}, {:.4}] (count: {})",
    N-1, N-1, N-1, lut[N-1][N-1][N-1][0], lut[N-1][N-1][N-1][1], lut[N-1][N-1][N-1][2], count[N-1][N-1][N-1]);
  println!("   Mid [{},{},{}] -> [{:.4}, {:.4}, {:.4}] (count: {})",
    N/2, N/2, N/2, lut[N/2][N/2][N/2][0], lut[N/2][N/2][N/2][1], lut[N/2][N/2][N/2][2], count[N/2][N/2][N/2]);

  Ok(())
}

/// Apply brightness bias correction in LAB space to all LUT cells
fn apply_brightness_correction(lut: &mut Vec<Vec<Vec<[f32; 3]>>>) -> Result<()> {
  let n = lut.len();
  
  for i in 0..n {
    for j in 0..n {
      for k in 0..n {
        let rgb = lut[i][j][k];
        
        // Convert RGB to LAB
        let mut bgr_mat = unsafe { Mat::new_rows_cols(1, 1, core::CV_32FC3)? };
        let pixel = bgr_mat.at_2d_mut::<core::Vec3f>(0, 0)?;
        pixel[0] = rgb[2]; // B
        pixel[1] = rgb[1]; // G
        pixel[2] = rgb[0]; // R
        
        let mut lab_mat = Mat::default();
        imgproc::cvt_color(
          &bgr_mat,
          &mut lab_mat,
          imgproc::COLOR_BGR2Lab,
          0,
          core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        
        let lab_pixel = lab_mat.at_2d_mut::<core::Vec3f>(0, 0)?;
        
        // Apply correction to L* channel
        // OpenCV LAB: L is [0, 100], but stored as float
        lab_pixel[0] = (lab_pixel[0] - CALIBRATED_BIAS_L).clamp(0.0, 100.0);
        
        // Convert back to RGB
        let mut corrected_bgr = Mat::default();
        imgproc::cvt_color(
          &lab_mat,
          &mut corrected_bgr,
          imgproc::COLOR_Lab2BGR,
          0,
          core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        
        let corrected_pixel = corrected_bgr.at_2d::<core::Vec3f>(0, 0)?;
        
        // Update LUT with corrected values (clamped to [0, 1])
        lut[i][j][k][0] = corrected_pixel[2].clamp(0.0, 1.0); // R
        lut[i][j][k][1] = corrected_pixel[1].clamp(0.0, 1.0); // G
        lut[i][j][k][2] = corrected_pixel[0].clamp(0.0, 1.0); // B
      }
    }
  }
  
  Ok(())
}

/// Write LUT in .cube format
fn write_cube_file(mut file: File, lut: &Vec<Vec<Vec<[f32; 3]>>>) -> Result<()> {
  use std::io::Write;

  // Write header
  writeln!(file, "# 3D LUT for Classic Chrome Film Simulation (Bias Corrected)")?;
  writeln!(file, "# Generated from pixel comparison data with {:+.3} L* bias correction", CALIBRATED_BIAS_L)?;
  writeln!(file, "TITLE \"Classic Chrome LUT - Corrected\"")?;
  writeln!(file, "LUT_3D_SIZE {}", N)?;
  writeln!(file)?;

  // Write LUT data in BGR order (Blue changes fastest)
  for r in 0..N {
    for g in 0..N {
      for b in 0..N {
        writeln!(
          file,
          "{:.6} {:.6} {:.6}",
          lut[r][g][b][0], lut[r][g][b][1], lut[r][g][b][2]
        )?;
      }
    }
  }

  Ok(())
}
