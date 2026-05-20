use anyhow::Result;
use csv::Reader;
use serde::Deserialize;
use std::fs::File;
use std::io::Write;

const LUT_SIZE: usize = 17;
const LUT_TOTAL: usize = LUT_SIZE * LUT_SIZE * LUT_SIZE;

#[derive(Debug, Deserialize)]
struct ResidualRow {
  #[allow(dead_code)]
  sr: f32,
  #[allow(dead_code)]
  sg: f32,
  #[allow(dead_code)]
  sb: f32,
  #[allow(dead_code)]
  cr: f32,
  #[allow(dead_code)]
  cg: f32,
  #[allow(dead_code)]
  cb: f32,
  #[allow(dead_code)]
  mr: f32,
  #[allow(dead_code)]
  mg: f32,
  #[allow(dead_code)]
  mb: f32,
  tr: f32,
  tg: f32,
  tb: f32,
  #[allow(dead_code)]
  y_matrix: f32,
  #[allow(dead_code)]
  y_target: f32,
  rr: f32,
  rg: f32,
  rb: f32,
}

fn main() -> Result<()> {
  println!("=== Step 8-9: Build Residual 3D LUT (Inverse Distance Weighting) ===\n");

  // Initialize accumulation arrays
  let mut sum_r = vec![vec![vec![0.0f32; LUT_SIZE]; LUT_SIZE]; LUT_SIZE];
  let mut sum_g = vec![vec![vec![0.0f32; LUT_SIZE]; LUT_SIZE]; LUT_SIZE];
  let mut sum_b = vec![vec![vec![0.0f32; LUT_SIZE]; LUT_SIZE]; LUT_SIZE];
  let mut count = vec![vec![vec![0u32; LUT_SIZE]; LUT_SIZE]; LUT_SIZE];

  // Step 8: Read CSV and accumulate residuals
  println!("Reading outputs/second_method/matrix_tone_residual.csv...");
  let mut reader = Reader::from_path("outputs/second_method/matrix_tone_residual.csv")?;
  let mut total_pixels = 0;

  for result in reader.deserialize() {
    let row: ResidualRow = result?;

    let tr = row.tr.clamp(0.0, 1.0);
    let tg = row.tg.clamp(0.0, 1.0);
    let tb = row.tb.clamp(0.0, 1.0);

    let ix = (tr * (LUT_SIZE - 1) as f32).round() as usize;
    let iy = (tg * (LUT_SIZE - 1) as f32).round() as usize;
    let iz = (tb * (LUT_SIZE - 1) as f32).round() as usize;

    // Accumulate residuals
    sum_r[ix][iy][iz] += row.rr;
    sum_g[ix][iy][iz] += row.rg;
    sum_b[ix][iy][iz] += row.rb;
    count[ix][iy][iz] += 1;

    total_pixels += 1;
  }

  println!("Processed {} pixels\n", total_pixels);

  // Count occupied cells
  let mut occupied_cells = 0;
  for ix in 0..LUT_SIZE {
    for iy in 0..LUT_SIZE {
      for iz in 0..LUT_SIZE {
        if count[ix][iy][iz] > 0 {
          occupied_cells += 1;
        }
      }
    }
  }

  let empty_cells = LUT_TOTAL - occupied_cells;
  println!("LUT statistics:");
  println!("  Total LUT cells:     {}", LUT_TOTAL);
  println!("  Occupied cells:      {}", occupied_cells);
  println!("  Empty cells:         {}", empty_cells);

  // Step 9: Average the accumulated values
  println!("\nAveraging occupied cells...");
  let mut lut = vec![[0.0f32; 3]; LUT_TOTAL];

  for ix in 0..LUT_SIZE {
    for iy in 0..LUT_SIZE {
      for iz in 0..LUT_SIZE {
        let idx = get_lut_index(ix, iy, iz);

        if count[ix][iy][iz] > 0 {
          let n = count[ix][iy][iz] as f32;
          lut[idx] = [
            sum_r[ix][iy][iz] / n,
            sum_g[ix][iy][iz] / n,
            sum_b[ix][iy][iz] / n,
          ];
        }
      }
    }
  }

  // Fill empty cells using inverse distance weighting
  println!("Filling empty cells using inverse distance weighting...");
  fill_empty_cells_idw(&mut lut, &count);

  // Save as .cube file
  println!("\nSaving to outputs/second_method/residual_lut_idw.cube...");
  save_cube_file(&lut, "outputs/second_method/residual_lut_idw.cube")?;

  println!("\n=== Summary ===");
  println!("Method: Inverse Distance Weighting");
  println!("Total LUT size:     {} ({}^3)", LUT_TOTAL, LUT_SIZE);
  println!("Occupied cells:     {}", occupied_cells);
  println!("Filled cells:       {}", empty_cells);
  println!(
    "Coverage:           {:.2}%",
    occupied_cells as f32 / LUT_TOTAL as f32 * 100.0
  );

  Ok(())
}

fn get_lut_index(r: usize, g: usize, b: usize) -> usize {
  b * LUT_SIZE * LUT_SIZE + g * LUT_SIZE + r
}

fn get_lut_coords(idx: usize) -> (usize, usize, usize) {
  let b = idx / (LUT_SIZE * LUT_SIZE);
  let remainder = idx % (LUT_SIZE * LUT_SIZE);
  let g = remainder / LUT_SIZE;
  let r = remainder % LUT_SIZE;
  (r, g, b)
}

fn fill_empty_cells_idw(lut: &mut [[f32; 3]], count: &[Vec<Vec<u32>>]) {
  let mut filled_count = 0;

  for idx in 0..LUT_TOTAL {
    let (r, g, b) = get_lut_coords(idx);

    // Skip if already occupied
    if count[r][g][b] > 0 {
      continue;
    }

    // Search for non-empty neighbors with increasing radius
    let mut found = false;
    for radius in 1..=LUT_SIZE {
      let mut weighted_sum = [0.0f32; 3];
      let mut weight_sum = 0.0f32;

      // Search all cells within this radius
      for di in -(radius as i32)..=(radius as i32) {
        for dj in -(radius as i32)..=(radius as i32) {
          for dk in -(radius as i32)..=(radius as i32) {
            let ni = r as i32 + di;
            let nj = g as i32 + dj;
            let nk = b as i32 + dk;

            // Skip out-of-bounds
            if ni < 0
              || nj < 0
              || nk < 0
              || ni >= LUT_SIZE as i32
              || nj >= LUT_SIZE as i32
              || nk >= LUT_SIZE as i32
            {
              continue;
            }

            let ni = ni as usize;
            let nj = nj as usize;
            let nk = nk as usize;

            // Skip empty neighbors
            if count[ni][nj][nk] == 0 {
              continue;
            }

            // Compute distance and weight (1/distance)
            let distance = ((di * di + dj * dj + dk * dk) as f32).sqrt();
            if distance == 0.0 {
              continue;
            }

            let weight = 1.0 / distance;

            // Accumulate weighted values
            let neighbor_idx = get_lut_index(ni, nj, nk);
            weighted_sum[0] += lut[neighbor_idx][0] * weight;
            weighted_sum[1] += lut[neighbor_idx][1] * weight;
            weighted_sum[2] += lut[neighbor_idx][2] * weight;
            weight_sum += weight;
          }
        }
      }

      // If we found at least one neighbor, fill the cell
      if weight_sum > 0.0 {
        lut[idx][0] = weighted_sum[0] / weight_sum;
        lut[idx][1] = weighted_sum[1] / weight_sum;
        lut[idx][2] = weighted_sum[2] / weight_sum;
        filled_count += 1;
        found = true;
        break;
      }
    }

    if !found {
      // Fallback: use identity mapping if no neighbors found
      lut[idx][0] = r as f32 / (LUT_SIZE - 1) as f32;
      lut[idx][1] = g as f32 / (LUT_SIZE - 1) as f32;
      lut[idx][2] = b as f32 / (LUT_SIZE - 1) as f32;
      filled_count += 1;
    }
  }

  println!("Filled {} empty cells", filled_count);
}

fn save_cube_file(lut: &[[f32; 3]], path: &str) -> Result<()> {
  let mut file = File::create(path)?;

  // Write header
  writeln!(file, "TITLE \"Residual LUT (IDW)\"")?;
  writeln!(file, "LUT_3D_SIZE {}", LUT_SIZE)?;
  writeln!(file, "DOMAIN_MIN 0.0 0.0 0.0")?;
  writeln!(file, "DOMAIN_MAX 1.0 1.0 1.0")?;
  writeln!(file)?;

  // Write LUT data in nested order: b, g, r
  for b in 0..LUT_SIZE {
    for g in 0..LUT_SIZE {
      for r in 0..LUT_SIZE {
        let idx = get_lut_index(r, g, b);
        let residual = lut[idx];
        writeln!(
          file,
          "{:.6} {:.6} {:.6}",
          residual[0], residual[1], residual[2]
        )?;
      }
    }
  }

  file.flush()?;
  Ok(())
}
