use anyhow::Result;
use csv::Reader;
use serde::Deserialize;
use std::fs::File;
use std::io::Write;

const LUT_SIZE: usize = 17;
const LUT_TOTAL: usize = LUT_SIZE * LUT_SIZE * LUT_SIZE;
const RBF_EPSILON: f32 = 2.0; // Shape parameter for Gaussian RBF

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
  println!("=== Step 8-9: Build Residual 3D LUT (Radial Basis Function) ===\n");

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

  // Fill empty cells using Radial Basis Function interpolation
  println!("Filling empty cells using RBF (Gaussian, epsilon={})...", RBF_EPSILON);
  fill_empty_cells_rbf(&mut lut, &count);

  // Save as .cube file
  println!("\nSaving to outputs/second_method/residual_lut_rbf.cube...");
  save_cube_file(&lut, "outputs/second_method/residual_lut_rbf.cube")?;

  println!("\n=== Summary ===");
  println!("Method: Radial Basis Function (Gaussian)");
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

// Gaussian Radial Basis Function
fn rbf_gaussian(distance: f32, epsilon: f32) -> f32 {
  (-epsilon * distance * distance).exp()
}

fn fill_empty_cells_rbf(lut: &mut [[f32; 3]], count: &[Vec<Vec<u32>>]) {
  let mut filled_count = 0;

  // Collect all occupied cell positions
  let mut occupied: Vec<(usize, usize, usize)> = Vec::new();
  for ir in 0..LUT_SIZE {
    for ig in 0..LUT_SIZE {
      for ib in 0..LUT_SIZE {
        if count[ir][ig][ib] > 0 {
          occupied.push((ir, ig, ib));
        }
      }
    }
  }

  for idx in 0..LUT_TOTAL {
    let (r, g, b) = get_lut_coords(idx);

    // Skip if already occupied
    if count[r][g][b] > 0 {
      continue;
    }

    // Apply RBF interpolation
    let mut weighted_sum = [0.0f32; 3];
    let mut weight_total = 0.0f32;

    // Use limited number of nearest neighbors for efficiency
    const MAX_NEIGHBORS: usize = 20;
    
    // Calculate distances to all occupied cells
    let mut distances: Vec<(f32, usize, usize, usize)> = occupied
      .iter()
      .map(|&(ir, ig, ib)| {
        let dr = (r as i32 - ir as i32) as f32;
        let dg = (g as i32 - ig as i32) as f32;
        let db = (b as i32 - ib as i32) as f32;
        let dist = (dr * dr + dg * dg + db * db).sqrt();
        (dist, ir, ig, ib)
      })
      .collect();

    // Sort by distance and take closest MAX_NEIGHBORS
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let num_neighbors = MAX_NEIGHBORS.min(distances.len());

    for i in 0..num_neighbors {
      let (dist, ir, ig, ib) = distances[i];
      
      // Compute RBF weight (Gaussian)
      let weight = rbf_gaussian(dist, RBF_EPSILON);

      // Accumulate weighted values
      let neighbor_idx = get_lut_index(ir, ig, ib);
      weighted_sum[0] += lut[neighbor_idx][0] * weight;
      weighted_sum[1] += lut[neighbor_idx][1] * weight;
      weighted_sum[2] += lut[neighbor_idx][2] * weight;
      weight_total += weight;
    }

    // Normalize by total weight
    if weight_total > 0.0 {
      lut[idx] = [
        weighted_sum[0] / weight_total,
        weighted_sum[1] / weight_total,
        weighted_sum[2] / weight_total,
      ];
      filled_count += 1;
    } else {
      // Fallback: identity mapping
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
  writeln!(file, "TITLE \"Residual LUT (RBF)\"")?;
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
