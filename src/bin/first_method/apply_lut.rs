use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgcodecs};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Represents a 3D LUT loaded from a .cube file
struct Lut3D {
  size: usize,
  data: Vec<Vec<Vec<[f32; 3]>>>, // [R][G][B] -> [R', G', B']
}

impl Lut3D {
  /// Load a 3D LUT from a .cube file
  fn from_cube_file<P: AsRef<Path>>(path: P) -> Result<Self> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut size = 0;
    let mut rgb_values = Vec::new();

    // Parse .cube file
    for line in reader.lines() {
      let line = line?;
      let line = line.trim();

      // Skip comments and empty lines
      if line.is_empty() || line.starts_with('#') {
        continue;
      }

      // Parse LUT_3D_SIZE
      if line.starts_with("LUT_3D_SIZE") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
          size = parts[1].parse()?;
        }
        continue;
      }

      // Skip TITLE and other metadata
      if line.starts_with("TITLE")
        || line.starts_with("DOMAIN_MIN")
        || line.starts_with("DOMAIN_MAX")
      {
        continue;
      }

      // Parse RGB values
      let parts: Vec<&str> = line.split_whitespace().collect();
      if parts.len() >= 3 {
        let r: f32 = parts[0].parse()?;
        let g: f32 = parts[1].parse()?;
        let b: f32 = parts[2].parse()?;
        rgb_values.push([r, g, b]);
      }
    }

    if size == 0 {
      anyhow::bail!("LUT_3D_SIZE not found in .cube file");
    }

    let expected_count = size * size * size;
    if rgb_values.len() != expected_count {
      anyhow::bail!(
        "Expected {} RGB values, found {}",
        expected_count,
        rgb_values.len()
      );
    }

    // Build 3D array from flat list
    // Standard .cube format: Blue changes fastest, then Green, then Red
    let mut data = vec![vec![vec![[0.0f32; 3]; size]; size]; size];
    let mut idx = 0;
    for r in 0..size {
      for g in 0..size {
        for b in 0..size {
          data[r][g][b] = rgb_values[idx];
          idx += 1;
        }
      }
    }

    Ok(Lut3D { size, data })
  }

  /// Apply LUT to a single RGB value using trilinear interpolation
  fn apply(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
    let n = self.size as f32;
    let n_max = (self.size - 1) as usize;

    // Step 1: Compute LUT position
    let x = r * (n - 1.0);
    let y = g * (n - 1.0);
    let z = b * (n - 1.0);

    // Step 2: Get surrounding indices
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;

    let x1 = (x0 + 1).min(n_max);
    let y1 = (y0 + 1).min(n_max);
    let z1 = (z0 + 1).min(n_max);

    // Step 3: Compute interpolation weights
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let dz = z - z0 as f32;

    // Step 4: Fetch 8 corner values
    let c000 = self.data[x0][y0][z0];
    let c001 = self.data[x0][y0][z1];
    let c010 = self.data[x0][y1][z0];
    let c011 = self.data[x0][y1][z1];
    let c100 = self.data[x1][y0][z0];
    let c101 = self.data[x1][y0][z1];
    let c110 = self.data[x1][y1][z0];
    let c111 = self.data[x1][y1][z1];

    // Step 5: Trilinear interpolation
    let mut result = [0.0f32; 3];
    for channel in 0..3 {
      let c00 = c000[channel] * (1.0 - dz) + c001[channel] * dz;
      let c01 = c010[channel] * (1.0 - dz) + c011[channel] * dz;
      let c10 = c100[channel] * (1.0 - dz) + c101[channel] * dz;
      let c11 = c110[channel] * (1.0 - dz) + c111[channel] * dz;

      let c0 = c00 * (1.0 - dy) + c01 * dy;
      let c1 = c10 * (1.0 - dy) + c11 * dy;

      result[channel] = c0 * (1.0 - dx) + c1 * dx;
    }

    result
  }

  /// Apply LUT to an entire image
  fn apply_to_image(&self, input: &Mat) -> Result<Mat> {
    let rows = input.rows();
    let cols = input.cols();

    let mut output = input.clone();

    // Process each pixel
    for y in 0..rows {
      for x in 0..cols {
        // Get BGR pixel
        let pixel = input.at_2d::<core::Vec3b>(y, x)?;

        // Convert to [0, 1] range (note: OpenCV uses BGR order)
        let b = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let r = pixel[2] as f32 / 255.0;

        // Apply LUT (with RGB order)
        let transformed = self.apply(r, g, b);

        // Clamp and convert back to [0, 255]
        let r_out = (transformed[0].clamp(0.0, 1.0) * 255.0) as u8;
        let g_out = (transformed[1].clamp(0.0, 1.0) * 255.0) as u8;
        let b_out = (transformed[2].clamp(0.0, 1.0) * 255.0) as u8;

        // Set output pixel (BGR order)
        let output_pixel = output.at_2d_mut::<core::Vec3b>(y, x)?;
        output_pixel[0] = b_out;
        output_pixel[1] = g_out;
        output_pixel[2] = r_out;
      }
    }

    Ok(output)
  }
}

fn main() -> Result<()> {
  println!("🎨 Applying 3D LUT to Image");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  // Paths
  let lut_path = "outputs/first_method/lut_33.cube";
  let input_path = "source/compare/standard/9.JPG";
  let output_path = "outputs/first_method/lut_33.jpg";

  // Step 1: Load LUT
  println!("📖 Loading LUT from: {}", lut_path);
  let lut = Lut3D::from_cube_file(lut_path)?;
  println!("✅ Loaded {}x{}x{} LUT", lut.size, lut.size, lut.size);

  // Show sample LUT values
  println!("\n🔍 Sample LUT values:");
  let black_out = lut.apply(0.0, 0.0, 0.0);
  println!(
    "   Black [0,0,0] -> [{:.4}, {:.4}, {:.4}]",
    black_out[0], black_out[1], black_out[2]
  );

  let white_out = lut.apply(1.0, 1.0, 1.0);
  println!(
    "   White [1,1,1] -> [{:.4}, {:.4}, {:.4}]",
    white_out[0], white_out[1], white_out[2]
  );

  let mid_out = lut.apply(0.5, 0.5, 0.5);
  println!(
    "   Mid [0.5,0.5,0.5] -> [{:.4}, {:.4}, {:.4}]",
    mid_out[0], mid_out[1], mid_out[2]
  );

  // Step 2: Load input image
  println!("\n📷 Loading input image: {}", input_path);
  let input = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;
  println!("✅ Loaded image: {}x{}", input.cols(), input.rows());

  // Step 3: Apply LUT
  println!("\n⚙️  Applying LUT with trilinear interpolation...");
  let output = lut.apply_to_image(&input)?;
  println!("✅ LUT applied successfully");

  // Step 4: Save output
  println!("\n💾 Saving output to: {}", output_path);
  imgcodecs::imwrite(output_path, &output, &core::Vector::new())?;
  println!("✅ Output saved successfully");

  println!("\n🎉 Done!");

  Ok(())
}
