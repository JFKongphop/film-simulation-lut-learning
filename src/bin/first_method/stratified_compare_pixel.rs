use anyhow::Result;
use film_simulation_lut_learning::utils::BasedImage;
use opencv::{core, imgcodecs, imgproc, prelude::*};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;

#[derive(Serialize)]
struct PixelData {
  index: usize,
  sr: f32,
  sg: f32,
  sb: f32,
  cr: f32,
  cg: f32,
  cb: f32,
  dr: f32,
  dg: f32,
  db: f32,
}

/// Convert RGB (0-255) to LAB color space
fn rgb_to_lab(r: u8, g: u8, b: u8) -> Result<(f32, f32, f32)> {
  // Create a 1x1 BGR image (OpenCV uses BGR)
  let mut bgr_mat = unsafe { Mat::new_rows_cols(1, 1, core::CV_8UC3)? };

  let pixel = bgr_mat.at_2d_mut::<core::Vec3b>(0, 0)?;
  pixel[0] = b;
  pixel[1] = g;
  pixel[2] = r;

  // Convert to LAB
  let mut lab_mat = Mat::default();
  imgproc::cvt_color(
    &bgr_mat,
    &mut lab_mat,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;

  let lab_pixel = lab_mat.at_2d::<core::Vec3b>(0, 0)?;

  // OpenCV LAB values are scaled: L: [0, 255] -> [0, 100], a/b: [0, 255] -> [-128, 127]
  let l = lab_pixel[0] as f32 * 100.0 / 255.0;
  let a = lab_pixel[1] as f32 - 128.0;
  let b = lab_pixel[2] as f32 - 128.0;

  Ok((l, a, b))
}

/// Compute LAB bucket (8x8x8 grid)
fn compute_bucket(l: f32, a: f32, b: f32) -> (usize, usize, usize) {
  // L bucket: 0-100 split into 8 equal ranges
  let l_bin = ((l / 12.5).floor() as usize).min(7);

  // a bucket: -128 to 127 split into 8 equal ranges (32 units each)
  let a_bin = (((a + 128.0) / 32.0).floor() as usize).min(7);

  // b bucket: -128 to 127 split into 8 equal ranges (32 units each)
  let b_bin = (((b + 128.0) / 32.0).floor() as usize).min(7);

  (l_bin, a_bin, b_bin)
}

fn main() -> Result<()> {
  // Use fixed seed for reproducibility
  let mut rng = StdRng::seed_from_u64(42);

  let mut all_pixel_data: Vec<PixelData> = Vec::new();

  // Process all image pairs (1.JPG through 8.JPG)
  for img_num in 1..=8 {
    let filename = format!("{}.JPG", img_num);
    let standard_path = format!("source/compare/standard/{}", filename);
    let chrome_path = format!("source/compare/classic-chrome/{}", filename);

    println!("📸 Processing image pair: {}", filename);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Load two images
    let img1_mat = imgcodecs::imread(&standard_path, imgcodecs::IMREAD_COLOR)?;
    let img2_mat = imgcodecs::imread(&chrome_path, imgcodecs::IMREAD_COLOR)?;

    // Convert to BasedImage for pixel access
    let img1 = BasedImage::from_mat(&img1_mat);
    let img2 = BasedImage::from_mat(&img2_mat);

    // Check if images have the same dimensions
    if img1.w != img2.w || img1.h != img2.h {
      println!(
        "❌ Images have different dimensions: {}x{} vs {}x{}",
        img1.w, img1.h, img2.w, img2.h
      );
      continue;
    }

    println!("   Image size: {}x{}", img1.w, img1.h);

    // Create buckets: (l_bin, a_bin, b_bin) -> Vec<PixelData>
    let mut buckets: HashMap<(usize, usize, usize), Vec<PixelData>> = HashMap::new();

    println!("   Assigning pixels to LAB buckets (8×8×8)...");

    let total_pixels = img1.w * img1.h;

    // Process all pixels and assign to buckets
    for pixel_idx in 0..total_pixels {
      let idx = pixel_idx * 3; // Convert pixel index to byte index (3 bytes per pixel)

      // Get BGR values (OpenCV uses BGR format)
      let b1 = img1.data[idx];
      let g1 = img1.data[idx + 1];
      let r1 = img1.data[idx + 2];

      let b2 = img2.data[idx];
      let g2 = img2.data[idx + 1];
      let r2 = img2.data[idx + 2];

      // Convert source RGB to LAB for bucketing
      let (l, a, b) = rgb_to_lab(r1, g1, b1)?;
      let bucket = compute_bucket(l, a, b);

      // Normalize RGB values to [0, 1]
      let sr = r1 as f32 / 255.0;
      let sg = g1 as f32 / 255.0;
      let sb = b1 as f32 / 255.0;
      let cr = r2 as f32 / 255.0;
      let cg = g2 as f32 / 255.0;
      let cb = b2 as f32 / 255.0;
      let dr = sr - cr;
      let dg = sg - cg;
      let db = sb - cb;

      // Store pixel data in bucket
      let pixel_data = PixelData {
        index: pixel_idx,
        sr,
        sg,
        sb,
        cr,
        cg,
        cb,
        dr,
        dg,
        db,
      };

      buckets
        .entry(bucket)
        .or_insert_with(Vec::new)
        .push(pixel_data);
    }

    println!("   Total buckets used: {}", buckets.len());

    // Sample from each bucket
    let mut selected_pixels = 0;
    let mut bucket_stats: Vec<(usize, usize)> = Vec::new(); // (original_size, sampled_size)

    for (_bucket_key, mut pixels) in buckets {
      let original_count = pixels.len();

      let sampled = if pixels.len() <= 200 {
        // Keep all pixels if 200 or fewer
        pixels
      } else {
        // Randomly sample exactly 200 pixels
        pixels.shuffle(&mut rng);
        pixels.into_iter().take(200).collect()
      };

      let sampled_count = sampled.len();
      selected_pixels += sampled_count;
      bucket_stats.push((original_count, sampled_count));

      all_pixel_data.extend(sampled);
    }

    // Show statistics
    println!(
      "   Selected pixels: {} (from {} total)",
      selected_pixels, total_pixels
    );
    println!(
      "   Coverage: {:.2}%",
      (selected_pixels as f64 / total_pixels as f64) * 100.0
    );

    // Show bucket distribution stats
    let buckets_with_sampling = bucket_stats
      .iter()
      .filter(|(orig, samp)| orig > samp)
      .count();
    println!(
      "   Buckets sampled (>200 pixels): {}",
      buckets_with_sampling
    );
    println!(
      "   Buckets fully included (≤200 pixels): {}",
      bucket_stats.len() - buckets_with_sampling
    );

    println!();
  } // End of image pair loop

  // Write all pixel data to CSV file
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  println!("📈 Combined Results:");
  println!("   Total samples collected: {}", all_pixel_data.len());

  println!("\n📝 Writing pixel data to CSV...");
  let file = File::create("outputs/pixel_comparison.csv")?;
  let mut wtr = csv::Writer::from_writer(file);

  // Write all pixel data
  for pixel_data in &all_pixel_data {
    wtr.serialize(pixel_data)?;
  }

  wtr.flush()?;
  println!("✅ CSV file saved: outputs/pixel_comparison.csv");
  println!("   Total records: {}", all_pixel_data.len());

  println!("\n💡 Stratified LAB sampling complete!");
  println!("   Better coverage of rare colors");
  println!("   Reduced redundancy from flat regions");

  Ok(())
}
