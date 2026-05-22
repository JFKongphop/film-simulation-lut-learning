use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc};

/// Compute Mean Squared Error between two images
fn compute_mse(img1: &Mat, img2: &Mat) -> Result<f64> {
  let rows = img1.rows();
  let cols = img1.cols();

  let mut sum_squared_diff = 0.0;
  let mut pixel_count = 0;

  for y in 0..rows {
    for x in 0..cols {
      let pixel1 = img1.at_2d::<core::Vec3b>(y, x)?;
      let pixel2 = img2.at_2d::<core::Vec3b>(y, x)?;

      for c in 0..3 {
        let diff = pixel1[c] as f64 - pixel2[c] as f64;
        sum_squared_diff += diff * diff;
        pixel_count += 1;
      }
    }
  }

  Ok(sum_squared_diff / pixel_count as f64)
}

/// Compute Peak Signal-to-Noise Ratio
fn compute_psnr(mse: f64) -> f64 {
  if mse == 0.0 {
    f64::INFINITY
  } else {
    let max_pixel = 255.0;
    20.0 * (max_pixel / mse.sqrt()).log10()
  }
}

/// Compute Delta E (CIE76) between two LAB colors
fn delta_e_cie76(l1: f32, a1: f32, b1: f32, l2: f32, a2: f32, b2: f32) -> f32 {
  let dl = l1 - l2;
  let da = a1 - a2;
  let db = b1 - b2;
  (dl * dl + da * da + db * db).sqrt()
}

/// Compute average Delta E between two images
fn compute_delta_e(img1: &Mat, img2: &Mat) -> Result<(f32, f32, f32)> {
  let rows = img1.rows();
  let cols = img1.cols();

  // Convert both images to LAB color space
  let mut lab1 = Mat::default();
  let mut lab2 = Mat::default();

  imgproc::cvt_color(
    img1,
    &mut lab1,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;
  imgproc::cvt_color(
    img2,
    &mut lab2,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;

  let mut sum_delta_e = 0.0f32;
  let mut max_delta_e = 0.0f32;
  let mut pixel_count = 0;

  for y in 0..rows {
    for x in 0..cols {
      let pixel1 = lab1.at_2d::<core::Vec3b>(y, x)?;
      let pixel2 = lab2.at_2d::<core::Vec3b>(y, x)?;

      // OpenCV LAB values are scaled: L: [0, 255] -> [0, 100], a/b: [0, 255] -> [-128, 127]
      let l1 = pixel1[0] as f32 * 100.0 / 255.0;
      let a1 = pixel1[1] as f32 - 128.0;
      let b1 = pixel1[2] as f32 - 128.0;

      let l2 = pixel2[0] as f32 * 100.0 / 255.0;
      let a2 = pixel2[1] as f32 - 128.0;
      let b2 = pixel2[2] as f32 - 128.0;

      let de = delta_e_cie76(l1, a1, b1, l2, a2, b2);
      sum_delta_e += de;
      max_delta_e = max_delta_e.max(de);
      pixel_count += 1;
    }
  }

  let avg_delta_e = sum_delta_e / pixel_count as f32;

  // Also compute median by collecting all values
  let mut delta_e_values = Vec::new();
  for y in 0..rows {
    for x in 0..cols {
      let pixel1 = lab1.at_2d::<core::Vec3b>(y, x)?;
      let pixel2 = lab2.at_2d::<core::Vec3b>(y, x)?;

      let l1 = pixel1[0] as f32 * 100.0 / 255.0;
      let a1 = pixel1[1] as f32 - 128.0;
      let b1 = pixel1[2] as f32 - 128.0;

      let l2 = pixel2[0] as f32 * 100.0 / 255.0;
      let a2 = pixel2[1] as f32 - 128.0;
      let b2 = pixel2[2] as f32 - 128.0;

      let de = delta_e_cie76(l1, a1, b1, l2, a2, b2);
      delta_e_values.push(de);
    }
  }

  delta_e_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median_delta_e = delta_e_values[delta_e_values.len() / 2];

  Ok((avg_delta_e, max_delta_e, median_delta_e))
}

/// Compute per-channel statistics
fn compute_channel_stats(img1: &Mat, img2: &Mat) -> Result<()> {
  let rows = img1.rows();
  let cols = img1.cols();

  let mut b_diff_sum = 0.0;
  let mut g_diff_sum = 0.0;
  let mut r_diff_sum = 0.0;
  let pixel_count = (rows * cols) as f64;

  for y in 0..rows {
    for x in 0..cols {
      let pixel1 = img1.at_2d::<core::Vec3b>(y, x)?;
      let pixel2 = img2.at_2d::<core::Vec3b>(y, x)?;

      b_diff_sum += (pixel1[0] as f64 - pixel2[0] as f64).abs();
      g_diff_sum += (pixel1[1] as f64 - pixel2[1] as f64).abs();
      r_diff_sum += (pixel1[2] as f64 - pixel2[2] as f64).abs();
    }
  }

  println!("\n📊 Per-Channel Mean Absolute Error:");
  println!("   Blue:  {:.4}", b_diff_sum / pixel_count);
  println!("   Green: {:.4}", g_diff_sum / pixel_count);
  println!("   Red:   {:.4}", r_diff_sum / pixel_count);

  Ok(())
}

fn main() -> Result<()> {
  println!("📊 Comparing First Method (33³ LUT) with Ground Truth (4 Test Images)");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  // Test images
  let test_images = vec![91, 92, 93, 94];

  // Store results for each image
  use std::collections::HashMap;
  let mut all_results: HashMap<u32, (f64, f64, f32, f32, f32)> = HashMap::new();

  println!("\n🔄 Processing images...\n");

  // Process each test image
  for img_num in &test_images {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     📸 Test Image: {}.JPG                                  ║", img_num);
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    // Load images
    let ground_truth_path = format!("source/compare/classic-chrome/{}.JPG", img_num);
    let lut_output_path = format!("outputs/first_method/lut_33_{}.jpg", img_num);

    let ground_truth = imgcodecs::imread(&ground_truth_path, imgcodecs::IMREAD_COLOR)?;
    let lut_output = imgcodecs::imread(&lut_output_path, imgcodecs::IMREAD_COLOR)?;

    println!("   Ground truth: {}x{}", ground_truth.cols(), ground_truth.rows());
    println!("   LUT output: {}x{}", lut_output.cols(), lut_output.rows());

    // Compute metrics
    let mse = compute_mse(&ground_truth, &lut_output)?;
    let psnr = compute_psnr(mse);
    let (avg_de, max_de, median_de) = compute_delta_e(&ground_truth, &lut_output)?;

    println!("\n┌────────────┬──────────┬───────────┬──────────┬──────────┬──────────┐");
    println!("│ Method     │   MSE    │ PSNR (dB) │ Avg ΔE   │ Med ΔE   │ Max ΔE   │");
    println!("├────────────┼──────────┼───────────┼──────────┼──────────┼──────────┤");
    println!(
      "│ {:10} │ {:8.4} │ {:9.4} │ {:8.4} │ {:8.4} │ {:8.2} │",
      "LUT 33³", mse, psnr, avg_de, median_de, max_de
    );
    println!("└────────────┴──────────┴───────────┴──────────┴──────────┴──────────┘");

    // Store results
    all_results.insert(*img_num, (mse, psnr, avg_de, median_de, max_de));
  }

  // Compute averages
  let num_images = test_images.len() as f64;
  let mut sum_mse = 0.0;
  let mut sum_psnr = 0.0;
  let mut sum_avg_de = 0.0;
  let mut sum_median_de = 0.0;
  let mut sum_max_de = 0.0;

  for img_num in &test_images {
    let (mse, psnr, avg_de, median_de, max_de) = all_results.get(img_num).unwrap();
    sum_mse += mse;
    sum_psnr += psnr;
    sum_avg_de += avg_de;
    sum_median_de += median_de;
    sum_max_de += max_de;
  }

  let avg_mse = sum_mse / num_images;
  let avg_psnr = sum_psnr / num_images;
  let avg_avg_de = sum_avg_de / num_images as f32;
  let avg_median_de = sum_median_de / num_images as f32;
  let avg_max_de = sum_max_de / num_images as f32;

  // Print average summary table
  println!("\n\n");
  println!("╔═══════════════════════════════════════════════════════════════════════════╗");
  println!("║           📊 AVERAGE RESULTS - First Method (4 Test Images)              ║");
  println!("╚═══════════════════════════════════════════════════════════════════════════╝");
  println!();
  println!("┌────────────┬──────────┬───────────┬──────────┬──────────┬──────────┐");
  println!("│ Method     │   MSE    │ PSNR (dB) │ Avg ΔE   │ Med ΔE   │ Max ΔE   │");
  println!("├────────────┼──────────┼───────────┼──────────┼──────────┼──────────┤");
  println!(
    "│ {:10} │ {:8.4} │ {:9.4} │ {:8.4} │ {:8.4} │ {:8.2} │",
    "LUT 33³", avg_mse, avg_psnr, avg_avg_de, avg_median_de, avg_max_de
  );
  println!("└────────────┴──────────┴───────────┴──────────┴──────────┴──────────┘");

  println!("\n🎉 Comparison complete!");
  println!("   First Method (33³ LUT): {:.4} dB PSNR, ΔE {:.4}", avg_psnr, avg_avg_de);

  Ok(())
}
