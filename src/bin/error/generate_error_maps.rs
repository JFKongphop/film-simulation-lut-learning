use anyhow::Result;
use opencv::{core, imgcodecs, imgproc, prelude::*};
use std::fs;

fn main() -> Result<()> {
  println!("Loading images...");

  // Load images
  let ground_truth = imgcodecs::imread(
    "source/compare/classic-chrome/9.JPG",
    imgcodecs::IMREAD_COLOR,
  )?;
  let method1_output =
    imgcodecs::imread("outputs/first_method/lut_33.jpg", imgcodecs::IMREAD_COLOR)?;
  let method2_output = imgcodecs::imread(
    "outputs/second_method/final_clone.jpg",
    imgcodecs::IMREAD_COLOR,
  )?;

  println!("Ground truth size: {:?}", ground_truth.size()?);
  println!("Method 1 size: {:?}", method1_output.size()?);
  println!("Method 2 size: {:?}", method2_output.size()?);

  // Convert BGR to LAB
  println!("Converting to LAB color space...");
  let mut gt_lab = core::Mat::default();
  let mut m1_lab = core::Mat::default();
  let mut m2_lab = core::Mat::default();

  imgproc::cvt_color(
    &ground_truth,
    &mut gt_lab,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;
  imgproc::cvt_color(
    &method1_output,
    &mut m1_lab,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;
  imgproc::cvt_color(
    &method2_output,
    &mut m2_lab,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;

  // Calculate Delta E
  println!("Calculating Delta E...");
  let delta_e_m1 = calculate_delta_e(&gt_lab, &m1_lab)?;
  let delta_e_m2 = calculate_delta_e(&gt_lab, &m2_lab)?;

  // Print statistics
  let stats_m1 = calculate_statistics(&delta_e_m1)?;
  let stats_m2 = calculate_statistics(&delta_e_m2)?;

  println!("\n{}", "=".repeat(60));
  println!("ERROR STATISTICS");
  println!("{}", "=".repeat(60));
  println!("Method 1 (Direct LUT):");
  println!("  Mean ΔE: {:.4}", stats_m1.mean);
  println!("  Median ΔE: {:.4}", stats_m1.median);
  println!("  Max ΔE: {:.4}", stats_m1.max);
  println!("  Std Dev: {:.4}", stats_m1.std_dev);
  println!();
  println!("Method 2 (Pipeline):");
  println!("  Mean ΔE: {:.4}", stats_m2.mean);
  println!("  Median ΔE: {:.4}", stats_m2.median);
  println!("  Max ΔE: {:.4}", stats_m2.max);
  println!("  Std Dev: {:.4}", stats_m2.std_dev);
  println!("{}", "=".repeat(60));
  println!();

  // Create output directory
  fs::create_dir_all("outputs/error")?;

  // Generate jet colormap visualization
  println!("Creating error maps with jet colormap...");
  let error_map_m1 = create_jet_colormap(&delta_e_m1, 0.0, 5.0)?;
  let error_map_m2 = create_jet_colormap(&delta_e_m2, 0.0, 5.0)?;

  // PNG compression: 0 (no compression) to 9 (max compression)
  let mut png_params = core::Vector::new();
  png_params.push(imgcodecs::IMWRITE_PNG_COMPRESSION);
  png_params.push(9); // Maximum compression

  imgcodecs::imwrite(
    "outputs/error/error_map_method1_jet.png",
    &error_map_m1,
    &png_params,
  )?;
  imgcodecs::imwrite(
    "outputs/error/error_map_method2_jet.png",
    &error_map_m2,
    &png_params,
  )?;
  println!("✓ Saved: outputs/error/error_map_method1_jet.png");
  println!("✓ Saved: outputs/error/error_map_method2_jet.png");

  // Generate custom color-coded visualization
  println!("Creating custom color-coded error maps...");
  let custom_m1 = create_custom_colormap(&delta_e_m1)?;
  let custom_m2 = create_custom_colormap(&delta_e_m2)?;

  let dist_m1 = calculate_distribution(&delta_e_m1)?;
  let dist_m2 = calculate_distribution(&delta_e_m2)?;

  println!();
  println!("Method 1 - Pixel distribution:");
  println!("  ΔE < 1: {:.2}%", dist_m1.0);
  println!("  1 ≤ ΔE < 2: {:.2}%", dist_m1.1);
  println!("  2 ≤ ΔE < 5: {:.2}%", dist_m1.2);
  println!("  ΔE ≥ 5: {:.2}%", dist_m1.3);
  println!();
  println!("Method 2 - Pixel distribution:");
  println!("  ΔE < 1: {:.2}%", dist_m2.0);
  println!("  1 ≤ ΔE < 2: {:.2}%", dist_m2.1);
  println!("  2 ≤ ΔE < 5: {:.2}%", dist_m2.2);
  println!("  ΔE ≥ 5: {:.2}%", dist_m2.3);

  imgcodecs::imwrite(
    "outputs/error/error_map_method1_custom.png",
    &custom_m1,
    &png_params,
  )?;
  imgcodecs::imwrite(
    "outputs/error/error_map_method2_custom.png",
    &custom_m2,
    &png_params,
  )?;
  println!("✓ Saved: outputs/error/error_map_method1_custom.png");
  println!("✓ Saved: outputs/error/error_map_method2_custom.png");

  // Generate amplified differences
  println!("Creating amplified difference maps...");
  let amp_m1 = create_amplified_diff(&ground_truth, &method1_output, 10.0)?;
  let amp_m2 = create_amplified_diff(&ground_truth, &method2_output, 10.0)?;

  imgcodecs::imwrite(
    "outputs/error/amplified_diff_method1.png",
    &amp_m1,
    &png_params,
  )?;
  imgcodecs::imwrite(
    "outputs/error/amplified_diff_method2.png",
    &amp_m2,
    &png_params,
  )?;
  println!("✓ Saved: outputs/error/amplified_diff_method1.png");
  println!("✓ Saved: outputs/error/amplified_diff_method2.png");

  println!();
  println!("{}", "=".repeat(60));
  println!("COMPLETED!");
  println!("{}", "=".repeat(60));
  println!("Generated 6 visualization files in outputs/error/");
  println!("{}", "=".repeat(60));

  Ok(())
}

struct Statistics {
  mean: f64,
  median: f64,
  max: f64,
  std_dev: f64,
}

fn calculate_delta_e(lab1: &core::Mat, lab2: &core::Mat) -> Result<core::Mat> {
  let size = lab1.size()?;
  let mut delta_e = core::Mat::new_rows_cols_with_default(
    size.height,
    size.width,
    core::CV_32F,
    core::Scalar::all(0.0),
  )?;

  for y in 0..size.height {
    for x in 0..size.width {
      let pixel1: core::Vec3b = *lab1.at_2d(y, x)?;
      let pixel2: core::Vec3b = *lab2.at_2d(y, x)?;

      let dl = pixel1[0] as f32 - pixel2[0] as f32;
      let da = pixel1[1] as f32 - pixel2[1] as f32;
      let db = pixel1[2] as f32 - pixel2[2] as f32;

      let de = (dl * dl + da * da + db * db).sqrt();
      *delta_e.at_2d_mut(y, x)? = de;
    }
  }

  Ok(delta_e)
}

fn calculate_statistics(delta_e: &core::Mat) -> Result<Statistics> {
  let size = delta_e.size()?;
  let total_pixels = (size.height * size.width) as f64;

  let mut sum = 0.0;
  let mut max_val = 0.0;
  let mut values = Vec::new();

  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)? as f64;
      sum += val;
      if val > max_val {
        max_val = val;
      }
      values.push(val);
    }
  }

  let mean = sum / total_pixels;

  // Calculate median
  values.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median = values[values.len() / 2];

  // Calculate standard deviation
  let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / total_pixels;
  let std_dev = variance.sqrt();

  Ok(Statistics {
    mean,
    median,
    max: max_val,
    std_dev,
  })
}

fn create_jet_colormap(delta_e: &core::Mat, min_val: f32, max_val: f32) -> Result<core::Mat> {
  let size = delta_e.size()?;
  let mut colored = core::Mat::new_rows_cols_with_default(
    size.height,
    size.width,
    core::CV_8UC3,
    core::Scalar::all(0.0),
  )?;

  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)?;
      let normalized = ((val - min_val) / (max_val - min_val)).clamp(0.0, 1.0);
      let color = jet_color(normalized);
      *colored.at_2d_mut(y, x)? = core::Vec3b::from([color.2, color.1, color.0]);
    }
  }

  Ok(colored)
}

fn jet_color(value: f32) -> (u8, u8, u8) {
  // Jet colormap: blue -> cyan -> green -> yellow -> red
  let r = ((1.5 - 4.0 * (value - 0.75).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  let g = ((1.5 - 4.0 * (value - 0.5).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  let b = ((1.5 - 4.0 * (value - 0.25).abs()).clamp(0.0, 1.0) * 255.0) as u8;
  (r, g, b)
}

fn create_custom_colormap(delta_e: &core::Mat) -> Result<core::Mat> {
  let size = delta_e.size()?;
  let mut colored = core::Mat::new_rows_cols_with_default(
    size.height,
    size.width,
    core::CV_8UC3,
    core::Scalar::all(0.0),
  )?;

  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)?;

      let color = if val < 1.0 {
        [255, 100, 0] // Blue (BGR)
      } else if val < 2.0 {
        [100, 255, 0] // Green
      } else if val < 5.0 {
        [0, 255, 255] // Yellow
      } else {
        [50, 50, 255] // Red
      };

      *colored.at_2d_mut(y, x)? = core::Vec3b::from(color);
    }
  }

  Ok(colored)
}

fn calculate_distribution(delta_e: &core::Mat) -> Result<(f64, f64, f64, f64)> {
  let size = delta_e.size()?;
  let total = (size.height * size.width) as f64;

  let mut count1 = 0;
  let mut count2 = 0;
  let mut count3 = 0;
  let mut count4 = 0;

  for y in 0..size.height {
    for x in 0..size.width {
      let val = *delta_e.at_2d::<f32>(y, x)?;
      if val < 1.0 {
        count1 += 1;
      } else if val < 2.0 {
        count2 += 1;
      } else if val < 5.0 {
        count3 += 1;
      } else {
        count4 += 1;
      }
    }
  }

  Ok((
    count1 as f64 / total * 100.0,
    count2 as f64 / total * 100.0,
    count3 as f64 / total * 100.0,
    count4 as f64 / total * 100.0,
  ))
}

fn create_amplified_diff(img1: &core::Mat, img2: &core::Mat, amplify: f32) -> Result<core::Mat> {
  let mut diff = core::Mat::default();
  core::absdiff(img1, img2, &mut diff)?;

  let mut amplified = core::Mat::default();
  diff.convert_to(&mut amplified, core::CV_8UC3, amplify as f64, 0.0)?;

  Ok(amplified)
}
