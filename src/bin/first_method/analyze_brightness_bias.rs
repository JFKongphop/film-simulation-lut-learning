use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc};

fn main() -> Result<()> {
  println!("🔍 Analyzing Brightness Bias in LUT Output");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  // Paths
  let ground_truth_path = "source/compare/classic-chrome/9.JPG";
  let lut_output_path = "outputs/first_method/lut_33.jpg";

  // Load images
  println!("\n📷 Loading images...");
  let ground_truth = imgcodecs::imread(ground_truth_path, imgcodecs::IMREAD_COLOR)?;
  let lut_output = imgcodecs::imread(lut_output_path, imgcodecs::IMREAD_COLOR)?;

  println!(
    "   Ground truth: {}x{}",
    ground_truth.cols(),
    ground_truth.rows()
  );
  println!("   LUT output: {}x{}", lut_output.cols(), lut_output.rows());

  // Verify same size
  if ground_truth.size()? != lut_output.size()? {
    return Err(anyhow::anyhow!("Image sizes don't match!"));
  }

  let rows = ground_truth.rows();
  let cols = ground_truth.cols();
  let total_pixels = (rows * cols) as f64;

  // Convert to LAB color space for luminance analysis
  println!("\n🎨 Converting to LAB color space...");
  let mut gt_lab = Mat::default();
  let mut lut_lab = Mat::default();

  imgproc::cvt_color(
    &ground_truth,
    &mut gt_lab,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;
  imgproc::cvt_color(
    &lut_output,
    &mut lut_lab,
    imgproc::COLOR_BGR2Lab,
    0,
    core::AlgorithmHint::ALGO_HINT_DEFAULT,
  )?;

  // Analyze per-channel bias and absolute errors
  println!("\n📊 Computing RGB Bias (Mean Error)...");

  let mut sum_b_error = 0.0;
  let mut sum_g_error = 0.0;
  let mut sum_r_error = 0.0;

  let mut sum_b_abs_error = 0.0;
  let mut sum_g_abs_error = 0.0;
  let mut sum_r_abs_error = 0.0;

  let mut sum_l_error = 0.0;
  let mut sum_a_error = 0.0;
  let mut sum_b_lab_error = 0.0;

  let mut sum_l_abs_error = 0.0;

  // Histogram of L differences
  let mut l_diff_histogram = vec![0u32; 51]; // -25 to +25, each bin = 1

  for y in 0..rows {
    for x in 0..cols {
      // RGB analysis
      let gt_pixel = ground_truth.at_2d::<core::Vec3b>(y, x)?;
      let lut_pixel = lut_output.at_2d::<core::Vec3b>(y, x)?;

      // Signed error (LUT - Ground Truth)
      let b_error = lut_pixel[0] as f64 - gt_pixel[0] as f64;
      let g_error = lut_pixel[1] as f64 - gt_pixel[1] as f64;
      let r_error = lut_pixel[2] as f64 - gt_pixel[2] as f64;

      sum_b_error += b_error;
      sum_g_error += g_error;
      sum_r_error += r_error;

      sum_b_abs_error += b_error.abs();
      sum_g_abs_error += g_error.abs();
      sum_r_abs_error += r_error.abs();

      // LAB analysis
      let gt_lab_pixel = gt_lab.at_2d::<core::Vec3b>(y, x)?;
      let lut_lab_pixel = lut_lab.at_2d::<core::Vec3b>(y, x)?;

      // L channel: [0, 255] maps to [0, 100] in real LAB
      let gt_l = gt_lab_pixel[0] as f64;
      let lut_l = lut_lab_pixel[0] as f64;
      let l_error = lut_l - gt_l;

      sum_l_error += l_error;
      sum_l_abs_error += l_error.abs();

      // a and b channels: [0, 255] maps to [-128, 127]
      let a_error = lut_lab_pixel[1] as f64 - gt_lab_pixel[1] as f64;
      let b_lab_error = lut_lab_pixel[2] as f64 - gt_lab_pixel[2] as f64;

      sum_a_error += a_error;
      sum_b_lab_error += b_lab_error;

      // Histogram of L difference
      let l_diff_bin = ((l_error / 255.0 * 100.0).round() as i32 + 25).clamp(0, 50) as usize;
      l_diff_histogram[l_diff_bin] += 1;
    }
  }

  // Compute mean errors (bias)
  let mean_b_error = sum_b_error / total_pixels;
  let mean_g_error = sum_g_error / total_pixels;
  let mean_r_error = sum_r_error / total_pixels;

  let mean_b_abs_error = sum_b_abs_error / total_pixels;
  let mean_g_abs_error = sum_g_abs_error / total_pixels;
  let mean_r_abs_error = sum_r_abs_error / total_pixels;

  let mean_l_error = sum_l_error / total_pixels;
  let mean_l_abs_error = sum_l_abs_error / total_pixels;
  let mean_a_error = sum_a_error / total_pixels;
  let mean_b_lab_error = sum_b_lab_error / total_pixels;

  // Print results
  println!("\n📈 RGB Bias Analysis (8-bit scale [0-255]):");
  println!("   ┌─────────┬────────────┬─────────────┐");
  println!("   │ Channel │ Mean Error │ Mean Abs Er │");
  println!("   ├─────────┼────────────┼─────────────┤");
  println!(
    "   │ Blue    │ {:+7.3}    │ {:7.3}     │",
    mean_b_error, mean_b_abs_error
  );
  println!(
    "   │ Green   │ {:+7.3}    │ {:7.3}     │",
    mean_g_error, mean_g_abs_error
  );
  println!(
    "   │ Red     │ {:+7.3}    │ {:7.3}     │",
    mean_r_error, mean_r_abs_error
  );
  println!("   └─────────┴────────────┴─────────────┘");

  println!("\n💡 Interpretation (Mean Error):");
  println!("   Positive = LUT output brighter than ground truth");
  println!("   Negative = LUT output darker than ground truth");
  println!("   Zero = No systematic bias");

  // LAB analysis
  println!("\n🌈 LAB Color Space Bias:");
  println!(
    "   L* (Luminance) Mean Error: {:+.3} (8-bit scale)",
    mean_l_error
  );
  println!(
    "   L* (Luminance) MAE:        {:.3} (8-bit scale)",
    mean_l_abs_error
  );
  println!(
    "   L* in real units:          {:+.3} (0-100 scale)",
    mean_l_error / 255.0 * 100.0
  );
  println!();
  println!("   a* (Green-Red) Mean Error: {:+.3}", mean_a_error);
  println!("   b* (Blue-Yellow) Mean Error: {:+.3}", mean_b_lab_error);

  // Brightness verdict
  println!("\n🔦 Brightness Verdict:");
  let l_bias_percent = (mean_l_error / 255.0) * 100.0;

  if l_bias_percent.abs() < 0.5 {
    println!(
      "   ✅ No significant brightness bias ({:+.2}%)",
      l_bias_percent
    );
  } else if l_bias_percent > 0.5 {
    println!(
      "   ⚠️  LUT output is BRIGHTER by {:.2}% ({:+.2} in 0-100 L*)",
      l_bias_percent,
      mean_l_error / 255.0 * 100.0
    );
  } else {
    println!(
      "   ⚠️  LUT output is DARKER by {:.2}% ({:+.2} in 0-100 L*)",
      l_bias_percent,
      mean_l_error / 255.0 * 100.0
    );
  }

  // Overall RGB brightness
  let overall_rgb_bias = (mean_r_error + mean_g_error + mean_b_error) / 3.0;
  println!(
    "   Overall RGB bias: {:+.3} ({:+.2}%)",
    overall_rgb_bias,
    (overall_rgb_bias / 255.0) * 100.0
  );

  // Color shift analysis
  println!("\n🎨 Color Shift Analysis:");
  let a_shift_direction = if mean_a_error > 0.5 {
    "towards RED"
  } else if mean_a_error < -0.5 {
    "towards GREEN"
  } else {
    "neutral (no shift)"
  };

  let b_shift_direction = if mean_b_lab_error > 0.5 {
    "towards YELLOW"
  } else if mean_b_lab_error < -0.5 {
    "towards BLUE"
  } else {
    "neutral (no shift)"
  };

  println!(
    "   a* channel: {} ({:+.2})",
    a_shift_direction, mean_a_error
  );
  println!(
    "   b* channel: {} ({:+.2})",
    b_shift_direction, mean_b_lab_error
  );

  // Histogram visualization
  println!("\n📊 Luminance Difference Distribution:");
  println!("   (L_LUT - L_GT in 0-100 scale)");
  println!();

  // Find max count for scaling
  let max_count = *l_diff_histogram.iter().max().unwrap_or(&1);
  let scale = 50.0 / max_count as f64;

  for (i, &count) in l_diff_histogram.iter().enumerate() {
    let l_diff = i as i32 - 25;
    let bar_length = (count as f64 * scale) as usize;
    let bar = "█".repeat(bar_length);

    if count > 0 {
      println!("   {:+3}: {} {}", l_diff, bar, count);
    }
  }

  println!("\n🎉 Analysis complete!");

  Ok(())
}
