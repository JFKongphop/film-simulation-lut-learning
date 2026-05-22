use anyhow::Result;
use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc};

fn main() -> Result<()> {
  println!("📊 Comparing Per-Channel Method with Ground Truth\n");

  let test_images = vec![91, 92, 93, 94];
  let mut sum_mse = 0.0;
  let mut sum_psnr = 0.0;
  let mut sum_delta_e = 0.0;

  for img_num in &test_images {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📸 Image {}.JPG", img_num);

    let gt_path = format!("source/compare/classic-chrome/{}.JPG", img_num);
    let out_path = format!("outputs/second_method/final_clone_{}_perchannel.jpg", img_num);

    let ground_truth = imgcodecs::imread(&gt_path, imgcodecs::IMREAD_COLOR)?;
    let output = imgcodecs::imread(&out_path, imgcodecs::IMREAD_COLOR)?;

    let mse = compute_mse(&ground_truth, &output)?;
    let psnr = compute_psnr(mse);
    let (avg_de, _, _) = compute_delta_e(&ground_truth, &output)?;

    println!("MSE:  {:.4}", mse);
    println!("PSNR: {:.4} dB", psnr);
    println!("ΔE:   {:.4}", avg_de);

    sum_mse += mse;
    sum_psnr += psnr;
    sum_delta_e += avg_de;
  }

  let n = test_images.len() as f64;
  println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  println!("📊 AVERAGE RESULTS (Per-Channel Method)");
  println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  println!("Average MSE:  {:.4}", sum_mse / n);
  println!("Average PSNR: {:.4} dB", sum_psnr / n);
  println!("Average ΔE:   {:.4}", sum_delta_e / n as f32);

  Ok(())
}

fn compute_mse(img1: &Mat, img2: &Mat) -> Result<f64> {
  let rows = img1.rows();
  let cols = img1.cols();
  let mut sum = 0.0f64;
  let total_pixels = (rows * cols * 3) as f64;

  for y in 0..rows {
    for x in 0..cols {
      let p1 = img1.at_2d::<core::Vec3b>(y, x)?;
      let p2 = img2.at_2d::<core::Vec3b>(y, x)?;

      for c in 0..3 {
        let diff = p1[c] as f64 - p2[c] as f64;
        sum += diff * diff;
      }
    }
  }

  Ok(sum / total_pixels)
}

fn compute_psnr(mse: f64) -> f64 {
  if mse < 1e-10 {
    100.0
  } else {
    10.0 * ((255.0 * 255.0) / mse).log10()
  }
}

fn compute_delta_e(img1: &Mat, img2: &Mat) -> Result<(f32, f32, f32)> {
  let mut lab1 = Mat::default();
  let mut lab2 = Mat::default();

  imgproc::cvt_color(img1, &mut lab1, imgproc::COLOR_BGR2Lab, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
  imgproc::cvt_color(img2, &mut lab2, imgproc::COLOR_BGR2Lab, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

  let rows = lab1.rows();
  let cols = lab1.cols();
  let mut delta_es = Vec::new();

  for y in 0..rows {
    for x in 0..cols {
      let p1 = lab1.at_2d::<core::Vec3b>(y, x)?;
      let p2 = lab2.at_2d::<core::Vec3b>(y, x)?;

      let l1 = p1[0] as f32 / 255.0 * 100.0;
      let a1 = p1[1] as f32 - 128.0;
      let b1 = p1[2] as f32 - 128.0;

      let l2 = p2[0] as f32 / 255.0 * 100.0;
      let a2 = p2[1] as f32 - 128.0;
      let b2 = p2[2] as f32 - 128.0;

      let dl = l1 - l2;
      let da = a1 - a2;
      let db = b1 - b2;

      let delta_e = (dl * dl + da * da + db * db).sqrt();
      delta_es.push(delta_e);
    }
  }

  let sum: f32 = delta_es.iter().sum();
  let avg = sum / delta_es.len() as f32;

  delta_es.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median = delta_es[delta_es.len() / 2];
  let max = delta_es[delta_es.len() - 1];

  Ok((avg, median, max))
}
