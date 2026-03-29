/// Base image structure and conversion utilities
/// This module provides the foundational BasedImage struct (like "Leg" class)
use opencv::{core, prelude::*};

/// Fast image buffer for pixel manipulation (Base class like "Leg")
/// Stores image data as a flat BGR array for direct pixel access
#[derive(Clone)]
pub struct BasedImage {
  pub w: usize,      // Image width in pixels
  pub h: usize,      // Image height in pixels
  pub data: Vec<u8>, // BGR pixel data (3 bytes per pixel)
}

impl BasedImage {
  /// Creates a BasedImage from an OpenCV Mat.
  /// Copies the image data from Mat into a Vec<u8> for fast pixel access.
  ///
  /// # Arguments
  /// * `mat` - OpenCV Mat containing BGR image data
  ///
  /// # Returns
  /// BasedImage with copied pixel data
  pub fn from_mat(mat: &Mat) -> Self {
    // Get image dimensions
    let w = mat.cols() as usize; // Image width in pixels
    let h = mat.rows() as usize; // Image height in pixels

    // Clone Mat to ensure it's continuous in memory (required for data_bytes())
    let continuous_mat = if mat.is_continuous() {
      mat.clone()
    } else {
      mat.clone() // Clone always creates a continuous Mat
    };

    // Extract raw pixel data from OpenCV Mat
    let slice = continuous_mat.data_bytes().unwrap();
    let data = slice.to_vec(); // Copy to owned Vec for mutation

    Self { w, h, data }
  }

  /// Converts BasedImage back to OpenCV Mat.
  /// Creates a new Mat and copies the pixel data into it.
  ///
  /// # Returns
  /// OpenCV Mat with BGR image data
  pub fn to_mat(&self) -> Mat {
    // Create a new Mat with same dimensions and CV_8UC3 type (8-bit unsigned, 3 channels)
    let mut mat = unsafe {
      Mat::new_rows_cols(
        self.h as i32, // Height in pixels
        self.w as i32, // Width in pixels
        core::CV_8UC3, // 8-bit BGR format
      )
      .unwrap()
    };
    // Copy pixel data from our buffer into the Mat
    mat.data_bytes_mut().unwrap().copy_from_slice(&self.data);
    mat
  }
}
