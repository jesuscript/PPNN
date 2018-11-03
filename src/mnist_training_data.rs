use na::{DMatrix};
use mnist::{Mnist, MnistBuilder};

pub struct TrainingData {
  pub images: DMatrix<u8>,
  pub labels: Vec<u8>
}

impl TrainingData {
  pub fn new(trn_size:u32,rows:u32,cols:u32) -> TrainingData {
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
      .label_format_digit()
      .training_set_length(trn_size)
      .validation_set_length(10_000)
      .test_set_length(10_000)
      .finalize();

    TrainingData {
      images: DMatrix::<u8>::from_row_slice(trn_size as usize, (rows*cols) as usize, &trn_img[..]),
      labels: trn_lbl
    }
  }
}
