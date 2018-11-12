use na::{DVector};
use mnist::{Mnist, MnistBuilder};

pub struct ImageData {
  training_size: u32,
  test_size: u32,
  pub training_data: Vec<(DVector<f32>, DVector<f32>)>,
  pub test_data: Vec<(DVector<f32>, DVector<f32>)>
}

impl ImageData {
  pub fn new(training_size:u32,test_size:u32,rows:usize,cols:usize) -> ImageData {
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
      .label_format_digit()
      .training_set_length(training_size)
      //.validation_set_length(10_000)
      .test_set_length(test_size)
      .finalize();

    ImageData {
      training_size,
      test_size,
      training_data: make_data(trn_img,trn_lbl,rows,cols),
      test_data: make_data(tst_img,tst_lbl,rows,cols)
    }
  }
}

fn make_data(img: Vec<u8>, lbl: Vec<u8>,rows:usize,cols:usize) -> Vec<(DVector<f32>, DVector<f32>)> {
  img.chunks(rows*cols).zip(lbl).map(make_datum).collect()
}

fn make_datum((img, l):(&[u8], u8)) -> (DVector<f32>,DVector<f32>) {
  let img = DVector::<f32>::from_iterator(img.len(), img.iter().map(|&i| i as f32));
  let lbl = DVector::<f32>::from_iterator(10 as usize, (0..10).enumerate().map(
    |(i,v)| if i==l as usize {1.0} else {0.0})
  );

  (img,lbl)
}
