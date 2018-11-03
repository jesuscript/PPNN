#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;

use std::fmt::{self,Display, Formatter};
use na::{Vector3,DMatrix,Dynamic,Matrix,MatrixMN};
use mnist::{Mnist, MnistBuilder};

type Sizes = Vector3<i32>;

struct Network {
  num_layers: usize,
  sizes: Sizes,
  biases: Vec<DMatrix<f32>>,
  weights: Vec<DMatrix<f32>>
}

impl Network {
  fn new(sizes:Sizes) -> Network {
    let mut biases = vec![];
    let mut weights = vec![];
    
    for (i, size) in sizes.iter().skip(1).enumerate() {
      biases.push(DMatrix::<f32>::new_random(*size as usize,1));

      weights.push(DMatrix::<f32>::new_random(*size as usize,sizes[i] as usize));
    }
    
    Network{
      sizes: sizes,
      num_layers: sizes.len(),
      biases: biases,
      weights: weights
    }
  }
}

impl Display for Network {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    write!(f, "Network[\nlayers:{},\nsizes:{}", self.num_layers, self.sizes);
    
    write!(f,",\nbiases: [");
    for c in self.biases.iter() {
      write!(f,"{}", c);
    }
    write!(f,"]");

    write!(f,",\nweights: [");
    for c in self.weights.iter() { 
      //write!(f,"{}", c); //can be huge
      
      write!(f, "Matrix of len: {} ", c.len());
    }

    write!(f,"]");
    
    write!(f,"]")
  }
}

struct TrainingData {
  images: DMatrix<u8>,
  labels: Vec<u8>
}

impl TrainingData {
  fn new(trn_size:u32,rows:u32,cols:u32) -> TrainingData {
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

fn main() {
  let sizes = Sizes::new(256,10,10);

  let network = Network::new(sizes);

  println!("{}", network);

  let training_data = TrainingData::new(10,28,28);

  println!("{:?}", training_data.labels);

}
