#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;
extern crate rand;

mod mnist_training_data;
mod sgd_network;

use sgd_network::{Network};

use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};

fn main() {
  // let sizes = Sizes::new(784,30,10);

  // let network = Network::new(sizes);

  // let training_data = mnist_training_data::TrainingData::new(10,28,28);
  let network = Network::new(&[2,3,2]);
  println!("{}", network);
  //println!("{:?}", network.backprop());

  //println!("{}", network.feedforward_step(&DVector::<f32>::from_row_slice(2 as usize, &[10.0,-1.0]), 0 as usize))
  //println!("{}", network.feedforward(&DVector::<f32>::from_row_slice(2 as usize, &[10.0,-1.0])));

  //println!("{}", &DMatrix::<f32>::zeros(3 as usize, 2 as usize));
}
