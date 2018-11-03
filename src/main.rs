#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;

mod mnist_training_data;
mod sgd_network;

use sgd_network::{Sizes,Network};

fn main() {
  let sizes = Sizes::new(256,10,10);

  let network = Network::new(sizes);

  println!("{}", network);

  let training_data = mnist_training_data::TrainingData::new(10,28,28);

  println!("{:?}", training_data.labels);

}
