#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;
extern crate rand;
#[macro_use] extern crate itertools;
extern crate elapsed;

use itertools::Itertools;
use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};
use elapsed::measure_time;


mod image_data;
mod sgd_network;


use sgd_network::{Network};


fn main() {
  let image_data = image_data::ImageData::new(1000,1000,28,28);
  let mut network = Network::new(&[784,100,10]);

  let (elapsed, _) = measure_time(|| {
    network.sgd(image_data.training_data, 3, 10, 0.1, Some(&image_data.test_data));
  });

  println!("elapsed = {}", elapsed);
}


