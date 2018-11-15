#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;
extern crate rand;
#[macro_use] extern crate itertools;
extern crate elapsed;
extern crate rayon;
#[macro_use] extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use elapsed::measure_time;

mod image_data;
mod sgd_network;


use sgd_network::{Network};


fn main() {
  let image_data = image_data::ImageData::new(50_000,10_000,28,28);
  let mut network = Network::new(&[784,30,10]);

  
  let (elapsed, _) = measure_time(|| {
    network.sgd(image_data.training_data, 10, 100, 3.0, Some(&image_data.test_data));
  });

  println!("elapsed = {}", elapsed);
}


