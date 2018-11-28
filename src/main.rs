#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;
extern crate rand;
extern crate elapsed;
extern crate rayon;
extern crate serde;
extern crate serde_json;
#[macro_use] extern crate itertools;
#[macro_use] extern crate serde_derive;

use elapsed::measure_time;

mod image_data;
mod network_initializer;
mod cost_function;
mod sigmoid;
mod sgd_network;


use network_initializer::NetworkInitializer;
use sgd_network::*;
use sgd_network::NetworkOption::*;
use cost_function::*;


fn main() {
  let image_data = image_data::ImageData::new(50_000,10_000,28,28);

  let mut network = Network::<CrossEntropyCost>::new::<network_initializer::Scaled>(&[784,100,10])
    .eta(0.5)
    .epochs(30)
    .mini_batch_size(10)
    .lambda(5.0)
    .options(&[
      MonitorTrainingCost,
      MonitorTrainingAccuracy,
      MonitorEvaluationCost,
      MonitorEvaluationAccuracy
    ]);

  
  let (elapsed, _) = measure_time(|| {
    network.sgd(image_data.training_data, Some(&image_data.test_data));
  });

  println!("elapsed = {}", elapsed);
}


