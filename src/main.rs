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


mod image_data;
mod price_data;
mod network_initializer;
mod cost_function;
mod sigmoid;
mod sgd_network;
mod image_recognition_scenarios;
mod price_prediction_scenarios;



fn main() {
  //price_prediction_scenarios::full();
  image_recognition_scenarios::full();
}


