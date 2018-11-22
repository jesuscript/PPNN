use rand::prelude::*;
use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
use na::{DVector,DMatrix};
use std::f32;

pub trait NetworkInitializer {
  fn init(sizes: &DVector<usize>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>);
}

pub struct Basic;

impl NetworkInitializer for Basic {
  fn init(sizes: &DVector<usize>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>){ 
    (
      make_biases(sizes),
      sizes.iter().skip(1).enumerate().map(|(i, size)| DMatrix::<f32>::from_fn(*size,sizes[i], |r,c| rng() ))
        .collect()
    )
  }
}

pub struct Scaled;

impl NetworkInitializer for Scaled {
  fn init(sizes: &DVector<usize>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
    (
      make_biases(sizes),
      sizes.iter().skip(1).enumerate()
        .map(|(i,size)| DMatrix::<f32>::from_fn(*size,sizes[i], |r,c| rng()/(sizes[i] as f32).sqrt() ))
        .collect()
    )
  }
}

fn make_biases(sizes: &DVector<usize>) -> Vec<DVector<f32>> {
  sizes.iter().skip(1).map(|size| DVector::<f32>::from_fn(*size, |r, c| rng() )).collect()
}

fn rng() -> f32 {
  SmallRng::from_entropy().sample(StandardNormal) as f32
}
