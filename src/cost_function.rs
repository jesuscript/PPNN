use na::{DVector};
use std::f32;
use std::ops::Sub;

use sigmoid::Sigmoid;


pub trait CostFunction{
  fn raw(a:&DVector<f32>, y:&DVector<f32>) -> f32;
  fn delta(a:&DVector<f32>, y:&DVector<f32>, z:&DVector<f32>) -> DVector<f32>;
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CrossEntropyCost;
#[derive(Serialize, Deserialize, Clone)]
pub struct QuadraticCost;

impl CostFunction for CrossEntropyCost{
  fn raw(a:&DVector<f32>, y:&DVector<f32>) -> f32 {
    (-y.component_mul(&a.map(|el| el.ln())) - (y.map(|el| 1.0 - el)).component_mul(&a.map(|el| (1.0-el).ln())))
      .iter().sum()
  }

  fn delta(a:&DVector<f32>, y:&DVector<f32>, z:&DVector<f32>) -> DVector<f32>{
    a - y
  }
}



impl CostFunction for QuadraticCost {
  fn raw(a:&DVector<f32>, y:&DVector<f32>) -> f32 {
    0.5 * (a-y).norm_squared()
  }
  
  fn delta(a:&DVector<f32>, y:&DVector<f32>, z:&DVector<f32>) -> DVector<f32> {
    (a-y).component_mul(&z.sigmoid_prime())
  }
}


mod tests {
  use super::*;
  
  #[test]
  fn test_cross_entropy_raw(){
    assert_eq!(CrossEntropyCost::raw(
      DVector::<f32>::from_row_slice(4, &[0.01,0.5,0.9,0.9]),
      DVector::<f32>::from_row_slice(4, &[0.0,0.0,0.0,1.0])
    ), 0.01005 + 0.693147 + 2.302585 + 0.105361)
  }

  #[test]
  fn test_quadratic_raw(){
    assert_eq!(QuadraticCost::raw(
      DVector::<f32>::from_row_slice(4, &[0.01,0.5,0.9,0.9]),
      DVector::<f32>::from_row_slice(4, &[0.0,0.0,0.0,1.0])
    ), 0.53505)
  }
}

