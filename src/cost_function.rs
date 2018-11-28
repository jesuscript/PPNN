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
    (-y.component_mul(&a.map(|el| inf_to_max(el.ln()))) -
     (y.map(|el| 1.0 - el)).component_mul(&a.map(|el| inf_to_max((1.0-el).ln()) )))
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

fn inf_to_max(x:f32) -> f32{
  if x.is_infinite() { f32::MAX * x.signum() } else { x }
}


mod tests {
  use super::*;
  
  #[test]
  fn test_cross_entropy_raw(){
    assert_eq!(CrossEntropyCost::raw(
      &DVector::<f32>::from_row_slice(4, &[0.01,0.5,0.9,0.9]),
      &DVector::<f32>::from_row_slice(4, &[0.0,0.0,0.0,1.0])
    ), 0.01005 + 0.693147 + 2.302585 + 0.105361)
  }

  #[test]
  fn test_quadratic_raw(){
    assert_eq!(QuadraticCost::raw(
      &DVector::<f32>::from_row_slice(4, &[0.01,0.5,0.9,0.9]),
      &DVector::<f32>::from_row_slice(4, &[0.0,0.0,0.0,1.0])
    ), 0.53505)
  }

  #[test]
  fn test_cross_entropy_raw_nan(){
    let cost = CrossEntropyCost::raw(
      &DVector::<f32>::from_row_slice(2, &[0.0,1.0]),
      
      &DVector::<f32>::from_row_slice(2, &[0.0,1.0])
    );

    assert!(cost == 0.0);
  }
}

