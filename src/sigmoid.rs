use na::{DVector};

pub trait Sigmoid {
  fn sigmoid(&self) -> Self;
  fn sigmoid_prime(&self) -> Self;
}

impl Sigmoid for f32 {
  fn sigmoid(&self) -> f32 { 1.0 / (1.0 + std::f32::consts::E.powf(-self)) }
  fn sigmoid_prime(&self) -> f32 { self.sigmoid() * (1.0 - self.sigmoid()) }
}

impl Sigmoid for DVector<f32> {
  fn sigmoid(&self) -> DVector<f32> { self.map(|em| em.sigmoid()) }
  fn sigmoid_prime(&self) -> DVector<f32> { self.map(|em| em.sigmoid_prime()) }
}
