use std::fmt::{self,Display, Formatter};
use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};
use rand::prelude::*;
use rand::distributions::StandardNormal;


pub struct Network {
  input_size: usize,
  num_layers: usize,
  sizes: DVector<usize>,
  biases: Vec<DVector<f32>>,
  weights: Vec<DMatrix<f32>>
}

impl Network {
  pub fn new(s:&[usize]) -> Network {
    let sizes = DVector::<usize>::from_row_slice(s.len() as usize, s);
    
    let mut biases = vec![];
    let mut weights = vec![];

    for (i, size) in sizes.iter().skip(1).enumerate() {
      //biases.push(DMatrix::<f32>::new_random(*size as usize,1));
      biases.push(DVector::<f32>::from_fn(*size, |r, c| rng() ));

      weights.push(DMatrix::<f32>::from_fn(*size,sizes[i], |r,c| rng() ));
    }

    Network{
      input_size: sizes[0],
      num_layers: sizes.len() - 1,
      sizes: sizes.remove_row(0),
      biases: biases,
      weights: weights
    }
  }

  fn feedforward_step(&self, a:&DVector<f32>, layer: usize) -> DVector<f32>{
    let (w,b) = (&self.weights[layer], &self.biases[layer]);

    DVector::<f32>::from_fn(self.sizes[layer], |r, c| sigmoid(w.row(r).transpose().dot(&a) + b[r]))
  }
  
  fn feedforward(&self, input:&DVector<f32>) -> DVector<f32>{
    let mut a = self.feedforward_step(input,0);
    
    for layer in 1..self.num_layers {
      a = self.feedforward_step(&a,layer);
    }
    
    return a
  }

  pub fn backprop(&self, input:&DVector<f32>, output:&DVector<f32>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
    let mut nabla_b = vec![];
    let mut nabla_w = vec![];

    for layer in 0..self.num_layers {
      nabla_b.push(DVector::<f32>::zeros(self.biases[layer].nrows()));
      nabla_w.push(DMatrix::<f32>::zeros(self.weights[layer].nrows(), self.weights[layer].ncols()))
    }

    let mut activation = input;

    let mut activations = vec![activation];
    let mut zs = vec![];

    for layer in 0..self.num_layers {
      let z = DVector::<f32>::from_fn(self.sizes[layer], |r,c| w.row(r).transpose().dot(&activation) + b[r]);
      zs.push(z);

      activation = sigmoid(z);
      activations.push(activation);
    }

    

    return (nabla_b,nabla_w)
  }
}

impl Display for Network {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    write!(f, "Network[\nlayers:{},\nsizes:{}", self.num_layers, self.sizes);
    
    write!(f,",\nbiases: [");
    for c in self.biases.iter() {
      write!(f,"{}", c);
    }
    write!(f,"]");

    write!(f,",\nweights: [");
    for c in self.weights.iter() { 
      write!(f,"{}", c); //can be huge
      
      //write!(f, "Matrix of len: {} ", c.len());
    }

    write!(f,"]");
    
    write!(f,"]")
  }
}


fn sigmoid(z:f32) -> f32 {
  1.0 / (1.0 + std::f32::consts::E.powf(-z))
}

fn sigmoid_prime(z:f32) -> f32 {
  sigmoid(z) * (1.0 - sigmoid(z))
}

fn cost_derivative(out_activations:&DVector<f32>, target_out:&DVector<f32>) -> DVector<f32> {
  out_activations - target_out
}

fn rng() -> f32 {
  SmallRng::from_entropy().sample(StandardNormal) as f32
}


#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn sigmoid_test() {
    assert_eq!((sigmoid(0.0), sigmoid(1.0), sigmoid(-1.0)),
               (       0.5,          0.7310586,    0.26894143));
  }
}
