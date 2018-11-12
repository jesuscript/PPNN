use std::fmt::{self,Display, Formatter};
use std::borrow::Cow;
use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};
use rand::prelude::*;
use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
use itertools::Itertools;

pub struct Network {
  input_size: usize,
  num_layers: usize,
  sizes: DVector<usize>,
  biases: Vec<DVector<f32>>,
  weights: Vec<DMatrix<f32>>
}

pub struct TrainItem {
  pub input: DVector<f32>,
  pub output: DVector<f32>
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

  fn weighted_inputs(&self, a:&DVector<f32>, layer: usize) -> DVector<f32>{
    let (w,b) = (&self.weights[layer], &self.biases[layer]);

    DVector::<f32>::from_fn(self.sizes[layer], |r, c| (w.row(r).transpose().dot(&a) + b[r]).sigmoid())
  }
  
  fn feedforward(&self, input:&DVector<f32>) -> DVector<f32>{
    //try using fold over layers
    let mut a = self.weighted_inputs(input,0).sigmoid();
    
    for layer in 1..self.num_layers {
      a = self.weighted_inputs(&a,layer).sigmoid();
    }
    
    return a
  }

  pub fn sgd(&mut self, mut training_data: Vec<(DVector<f32>,DVector<f32>)>, epochs:u16, mini_batch_size:usize,
             eta:f32, test_data:Option<&[(DVector<f32>,DVector<f32>)]>) {
    let mut test_n =0;
    
    if test_data.is_some(){
      test_n = test_data.unwrap().len();
    }
    
    let n = training_data.len();

    for i in 0..epochs {
      thread_rng().shuffle(&mut training_data);

      for mini_batch in training_data.chunks(mini_batch_size){
        self.update_mini_batch(&mini_batch, eta);
      }

      if test_data.is_some(){
        println!("Epoch {}: {}/{}", i, self.evaluate(test_data.unwrap()), test_n);
      }else{
        println!("Epoch {} complete", i);
      }
    }
  }

  fn update_mini_batch(&mut self, mini_batch: &[(DVector<f32>,DVector<f32>)], eta: f32){
    let mut nabla_b = vec![];
    let mut nabla_w = vec![];
    
    for layer in 0..self.num_layers {
      nabla_b.push(DVector::<f32>::zeros(self.biases[layer].nrows()));
      nabla_w.push(DMatrix::<f32>::zeros(self.weights[layer].nrows(), self.weights[layer].ncols()))
    }

    for (input,output) in mini_batch {
      let (delta_nabla_b, delta_nabla_w) = self.backprop(&input,&output);

      // println!("{}", nabla_b.iter().format(","));
      // println!("{}", delta_nabla_b.iter().format(", "));
      nabla_b = nabla_b.iter().zip(delta_nabla_b.iter()).map(|(nb,dnb)| nb+dnb).collect();
      nabla_w = nabla_w.iter().zip(delta_nabla_w.iter()).map(|(nw,dnw)| nw+dnw).collect();
    }

    let k = eta/ mini_batch.len() as f32;

    self.weights = self.weights.iter().zip(nabla_w.iter()).map(|(w,nw)| w - k*nw).collect();
    self.biases = self.biases.iter().zip(nabla_b.iter()).map(|(b,nb)| b - k*nb).collect();
  }

  fn backprop(&self, input:&DVector<f32>, output:&DVector<f32>) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
    let mut deltas = vec![]; // nabla_b === deltas
    let mut nabla_w = vec![];


    let mut activations = vec![input.clone()];
    let mut zs = vec![];

    for layer in 0..self.num_layers {
      let z = self.weighted_inputs(&activations[layer], layer);
      
      activations.push(z.sigmoid());
      zs.push(z);
    }

    let delta:DVector<f32> = cost_derivative(&activations[self.num_layers], output).component_mul(
      &zs[self.num_layers-1].sigmoid_prime()
    );
    
    nabla_w.insert(0,delta_from_upper_layer(&delta, &activations[self.num_layers - 1]));
    deltas.insert(0,delta);

    for l in (0..=(self.num_layers - 2)).rev() {
      let sp = zs[l].sigmoid_prime();
      //deltas[0] is the last inserted deltas, i.e. for l+1
      let delta = DVector::<f32>::from_fn(self.sizes[l], |r,c| self.weights[l+1].column(r).dot(&deltas[0])*sp[r]);

      nabla_w.insert(0,delta_from_upper_layer(&delta,&activations[l]));
      deltas.insert(0,delta);
    }

    return (deltas,nabla_w)
  }

  fn evaluate(&self, test_data: &[(DVector<f32>,DVector<f32>)]) -> i32{
    test_data.iter().fold(0, |sum, (input,output)| if self.feedforward(input).imax() == output.imax() {
      sum+1
    } else {
      sum
    })
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

trait Sigmoid {
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

fn delta_from_upper_layer(delta:&DVector<f32>, a:&DVector<f32>) -> DMatrix<f32> {
  DMatrix::<f32>::from_fn(delta.nrows(), a.nrows(), |r,c| delta[r] * a[c])
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
