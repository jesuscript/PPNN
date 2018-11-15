use std::borrow::Cow;
use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};
use rand::prelude::*;
use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
use itertools::Itertools;
use rayon::prelude::*;
use std::fs;

#[derive(Serialize, Deserialize, Debug, Clone)]
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

  pub fn new_from_file(path: &str) -> Network{
    let net_json = fs::read_to_string(path).expect("Unable to read file");

    serde_json::from_str(&net_json).unwrap()
  }


  pub fn sgd(&mut self, mut training_data: Vec<(DVector<f32>,DVector<f32>)>, epochs:u16, mini_batch_size:usize, eta:f32, test_data:Option<&[(DVector<f32>,DVector<f32>)]>) {
    let mut test_n =0;
    
    if test_data.is_some(){
      test_n = test_data.unwrap().len();
    }
    
    for i in 0..epochs {
      thread_rng().shuffle(&mut training_data);

      training_data.chunks(mini_batch_size).for_each(|mini_batch| self.update_mini_batch(&mini_batch, eta));

      if test_data.is_some(){
        println!("Epoch {}: {}/{}", i, self.evaluate(test_data.unwrap()), test_n);
      }else{
        println!("Epoch {} complete", i);
      }
    }
  }

  pub fn save_to_file(&self, path: &str){
    fs::write(path, serde_json::to_string(&self).unwrap()).expect("Unable to write file");
  }

  pub fn evaluate(&self, test_data: &[(DVector<f32>,DVector<f32>)]) -> u32{
    test_data.par_iter().fold_with(0, |sum, (input,output)| if self.feedforward(input).imax() == output.imax() {
      sum+1
    } else {
      sum
    }).reduce(|| 0, |a,b| a+b)
  }
  
  fn weighted_inputs(&self, a:&DVector<f32>, layer: usize) -> DVector<f32>{
    let (w,b) = (&self.weights[layer], &self.biases[layer]);

    let wi:Vec<f32> = (0..self.sizes[layer]).map(|r| w.row(r).transpose().dot(&a) + b[r]).collect();

    DVector::<f32>::from_row_slice(self.sizes[layer], &wi)
  }

  fn feedforward(&self, input:&DVector<f32>) -> DVector<f32>{
    (0..self.num_layers).fold(input.clone(), |a, l| self.weighted_inputs(&a,l).sigmoid())
  }

  fn update_mini_batch(&mut self, mini_batch: &[(DVector<f32>,DVector<f32>)], eta: f32){
    let bw_zeros = (
      (0..self.num_layers).map(|l| DVector::<f32>::zeros(self.biases[l].nrows())).collect(),
      (0..self.num_layers).map(|l| DMatrix::<f32>::zeros(self.weights[l].nrows(), self.weights[l].ncols())).collect()
    );
    
    let (nb, nw) = mini_batch.par_iter().fold_with(bw_zeros.clone(), |a,(i,o)| {
      add_delta(a, &self.backprop(i,o))
    }).reduce(|| bw_zeros.clone(), |a, b|{
      add_delta(a,&b)
    });

    let k = eta/ mini_batch.len() as f32;
    self.weights = self.weights.iter().zip(nw).map(|(w,nw)| w - k*nw).collect();
    self.biases = self.biases.iter().zip(nb).map(|(b,nb)| b - k*nb).collect();
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

    nabla_w.insert(0,make_nabla_w(&delta, &activations[self.num_layers - 1]));
    
    deltas.insert(0,delta);

    

    for l in (0..(self.num_layers - 1)).rev() {
      let sp = zs[l].sigmoid_prime();
      let delta = DVector::<f32>::from_fn(self.sizes[l], |r,c| self.weights[l+1].column(r).dot(&deltas[0])*sp[r]);

      nabla_w.insert(0,make_nabla_w(&delta,&activations[l]));
      deltas.insert(0,delta);
    }

    return (deltas,nabla_w)
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

fn make_nabla_w(delta:&DVector<f32>, a:&DVector<f32>) -> DMatrix<f32> {
  DMatrix::<f32>::from_fn(delta.nrows(), a.nrows(), |r,c| delta[r] * a[c])
}

fn cost_derivative(out_activations:&DVector<f32>, target_out:&DVector<f32>) -> DVector<f32> {
  out_activations - target_out
}

fn rng() -> f32 {
  SmallRng::from_entropy().sample(StandardNormal) as f32
}

type BW = (Vec<DVector<f32>>,Vec<DMatrix<f32>>);

fn add_delta<I,J>((nb,nw):(I,J), (dnb,dnw): &BW) -> BW
  where 
        I: IntoIterator<Item = DVector<f32>>,
        J: IntoIterator<Item = DMatrix<f32>>
{
  (
    nb.into_iter().zip(dnb.iter()).map(|(a,b)| a+b).collect(),
    nw.into_iter().zip(dnw.iter()).map(|(a,b)| a+b).collect()
  )
}
