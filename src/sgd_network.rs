use std::borrow::Cow;
use na::{DVector,DMatrix};
use itertools::Itertools;
use rayon::prelude::*;
use std::fs;
use rand::thread_rng;
use rand::Rng;
use std::boxed::Box;
use std::marker::PhantomData;

use network_initializer::*;
use sigmoid::Sigmoid;
use cost_function::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Network <CF>{
  input_size: usize,
  num_layers: usize,
  sizes: DVector<usize>,
  biases: Vec<DVector<f32>>,
  weights: Vec<DMatrix<f32>>,
  phantom: PhantomData<CF>,
  eta: f32,
  epochs: u16,
  mini_batch_size: usize,
  lambda: f32,
  options: Vec<NetworkOption>
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NetworkOption {
  MonitorEvaluationCost,
  MonitorEvaluationAccuracy,
  MonitorTrainingCost,
  MonitorTrainingAccuracy
}

type EvaluationData<'a> = Option<&'a Vec<(DVector<f32>,DVector<f32>)>>;

impl <CF: CostFunction+std::marker::Sync> Network<CF> {
  
  pub fn new<I:NetworkInitializer>(s:&[usize]) -> Network<CF> {
    let sizes = DVector::<usize>::from_row_slice(s.len() as usize, s);
    let (biases, weights) = I::init(&sizes);

    Network{
      input_size: sizes[0],
      num_layers: sizes.len() - 1,
      sizes: sizes.remove_row(0),
      biases: biases,
      weights: weights,
      phantom: PhantomData,
      eta: 0.0,
      epochs: 1,
      mini_batch_size: 1,
      lambda: 5.0,
      options: vec![]
    }
  }

  pub fn new_from_file(path: &str) -> Network<CF>{
    let net_json = fs::read_to_string(path).expect("Unable to read file");

    serde_json::from_str(&net_json).unwrap()
  }


  pub fn sgd(&mut self,
             mut training_data: Vec<(DVector<f32>,DVector<f32>)>,
             evaluation_data: EvaluationData
  ) {
    assert!(self.eta > 0.0, "eta has to be greater than 0 but is {}. Use .eta(...) to set the eta.", self.eta);

    let train_n = training_data.len();

    for i in 0..self.epochs {
      thread_rng().shuffle(&mut training_data);

      training_data.chunks(self.mini_batch_size).for_each(|mini_batch| self.update_mini_batch(&mini_batch, train_n));

      println!("Epoch {} complete", i);

      self.monitor(&training_data, &evaluation_data);
    }
  }

  pub fn save_to_file(&self, path: &str){
    fs::write(path, serde_json::to_string(&self).unwrap()).expect("Unable to write file");
  }

  pub fn eta(mut self, eta:f32) -> Self { self.eta = eta; self }
  pub fn epochs(mut self, epochs:u16) -> Self { self.epochs = epochs; self }
  pub fn mini_batch_size(mut self, bs:usize) -> Self { self.mini_batch_size = bs; self }
  pub fn lambda(mut self, lambda:f32) -> Self { self.lambda = lambda; self }
  pub fn options(mut self, options:&[NetworkOption]) -> Self {
    self.options = options.to_vec();
    self
  }

  fn monitor(&self, train_data: &Vec<(DVector<f32>,DVector<f32>)>, eval_data: &EvaluationData) {
    self.options.iter().for_each(|opt| {
      use NetworkOption::*;
      
      match opt {
        MonitorTrainingCost => println!("Cost on training data: {}", self.total_cost(train_data)),
        MonitorTrainingAccuracy => {
          println!("Accuracy on training data: {} / {}", self.accuracy(train_data), train_data.len())
        },
        MonitorEvaluationCost => {
          assert!(eval_data.is_some(), "Evaluation data must be present for MonitorEvaluationAccuracy");

          println!("Cost on evaluation data: {}", self.total_cost(eval_data.unwrap()))
        },
        MonitorEvaluationAccuracy => {
          assert!(eval_data.is_some(), "Evaluation data must be present for MonitorEvaluationAccuracy");
          
          let data = eval_data.unwrap();

          println!("Accuracy on evaluation data: {} / {}", self.accuracy(data), data.len())
        },
      }
    });
  }

  fn weighted_inputs(&self, a:&DVector<f32>, layer: usize) -> DVector<f32>{
    let (w,b) = (&self.weights[layer], &self.biases[layer]);

    let wi:Vec<f32> = (0..self.sizes[layer]).map(|r| w.row(r).transpose().dot(&a) + b[r]).collect();

    DVector::<f32>::from_row_slice(self.sizes[layer], &wi)
  }

  fn feedforward(&self, input:&DVector<f32>) -> DVector<f32>{
    (0..self.num_layers).fold(input.clone(), |a, l| self.weighted_inputs(&a,l).sigmoid())
  }

  fn update_mini_batch(&mut self, mini_batch: &[(DVector<f32>,DVector<f32>)], train_len:usize){
    let bw_zeros = (
      (0..self.num_layers).map(|l| DVector::<f32>::zeros(self.biases[l].nrows())).collect(),
      (0..self.num_layers).map(|l| DMatrix::<f32>::zeros(self.weights[l].nrows(), self.weights[l].ncols())).collect()
    );
    
    let (nb, nw) = mini_batch.par_iter().fold_with(bw_zeros.clone(), |a,(i,o)| {
      add_delta(a, &self.backprop(i,o))
    }).reduce(|| bw_zeros.clone(), |a, b|{
      add_delta(a,&b)
    });

    let k = self.eta / mini_batch.len() as f32;
    let m = 1.0 - self.eta * self.lambda / train_len as f32;

    self.weights = self.weights.iter().zip(nw).map(|(w,nw)| m*w - k*nw).collect();
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
    
    let delta:DVector<f32> = CF::delta(&activations[self.num_layers], output, &zs[self.num_layers-1]);

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

  fn total_cost(&self, data: &Vec<(DVector<f32>,DVector<f32>)>) -> f32 {
    let n = data.len() as f32;
    
    data.par_iter().fold_with(0.0, |cost, (input,output)| {
      cost + CF::raw(&self.feedforward(input), output) / n
    }).reduce(|| 0.0, |a,b| a+b) +
      
      0.5*(self.lambda/n)*self.weights.iter().map(|w| w.norm_squared()).sum::<f32>()
  }

  fn accuracy(&self, data: &[(DVector<f32>,DVector<f32>)]) -> u32 {
    data.par_iter().fold_with(0, |sum, (input,output)| if self.feedforward(input).imax() == output.imax() {
      sum+1
    } else {
      sum
    }).reduce(|| 0, |a,b| a+b)
  }
}



fn make_nabla_w(delta:&DVector<f32>, a:&DVector<f32>) -> DMatrix<f32> {
  DMatrix::<f32>::from_fn(delta.nrows(), a.nrows(), |r,c| delta[r] * a[c])
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
