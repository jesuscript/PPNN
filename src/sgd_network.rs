use std::fmt::{self,Display, Formatter};
use na::{Vector3,DMatrix,Dynamic,Matrix,MatrixMN};


pub type Sizes = Vector3<i32>;

pub struct Network {
  num_layers: usize,
  sizes: Sizes,
  biases: Vec<DMatrix<f32>>,
  weights: Vec<DMatrix<f32>>
}

impl Network {
  pub fn new(sizes:Sizes) -> Network {
    let mut biases = vec![];
    let mut weights = vec![];
    
    for (i, size) in sizes.iter().skip(1).enumerate() {
      biases.push(DMatrix::<f32>::new_random(*size as usize,1));

      weights.push(DMatrix::<f32>::new_random(*size as usize,sizes[i] as usize));
    }
    
    Network{
      sizes: sizes,
      num_layers: sizes.len(),
      biases: biases,
      weights: weights
    }
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
      //write!(f,"{}", c); //can be huge
      
      write!(f, "Matrix of len: {} ", c.len());
    }

    write!(f,"]");
    
    write!(f,"]")
  }
}
