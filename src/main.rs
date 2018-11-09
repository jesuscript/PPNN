#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nalgebra as na;
extern crate mnist;
extern crate rand;

mod image_data;
mod sgd_network;

use sgd_network::{Network};

use na::{DVector,Vector,Vector3,DMatrix,Dynamic,Matrix,MatrixMN};

fn main() {
  // let sizes = Sizes::new(784,30,10);

  // let network = Network::new(sizes);

  // let training_data = image_data::ImageData::new(10,28,28);
  let network = Network::new(&[2,3,2]);
  println!("{}", network);
  //println!("{:?}", network.backprop());

  //println!("{}", network.feedforward_step(&DVector::<f32>::from_row_slice(2 as usize, &[10.0,-1.0]), 0 as usize))
  //println!("{}", network.feedforward(&DVector::<f32>::from_row_slice(2 as usize, &[10.0,-1.0])));

  //println!("{}", &DMatrix::<f32>::zeros(3 as usize, 2 as usize));

  let v1 = DVector::<f32>::from_row_slice(2 as usize, &[1.0,2.0]);
  let v2 = DVector::<f32>::from_row_slice(2 as usize, &[3.0,4.0]);

  let m1 = DMatrix::<f32>::from_row_slice(2,3, &[
    1.0,2.0,3.0,
    10.0,20.0,30.0
  ]);

  for (x,y) in v1.iter().zip(v2.iter()){
    println!("{},{}", x, y);
  }

  let v3 = DVector::<f32>::from_row_slice(6 as usize, &[0.1,0.2,0.3,0.4,0.3,0.2]);

  println!("{}", v3.imax());

  println!("{}", if 1==2  {1} else {0});

  // let v3:Vec<f32> = v1.iter().zip(v2.iter()).map(|(x,y)| x+y).collect();

  // println!("{:?}", v3);

  // let n = 10;

  // let vec: Vec<u32> = (0..n).collect();
  
  // thread_rng().shuffle(&mut vec);

  // println!("{:?}", vec);


  // for c in vec.chunks(3){
  //   println!("chunk: {:?}", c);
  // }


  //println!("{}", DMatrix::<f32>::from_fn(v1.nrows(), v2.nrows(), |r,c| v1.get());

}
