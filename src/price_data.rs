use na::{DVector,DMatrix,Dynamic};
use std::{f32,fs};

type TrainingData = Vec<(DVector<f32>, DVector<f32>)>;

pub struct PriceData {
  pub training_data: TrainingData,
  pub test_data: TrainingData,
  pub inputs_size: usize,
  pub outputs_size: usize
}

#[derive(Deserialize,Debug)]
struct Record {
  target: Vec<f32>,
  inputs: Vec<f32>
}


impl PriceData {
  pub fn new(path: &str, input_size: usize, target_offset: usize, test_pct: f32) -> PriceData {
    assert!(test_pct >= 0.0 && test_pct <= 1.0, "test_pct has to be between 0 and 1");

    let recs_json = fs::read_to_string(path).expect("Unable to read file");
    
    let records: Vec<Record> = serde_json::from_str(&recs_json).unwrap();

    let inputs_len = records[0].inputs.len();

    println!("{} records loaded",records.len());


    let full_data: TrainingData = records.iter().skip(input_size+target_offset).enumerate().map(|(i, rec)| {
      let flat_inputs:Vec<f32> = records.iter().skip(i).take(input_size).flat_map(|r| r.inputs.clone()).collect();
      
      let inputs = DVector::<f32>::from_row_slice(inputs_len*input_size, flat_inputs.as_slice());

      (inputs, DVector::<f32>::from_row_slice(rec.target.len(), rec.target.as_slice()))
    }).collect();

    let train_data_len = ((full_data.len()  as f32)*(1.0-test_pct)).round() as usize;

    println!("{} training records, {} test records",train_data_len, full_data.len() - train_data_len);

    let (inputs_size, outputs_size) = (full_data[0].0.len(), full_data[0].1.len());

    println!("{} inputs, {} outputs", inputs_size, outputs_size);


    PriceData {
      training_data: full_data[..train_data_len].to_vec(),
      test_data: full_data[train_data_len..].to_vec(),
      inputs_size,
      outputs_size
    }
  }
}

