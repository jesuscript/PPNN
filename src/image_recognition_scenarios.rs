use elapsed::measure_time;
use network_initializer;
use sgd_network::*;
use self::NetworkOption::*;
use cost_function::*;
use image_data;

pub fn full(){
  let image_data = image_data::ImageData::new(1_000,10_000,28,28);

  let mut network = Network::<CrossEntropyCost>::new::<network_initializer::Scaled>(&[784,30,10])
    .eta(0.5)
    .epochs(1)
    .mini_batch_size(10)
    .lambda(1.0)
    .options(&[
      MonitorTrainingCost,
      MonitorTrainingAccuracy,
      MonitorEvaluationCost,
      MonitorEvaluationAccuracy
    ]);


  let (elapsed, _) = measure_time(|| {
    network.sgd(image_data.training_data, Some(&image_data.test_data));
  });

  println!("elapsed = {}", elapsed);
}
