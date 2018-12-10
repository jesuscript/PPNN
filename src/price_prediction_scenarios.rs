use elapsed::measure_time;
use network_initializer;
use sgd_network::*;
use sgd_network::{Network,MonitoringOption::*};
use cost_function::*;
use price_data;


pub fn full(){
  let price_data = price_data::PriceData::new("data/Bitfinex_ETHUSD_1h.json", 6, 0, 0.2);

  let mut network = Network::<CrossEntropyCost>::new::<network_initializer::Scaled>(&[
    price_data.inputs_size,
    50,
    price_data.outputs_size
  ])
    .eta(0.05)
    .epochs(100)
    .mini_batch_size(10)
    .lambda(0.0)
    .monitoring_options(&[
      MonitorTrainingCost,
      MonitorEvaluationCost,
      //MonitorTrainingAccuracy,
      //MonitorEvaluationAccuracy
    ]);


  let (elapsed, _) = measure_time(|| {
      
    network.sgd(price_data.training_data, Some(&price_data.test_data));

    price_data.test_data.iter().take(10).for_each(|d|{
      println!("exp: {} act: {}",d.1, network.feedforward(&d.0));
    });
    

  });

  println!("elapsed = {}", elapsed);
}
