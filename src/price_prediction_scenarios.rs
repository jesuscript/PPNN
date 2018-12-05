use elapsed::measure_time;
use network_initializer;
use sgd_network::*;
use self::NetworkOption::*;
use cost_function::*;
use price_data;


pub fn full(){
  let price_data = price_data::PriceData::new("data/Bitfinex_ETHUSD_1h.json", 336, 0, 0.2);

  let mut network = Network::<CrossEntropyCost>::new::<network_initializer::Scaled>(&[
    price_data.inputs_size,
    100,
    price_data.outputs_size
  ])
    .eta(0.1)
    .epochs(150)
    .mini_batch_size(10)
    .lambda(0.0)
    .options(&[
      MonitorTrainingCost,
      MonitorEvaluationCost,
      MonitorTrainingAccuracy,
      MonitorEvaluationAccuracy
    ]);


  let (elapsed, _) = measure_time(|| {
    network.sgd(price_data.training_data, Some(&price_data.test_data));

    //println!("{} {}",price_data.test_data[0].1, network.feedforward(&price_data.test_data[0].0));

  });

  println!("elapsed = {}", elapsed);
}
