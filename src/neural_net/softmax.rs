use rand::Rng;

use super::INITIAL_WEIGHT_RANGE;
use crate::classification::{
    Classification,
    Classification::{
        Unclassifiable,
        Probs
    }
};

pub struct Softmax {
    pub weights: Vec<f64>,
    // pub bias: f64,
    // pub learning_rate: f64,
    pub last_output: Vec<f64>
}

impl Softmax {
    pub fn new(
        size: usize
    ) -> Self {
        let mut rng = rand::thread_rng();
        Softmax {
            weights: vec!(0; size).into_iter()
                .map(|_| rng.gen_range(0f64..INITIAL_WEIGHT_RANGE))
                .collect(),
            // bias: rng.gen_range(0f64..INITIAL_WEIGHT_RANGE),
            // learning_rate: LEARNING_RATE,
            last_output: vec!(0f64; size)
        }
    }

    // returns a vec of the probabilities that the data
    // provided is of the classes represented by the
    // output vector's indexes
    pub fn thoughts(&self, data: &Vec<f64>) -> Classification {
        if data.len() != self.weights.len() {
            return Unclassifiable("Input length incompatible".to_string());
        }

        // normalise the data to prevent overflows
        let constant = -data.iter().max_by(
            |l, r| match *r - *l {
                x if x < 0f64 => std::cmp::Ordering::Greater,
                x if x > 0f64 => std::cmp::Ordering::Less,
                _ => std::cmp::Ordering::Equal
            }
        ).unwrap();
        let mut total = 0f64;
        let spread = data.iter().map(
            |v| {
                let cur = (v + constant).exp();
                total += cur;
                cur
            }
        ).collect::<Vec<f64>>();
        Probs(spread.iter().map(|v| v / total).collect::<Vec<f64>>())
    }

    pub fn classify_for_training(&mut self, data: &Vec<f64>) -> &Vec<f64> {
        self.last_output = self.thoughts(data).thoughts();
        &self.last_output
    }

    // returns the class (as a number from 0 to #classes - 1,
    // or -1 if the data cannot be used for this neuron)
    // and the certainty with which this class was selected
    pub fn classify(&self, data: &Vec<f64>) -> (i32, f64) {
        let (mut index, mut prob) = (0, 0f64);
        match self.thoughts(data) {
            Probs(vs) => for (i, v) in vs.iter().enumerate() {
                if *v > prob {
                    prob = *v;
                    index = i as i32;
                }
            },
            _ => {
                index = -1;
                prob = 1f64;
            }
        }
        (index, prob)
    }
}