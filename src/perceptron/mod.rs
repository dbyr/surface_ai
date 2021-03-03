#[cfg(test)]
mod tests;

mod neuron;
mod helpers;

use std::{
    fmt,
    fmt::Debug
};

use neuron::{
    Neuron,
    LEARNING_RATE
};
use helpers::{
    linear_function,
    linear_reweight,
    logistic_function,
    logistic_reweight
};

use crate::classifier::{
    Classifier,
    Resettable
};
use crate::classification::Classification;

const LEARNING_RATE_DECAY: f64 = 0.1;

// TODO: add extra options such as allowing
// deteriorating learning rate, and batch
// learning
pub struct Perceptron {
    neuron: Neuron
}

impl Perceptron {

    pub fn new_linear(size: usize) -> Self {
        Perceptron {
            neuron: Neuron::new(
                size,
                Box::new(linear_function),
                Box::new(linear_reweight)
            )
        }
    }

    pub fn new_logistic(size: usize) -> Self {
        Perceptron {
            neuron: Neuron::new(
                size,
                Box::new(logistic_function),
                Box::new(logistic_reweight)
            )
        }
    }

    pub fn input_size(&self) -> usize {
        self.neuron.input_size()
    }

    // returns true if the perceptron learned something
    pub fn stochastic_learn(&mut self, datum: &Vec<f64>, expected: &Classification) -> bool {
        self.neuron.learn(datum, expected)
    }
}

impl Classifier<Vec<f64>, Classification> for Perceptron {

    fn train(&mut self, data: &[Vec<f64>], expect: &[Classification]) -> bool {
        if data.len() != expect.len() {
            return false;
        }

        let mut learning_rate = LEARNING_RATE;
        let mut previous_w = self.neuron.weights().clone();
        let mut previous_b = self.neuron.bias();
        // let mut iters = 0..;
        loop {
            for (i, datum) in data.iter().enumerate() {
                self.stochastic_learn(datum, &expect[i]);
            }
            if !self.neuron.weights_compare(&mut previous_w, &mut previous_b) {
                break;
            }
            learning_rate -= learning_rate * LEARNING_RATE_DECAY;
            self.neuron.set_learning_rate(learning_rate);
            // iters.next();
        }
        // println!("{:?} iterations done", iters.next());
        self.neuron.set_learning_rate(LEARNING_RATE);
        true
    }

    fn classify(&self, datum: &Vec<f64>) -> Classification {
        self.neuron.classify(datum)
    }
}

impl Resettable for Perceptron {
    fn reset(&mut self) -> bool {
        self.neuron.reset()
    }
}

impl Debug for Perceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Perceptron:")
         .field("neuron", &self.neuron)
         .finish()
    }
}