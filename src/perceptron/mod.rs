#[cfg(test)]
mod tests;

pub(crate) mod neuron;
pub(crate) mod helpers;

use std::{
    fmt,
    fmt::Debug
};

use neuron::{
    Neuron,
    Type::{Linear, Logistic},
    LEARNING_RATE
};

use crate::classifier::{
    ClassifierBuilder,
    Classifier,
    Resettable
};
use crate::classification::Classification;

const LEARNING_RATE_DECAY: f64 = 0.9;

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
                Linear
            )
        }
    }

    pub fn new_logistic(size: usize) -> Self {
        Perceptron {
            neuron: Neuron::new(
                size,
                Logistic
            )
        }
    }

    pub fn input_size(&self) -> usize {
        self.neuron.input_size()
    }

    // returns the of the learning action
    pub fn stochastic_learn(&mut self, datum: &Vec<f64>, expected: &Classification) -> f64 {
        self.neuron.learn(datum, expected)
    }
}

impl ClassifierBuilder<Vec<f64>, Classification> for Perceptron {
    type Result = Perceptron;

    fn train(self, data: &[Vec<f64>], expect: &[Classification]) -> Option<Perceptron> {
        if data.len() != expect.len() {
            return None;
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
            learning_rate *= LEARNING_RATE_DECAY;
            self.neuron.set_learning_rate(learning_rate);
            // iters.next();
        }
        // println!("{:?} iterations done", iters.next());
        self.neuron.set_learning_rate(LEARNING_RATE);
        Some(self)
    }
}

impl Classifier<Vec<f64>, Classification> for Perceptron {

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