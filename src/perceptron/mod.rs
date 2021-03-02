mod neuron;
mod helpers;

use neuron::Neuron;
use helpers::{
    linear_function,
    linear_reweight,
    logistic_function,
    logistic_reweight
};

use crate::classifier::Classifier;
use crate::classification::Classification;

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

    pub fn new_sigmoid(size: usize) -> Self {
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

    pub fn stochastic_learn(&mut self, datum: &Vec<f64>, expected: &Classification) {
        self.neuron.learn(datum, expected);
    }
}

impl Classifier<Vec<f64>, Classification> for Perceptron {

    fn train(&mut self, data: &Vec<Vec<f64>>, expect: &Vec<Classification>) -> bool {
        if data.len() != expect.len() {
            return false;
        }
        // TODO: get the training method to use the data
        // randomly instead of in order
        for (i, datum) in data.iter().enumerate() {
            self.stochastic_learn(datum, &expect[i]);
        }
        true
    }

    fn classify(&self, datum: &Vec<f64>) -> Classification {
        self.neuron.classify(datum)
    }
}