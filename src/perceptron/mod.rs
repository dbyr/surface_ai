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

pub struct LinearPerceptron {
    neuron: Neuron
}

pub struct SigmoidPerceptron {
    neuron: Neuron
}

impl LinearPerceptron {

    pub fn new(size: usize) -> Self {
        LinearPerceptron {
            neuron: Neuron::new(
                size,
                Box::new(linear_function),
                Box::new(linear_reweight)
            )
        }
    }

    pub fn input_size(&self) -> usize {
        self.neuron.input_size()
    }
}

impl SigmoidPerceptron {

    pub fn new(size: usize) -> Self {
        SigmoidPerceptron {
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
}

impl Classifier<Vec<f64>> for LinearPerceptron {

    fn train(&mut self, data: &Vec<Vec<f64>>) {

    }

    fn classify(&self, datum: &Vec<f64>) -> Classification {
        self.neuron.classify(datum)
    }
}

impl Classifier<Vec<f64>> for SigmoidPerceptron {

    fn train(&mut self, data: &Vec<Vec<f64>>) {

    }

    fn classify(&self, datum: &Vec<f64>) -> Classification {
        self.neuron.classify(datum)
    }
}