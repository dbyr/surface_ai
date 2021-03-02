use std::f64::consts::E;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative, Unclassifiable}
};

enum Type {
    Perceptron,
    SigPerceptron
}

pub(crate) struct Neuron {
    weights: Vec<f64>,
    act_func: Box<dyn Fn(f64) -> Classification>,
    reweight_func: Box<dyn Fn(&mut Vec<f64>, &Vec<f64>)>,
    bias: f64,
    learning_rate: f64
}

impl Neuron {

    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    // classifies the given input using the present
    // state of the neuron
    pub fn classify(&self, inputs: &Vec<f64>) -> Classification {
        
        // calculate the vector for the activation function input
        if inputs.len() != self.weights.len() {
            return Unclassifiable(
                format!("Input vector must be of length {}", self.input_size())
            );
        }
        let mut input = 0f64;
        for i in 0..inputs.len() {
            input += inputs[i] * self.weights[i];
        }
        (self.act_func)(input)
    }

    // pub fn batch_learn(&mut self, )

    pub fn learn(&mut self, input: &Vec<f64>, expected: Classification) {
        let f_x = self.classify(input);
        if !f_x.class_match(expected) {
            (self.reweight_func)(&mut self.weights, input);
        }
    }
}

// activation function for classic perceptron
fn linear_function(x: f64) -> Classification {
    if x > 0f64 {
        Positive(1f32)
    } else {
        Negative(0f32)
    }
}

// activation function for sigmoid-perceptrons
fn logistic_function(x: f64) -> Classification {
    let result = 1f64 / (1f64 + E.powf(-x));
    if result > 0.5 {
        Positive(result as f32)
    } else {
        Negative(result as f32)
    }
}

// reweight function for classic perceptron
fn linear_reweight(weights: &mut Vec<f64>, input: &Vec<f64>) {

}