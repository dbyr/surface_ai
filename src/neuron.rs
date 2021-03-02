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
    reweight_func: Box<dyn Fn(
        &mut Vec<f64>,
        &Vec<f64>,
        &mut f64,
        &f64,
        &Classification,
        &Classification
    )>,
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

    pub fn learn(&mut self, input: &Vec<f64>, expected: &Classification) {
        let f_x = self.classify(input);

        // saves doing a O(input.len()) operation,
        // even though the weights wouldn't change
        // anyway if f_x == expected
        if !f_x.class_match(expected) {
            (self.reweight_func)(
                &mut self.weights,
                input,
                &mut self.bias,
                &self.learning_rate,
                expected,
                &f_x
            );
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
fn linear_reweight(
    weights: &mut Vec<f64>,
    input: &Vec<f64>,
    bias: &mut f64,
    learning_rate: &f64,
    expected: &Classification,
    actual: &Classification
) {

    // value by which all weights are changed
    let mut reweight_by = if expected.positive() {
        1f64
    } else {
        0f64
    };
    reweight_by -= if actual.positive() {
        1f64
    } else {
        0f64
    };
    reweight_by *= learning_rate;

    // all weights get reweighted, including bias
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight += reweight_by * input[i];
    }
    *bias += reweight_by; // bias "input" always 1
}

// reweight function for sigmoid-perceptrons
fn logistic_reweight(
    weights: &mut Vec<f64>,
    input: &Vec<f64>,
    bias: &mut f64,
    learning_rate: &f64,
    expected: &Classification,
    actual: &Classification
) {

    // get the linear values out of the classifications
    let actual_rate = actual.certainty() as f64;
    if actual_rate < 0f64 { return; }
    let expected_rate = expected.certainty() as f64;
    if expected_rate < 0f64 { return; }

    let reweight_by = learning_rate * (expected_rate - actual_rate)
                    * actual_rate * (1f64 - actual_rate);
    
    // all weights get reweighted, including bias
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight += reweight_by * input[i];
    }
    *bias += reweight_by; // bias "input" always 1
}