use std::f64::consts::E;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative}
};

// activation function for classic perceptron
pub fn linear_function(x: f64) -> Classification {
    if x > 0f64 {
        Positive(1f32)
    } else {
        Negative(0f32)
    }
}

// activation function for sigmoid-perceptrons
pub fn logistic_function(x: f64) -> Classification {
    let result = 1f64 / (1f64 + E.powf(-x));
    if result > 0.5 {
        Positive(result as f32)
    } else {
        Negative(result as f32)
    }
}

// reweight function for classic perceptron
pub fn linear_reweight(
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
pub fn logistic_reweight(
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