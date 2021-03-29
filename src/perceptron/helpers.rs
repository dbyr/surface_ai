use std::f64::consts::E;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative}
};

// activation function for classic perceptron
pub fn linear_function(x: f64) -> Classification {
    if x > 0f64 {
        Positive(1f64)
    } else {
        Negative(0f64)
    }
}

// activation function for sigmoid-perceptrons
pub fn logistic_function(x: f64) -> Classification {
    let result = 1f64 / (1f64 + E.powf(-x));
    if result > 0.5 {
        Positive(result)
    } else {
        Negative(result)
    }
}

fn linear_reweight_by(
    expected: &Classification,
    actual: &Classification,
    learning_rate: &f64
) -> f64 {
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
    reweight_by * learning_rate
}

// derivitive of the activation functions
pub fn linear_derivitive(x: f64) -> f64 {
    x
}
pub fn logistic_derivitive(x: f64) -> f64 {
    let h_x = logistic_function(x).certainty();
    h_x * (1f64 - h_x)
}

// reweight function for classic perceptron
// returns the "modified error" vector of the
// inputs
pub fn linear_reweight(
    weights: &mut Vec<f64>,
    input: &Vec<f64>,
    bias: &mut f64,
    learning_rate: &f64,
    expected: &Classification,
    actual: &Classification
) {

    // value by which all weights are changed
    let reweight_by = linear_reweight_by(expected, actual, learning_rate);

    // all weights get reweighted, including bias
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight += reweight_by * input[i];
    }
    *bias += reweight_by; // bias "input" always 1
}

// reweight function for sigmoid-perceptrons
// returns the "modified error" vector of the
// inputs
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
    if actual_rate < 0f64 { return;}// Vec::new(); }
    let expected_rate = expected.certainty() as f64;
    if expected_rate < 0f64 { return;}// Vec::new(); }
    let modified_error = expected_rate - actual_rate;

    let reweight_by = learning_rate * modified_error
                    * logistic_derivitive(actual_rate);
    
    // all weights get reweighted, including bias
    // let mut errors = Vec::with_capacity(input.len());
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight += reweight_by * input[i];
        // errors.push(modified_error * logistic_derivitive(x: f64))
    }
    *bias += reweight_by;

}

pub fn cross_entropy_loss(expected: &Vec<f64>, actual: &Vec<f64>) -> f64 {
    let mut sum = 0f64;
    for (p, q) in expected.iter().zip(actual.iter()) {
        sum += p * q.log10();
    }
    -sum
}