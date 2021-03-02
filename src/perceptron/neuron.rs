use std::{
    fmt,
    fmt::Debug
};

use super::helpers::reasonably_equal;

use crate::classification::{
    Classification,
    Classification::Unclassifiable
};

pub const LEARNING_RATE: f64 = 1f64;
const INITIAL_WEIGHT_VAL: f64 = 0f64;

pub struct Neuron {
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
    pub fn new(
        size: usize,
        activation_func: Box<dyn Fn(f64) -> Classification>,
        reweight_func: Box<dyn Fn(
            &mut Vec<f64>,
            &Vec<f64>,
            &mut f64,
            &f64,
            &Classification,
            &Classification
        )>
    ) -> Self {
        Neuron {
            weights: vec!(INITIAL_WEIGHT_VAL; size),
            act_func: activation_func,
            reweight_func: reweight_func,
            bias: INITIAL_WEIGHT_VAL,
            learning_rate: LEARNING_RATE
        }
    }

    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }
    pub fn bias(&self) -> f64 {
        self.bias
    }

    // compares the weights of the given values with
    // what's currently held, updating any that are not
    // equal and returning true only if anything was updated
    pub fn weights_compare(&self, weights: &mut Vec<f64>, bias: &mut f64) -> bool {
        if self.weights.len() != weights.len() { return false; }
        
        let mut updated = false;
        if !reasonably_equal(&self.bias, bias) {
            updated = true;
        }
        *bias = self.bias;
        for (l, r) in self.weights.iter().zip(weights.iter_mut()) {
            if !reasonably_equal(l, r) {
                updated = true;
            }
            *r = *l;
        }
        updated
    }

    pub fn set_learning_rate(&mut self, new_rate: f64) {
        self.learning_rate = new_rate;
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
        let mut input = self.bias;
        for i in 0..inputs.len() {
            input += inputs[i] * self.weights[i];
        }
        (self.act_func)(input)
    }

    // pub fn batch_learn(&mut self, )

    // returns true if the neuron learned
    pub fn learn(&mut self, input: &Vec<f64>, expected: &Classification) -> bool {
        let f_x = self.classify(input);
        if f_x.certainty() < 0f64 {
            return false;
        }

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
            true
        } else {
            false
        }
    }
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Neuron:")
         .field("weights", &self.weights)
         .field("bias", &self.bias)
         .field("learning_rate", &self.learning_rate)
         .finish()
    }
}