use std::fmt::Debug;

use super::Neuron;
use crate::perceptron::neuron::Type::Logistic;
use crate::classifier::Resettable;

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation_input: Vec<f64>,
    pub last_input: Vec<f64>,
    pub last_output: Vec<f64>
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let mut p = Vec::with_capacity(outputs);
        for _ in 0..outputs {
            p.push(Neuron::new(inputs, Logistic));
        }
        Layer {
            neurons: p,
            activation_input: vec!(0f64; outputs),
            last_input: vec!(0f64; inputs),
            last_output: vec!(0f64; outputs)
        }
    }

    pub fn classify(&self, input: &Vec<f64>) -> Option<Vec<f64>> {
        if input.len() != self.neurons[0].input_size() {return None;}
        let mut output = Vec::with_capacity(self.neurons.len());

        for i in 0..self.neurons.len() {
            output.push(self.neurons[i].classify(&input).certainty());
        }
        Some(output)
    }

    fn activate(&self, input: &Vec<f64>) -> Option<(Vec<f64>, Vec<f64>)> {
        if input.len() != self.neurons[0].input_size() {return None;}
        let mut output = Vec::with_capacity(self.neurons.len());
        let mut activation_input = Vec::with_capacity(self.neurons.len());

        for i in 0..self.neurons.len() {
            let mut inp: f64 = input.iter().enumerate()
                .map(|(j, v)| v * self.neurons[i].weights()[j])
                .sum();
            inp += self.neurons[i].bias();
            output.push(self.neurons[i].activate(inp).certainty());
            activation_input.push(inp);
        }
        Some((output, activation_input))
    }

    pub fn classify_for_training(&mut self, input: &Vec<f64>) -> Option<&Vec<f64>> {
        let (last_out, last_in) = self.activate(input)?;
        self.last_input = input.clone();
        self.last_output = last_out;
        self.activation_input = last_in;
        Some(&self.last_output)
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn reset(&mut self) -> bool {
        let mut did_reset = true;
        for neuron in self.neurons.iter_mut() {
            did_reset &= neuron.reset();
        }
        did_reset
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer:")
         .field("neurons", &self.neurons)
         .finish()
    }
}