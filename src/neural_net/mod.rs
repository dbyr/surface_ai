use crate::perceptron::neuron::{
    Neuron,
    Type::{Logistic, Linear},
    LEARNING_RATE
};
use crate::classifier::Classifier;
use crate::perceptron::helpers::{
    logistic_derivitive,
    linear_derivitive
};

struct Layer {
    neurons: Vec<Neuron>,
    last_input: Vec<f64>,
    last_output: Vec<f64>
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let mut p = Vec::with_capacity(outputs);
        for _ in 0..outputs {
            p.push(Neuron::new(inputs, Logistic));
        }
        Layer {
            neurons: p,
            last_input: vec!(0f64; inputs),
            last_output: vec!(0f64; outputs)
        }
    }

    pub fn classify(&self, input: &Vec<f64>) -> Option<Vec<f64>> {
        if input.len() != self.neurons[0].input_size() {return None;}
        let mut output = Vec::with_capacity(self.neurons.len());

        for i in 0..self.last_output.len() {
            output[i] = self.neurons[i].classify(&input).certainty();
        }
        Some(output)
    }

    pub fn classify_for_training(&mut self, input: &Vec<f64>) -> Option<&Vec<f64>> {
        self.last_output = self.classify(input)?;
        self.last_input = input.clone();
        Some(&self.last_output)
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    // pub fn error(&self, expect: &Vec<f64>) -> Option<f64> {
    //     if self.last_output.len() != expect.len() {return None;}

    //     let mut total_err = 0f64;
    //     for i in 0..expect.len() {
    //         total_err += (expect[i] - self.last_output[i]).powi(2);
    //     }
    //     Some(total_err / 2f64)
    // }

    // returns the "delta k" to use for the next layer
    // (that is, the layer that feeds into this one)
    pub fn back_propogate(&mut self) {
        for neuron in self.neurons.iter_mut() {
            for weight in neuron.weights_mut().iter_mut() {

            }
        }
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>
}

#[inline]
fn hidden_layer_size(inputs: usize, outputs: usize) -> usize {
    ((inputs + outputs) as f64 * 1.7) as usize
}

fn error(expect: &Vec<f64>, actual: &Vec<f64>) -> Option<Vec<f64>> {
    if expect.len() != actual.len() {return None;}

    let mut error = Vec::with_capacity(expect.len());
    for (e, a) in expect.iter().zip(actual.iter()) {
        error.push(e - a);
    }
    Some(error)
}

impl NeuralNet {
    pub fn new(inputs: usize, outputs: usize) -> NeuralNet {
        let hiddens = hidden_layer_size(inputs, outputs);
        let mut l = Vec::with_capacity(3);
        l.push(Layer::new(inputs, hiddens));
        l.push(Layer::new(hiddens, outputs));
        l.push(Layer::new(outputs, 1));
        NeuralNet {
            layers: l
        }
    }

    pub fn classify_for_training(&mut self, datum: &Vec<f64>) -> Option<&Vec<f64>> {
        let mut input: Option<&Vec<f64>> = None;
        for layer in self.layers.iter_mut() {
            input = match input {
                Some(v) => layer.classify_for_training(&v),
                None => layer.classify_for_training(datum)
            };
        }
        match input {
            Some(v) => Some(v),
            None => None
        }
    }

    pub fn learn(&mut self, input: &Vec<f64>, expect: &Vec<f64>) -> Option<()> {
        // feed the input through
        let derivitive = match self.layers[0].neurons[0].of_type() {
            Logistic => logistic_derivitive,
            Linear => linear_derivitive
        };
        let output = self.classify_for_training(input)?;
        let errorv = error(expect, output)?;
        let output_layer_i = self.layers.len() - 1;
        let output_layer = self.layers.get(output_layer_i).unwrap();
        let mut delta_i: Vec<f64> = errorv
            .into_iter().enumerate()
            .map(|(i, v)| v * derivitive(output_layer.last_input[i]))
            .collect();
        let mut next_delta_i: Vec<f64>;

        // back propagate the error
        for i in (1..self.layers.len()).rev() {

            // get the delta for the next layer ready
            let this_layer = &self.layers[i].neurons;
            let next_layer = &self.layers[i - 1];
            next_delta_i = (0..next_layer.len())
                .map(|j| derivitive(next_layer.last_input[j])).enumerate()
                .map(
                    |(j, g_j)| 
                        g_j * (0..this_layer.len())
                        .map(|k| this_layer[k].weights()[j]).sum::<f64>()
                ).collect();
            
            // update the weights for the current layer
            // let last_input = &self.layers[i].last_input;
            let this_layer = &mut self.layers[i];
            for (k, neuron) in this_layer.neurons.iter_mut().enumerate() {
                for (i, weight) in neuron.weights_mut().iter_mut().enumerate() {
                    *weight += LEARNING_RATE * this_layer.last_input[i] * delta_i[k];
                }
            }
            delta_i = next_delta_i;
        }
        Some(())
    }
}

impl Classifier<Vec<f64>, Vec<f64>> for NeuralNet {
    fn train(&mut self, data: &[Vec<f64>], expect: &[Vec<f64>]) -> bool {
        false
    }

    // TODO: do proper error handling
    fn classify(&self, datum: &Vec<f64>) -> Vec<f64> {
        let mut input = None;
        for layer in self.layers.iter() {
            input = match input {
                Some(v) => layer.classify(&v),
                None => layer.classify(datum)
            };
        }
        match input {
            Some(v) => v,
            None => vec!()
        }
    }
}