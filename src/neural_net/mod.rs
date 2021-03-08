#[cfg(test)]
mod tests;

use std::{fmt, fmt::Debug};

use crate::perceptron::neuron::{
    Neuron,
    Type::{Logistic, Linear}
};
use crate::classifier::{
    Classifier,
    Resettable
};
use crate::perceptron::helpers::{
    logistic_derivitive,
    linear_derivitive
};
use crate::common::within_difference;

struct Layer {
    neurons: Vec<Neuron>,
    act_input: Vec<f64>,
    last_input: Vec<f64>,
    last_output: Vec<f64>
}

const LEARNING_RATE: f64 = 1f64;
const ACCEPTABLE_LOSS: f64 = 0.0001;

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Layer {
        let mut p = Vec::with_capacity(outputs);
        for _ in 0..outputs {
            p.push(Neuron::new(inputs, Logistic));
        }
        Layer {
            neurons: p,
            act_input: vec!(0f64; outputs),
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
        let mut act_input = Vec::with_capacity(self.neurons.len());

        for i in 0..self.neurons.len() {
            let mut inp: f64 = input.iter().enumerate()
                .map(|(j, v)| v * self.neurons[i].weights()[j])
                .sum();
            inp += self.neurons[i].bias();
            output.push(self.neurons[i].activate(inp).certainty());
            act_input.push(inp);
        }
        Some((output, act_input))
    }

    pub fn classify_for_training(&mut self, input: &Vec<f64>) -> Option<&Vec<f64>> {
        let (last_out, last_in) = self.activate(input)?;
        self.last_input = input.clone();
        self.last_output = last_out;
        self.act_input = last_in;
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Layer:")
         .field("neurons", &self.neurons)
         .finish()
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>,
    learning_rate: f64,
    input_normalise_factor: Vec<i8>
}

// used for the "new" operator which defaults to using
// a single hidden layer
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

fn loss(error: &Vec<f64>) -> f64 {
    error.iter().sum::<f64>().powi(2)
}

// constructs a custom neural net with hidden layers of
// the provided size in the patter 
// neural_net!((4; 3), 2, 5)
// which creates a neural net with 4 inputs, 3 outputs,
// the first hidden layer has 2 neurons, and the second 5
#[macro_export]
macro_rules! neural_net {
    (($i:expr; $o:expr) $(, $h:expr )*) => {
        return NeuralNet::new_custom($i, $o, vec!($($h,)*));
    };
}

impl NeuralNet {
    /// Returns a NeuralNet with `inputs` number of inputs,
    /// and `outputs` number of outputs (as f64).
    pub fn new(inputs: usize, outputs: usize) -> NeuralNet {
        let hiddens = hidden_layer_size(inputs, outputs);
        let mut l = Vec::with_capacity(2);
        l.push(Layer::new(inputs, hiddens)); // hidden layer
        l.push(Layer::new(hiddens, outputs)); // output layer
        NeuralNet {
            layers: l,
            learning_rate: LEARNING_RATE,
            input_normalise_factor: vec!(0; inputs)
        }
    }

    /// Returns a NeuralNet with `inputs` number of inputs,
    /// `outputs` number of outputs (as f64), and `hiddens.len()`
    /// number of hidden layers, where the ith layer contains
    /// hiddens\[i\] neurons.
    pub fn new_custom(inputs: usize, outputs: usize, hiddens: Vec<usize>) -> NeuralNet {
        let length = hiddens.len();
        if length == 0 {
            return NeuralNet::new(inputs, outputs);
        }
        let mut neurons = vec!();
        neurons.push(Layer::new(inputs, hiddens[0]));
        for i in 0..(length - 1) {
            neurons.push(Layer::new(hiddens[i], hiddens[i+1]));
        }
        neurons.push(Layer::new(hiddens[length-1], outputs));
        return NeuralNet {
            layers: neurons,
            learning_rate: LEARNING_RATE,
            input_normalise_factor: vec!(0; inputs)
        };
    }

    fn classify_for_training(&mut self, datum: &Vec<f64>) -> Option<&Vec<f64>> {
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

    // returns the loss
    fn learn(&mut self, input: &Vec<f64>, expect: &Vec<f64>) -> Option<f64> {
        // feed the input through
        // TODO: find a better way to get the type
        let derivitive = match self.layers[0].neurons[0].of_type() {
            Logistic => logistic_derivitive,
            Linear => linear_derivitive
        };
        let output = self.classify_for_training(input)?;
        let errorv = error(expect, output)?;
        let lossv = loss(&errorv);
        let output_layer = self.layers.get(self.layers.len() - 1).unwrap();
        let mut delta_i: Vec<f64> = errorv.into_iter().enumerate()
            .map(|(i, v)| v * derivitive(output_layer.act_input[i]))
            .collect();
        let mut next_delta_i: Vec<f64>;

        // back propagate the error
        for i in (0..self.layers.len()).rev() {

            // get the delta for the next layer ready
            next_delta_i = if i != 0 {
                let this_layer = &self.layers[i];
                let next_layer = &self.layers[i - 1];

                (0..next_layer.len())
                    .map(|j| derivitive(next_layer.act_input[j])).enumerate()
                    .map(
                        |(j, g_j)| 
                            g_j * (0..this_layer.neurons.len())
                            .map(|k| this_layer.neurons[k].weights()[j] * delta_i[k]).sum::<f64>()
                    ).collect()
            } else {
                vec!()
            };
            
            // update the weights for the current layer
            let this_layer = &mut self.layers[i];
            for (k, neuron) in this_layer.neurons.iter_mut().enumerate() {
                for (j, weight) in neuron.weights_mut().iter_mut().enumerate() {
                    *weight += self.learning_rate * this_layer.last_input[j] * delta_i[k];
                }
                *neuron.bias_mut() += self.learning_rate * delta_i[k];
            }
            delta_i = next_delta_i;
        }
        Some(lossv)
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    // assumes all given data has the same length
    fn set_normalise_factor(&mut self, data: &[Vec<f64>]) -> Option<()> {
        let mut factors = vec!(0i8; data.get(0)?.len());
        for datum in data.iter() {
            for (f, attr) in datum.iter().enumerate() {
                while (*attr * 10f64.powi(factors[f] as i32)).abs() > 1f64 {
                    factors[f] -= 1;
                }
            }
        }
        self.input_normalise_factor = factors;
        Some(())
    }

    #[cfg(test)]
    pub fn get_weight(&self, layer: usize, neuron: usize, weight: usize) -> &f64 {
        &self.layers[layer].neurons[neuron].weights()[weight]
    }

    #[cfg(test)]
    pub fn get_weight_mut(&mut self, layer: usize, neuron: usize, weight: usize) -> &mut f64 {
        &mut self.layers[layer].neurons[neuron].weights_mut()[weight]
    }

    #[cfg(test)]
    pub fn get_bias(&self, layer: usize, neuron: usize) -> f64 {
        self.layers[layer].neurons[neuron].bias()
    }

    #[cfg(test)]
    pub fn get_bias_mut(&mut self, layer: usize, neuron: usize) -> &mut f64 {
        self.layers[layer].neurons[neuron].bias_mut()
    }
}

// TODO: extend this to support any datatype that supports Into<Vec<f64>>
impl Classifier<Vec<f64>, Vec<f64>> for NeuralNet {
    fn train(&mut self, data: &[Vec<f64>], expect: &[Vec<f64>]) -> bool {
        if self.set_normalise_factor(data).is_none() {
            return false;
        }
        let mut datum = vec!(0f64; data.get(0).unwrap().len());
        let mut avg_loss = 1f64;
        while !within_difference(&avg_loss, &0f64, &ACCEPTABLE_LOSS) {
            let mut loss_sum = 0f64;
            for (datum_raw, expected) in data.iter().zip(expect.iter()) {
                for (i, attr) in datum_raw.iter().enumerate() {
                    datum[i] = attr * 10f64.powi(self.input_normalise_factor[i] as i32);
                }
                loss_sum += self.learn(&datum, expected).unwrap();
            }
            avg_loss = loss_sum / data.len() as f64;
            // println!("average loss = {}", avg_loss);
        }
        true
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

impl Resettable for NeuralNet {
    fn reset(&mut self) -> bool {
        let mut did_reset = true;
        for layer in self.layers.iter_mut() {
            did_reset &= layer.reset();
        }
        did_reset
    }
}

impl Debug for NeuralNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNet:")
         .field("layers", &self.layers)
         .finish()
    }
}

// example from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#[test]
fn test_learn_function() {
    use crate::compare_floats;

    let mut l = Vec::with_capacity(2);
    l.push(Layer::new(2, 2));
    l.push(Layer::new(2, 2));

    // setup the starting conditions
    let mut nn = NeuralNet::new_custom(2, 2, vec!(2));
    *nn.get_weight_mut(0, 0, 0) = 0.15;
    *nn.get_weight_mut(0, 0, 1) = 0.2;
    *nn.get_weight_mut(0, 1, 0) = 0.25;
    *nn.get_weight_mut(0, 1, 1) = 0.3;
    *nn.get_weight_mut(1, 0, 0) = 0.4;
    *nn.get_weight_mut(1, 0, 1) = 0.45;
    *nn.get_weight_mut(1, 1, 0) = 0.5;
    *nn.get_weight_mut(1, 1, 1) = 0.55;
    *nn.get_bias_mut(0, 0) = 0.35;
    *nn.get_bias_mut(0, 1) = 0.35;
    *nn.get_bias_mut(1, 0) = 0.6;
    *nn.get_bias_mut(1, 1) = 0.6;
    nn.set_learning_rate(0.5);

    // test the forward pass
    let input = vec!(0.05, 0.1);
    let expect = vec!(0.01, 0.99);
    let actual = nn.classify(&input);
    compare_floats!(&actual[0], &0.75136, &0.00001);
    compare_floats!(&actual[1], &0.77292, &0.00001);

    // test the back propagation
    nn.learn(&input, &expect);
    compare_floats!(nn.get_weight(1, 0, 0), &0.35891648, &0.00000001);
    compare_floats!(nn.get_weight(1, 0, 1), &0.408666186, &0.000000001);
    compare_floats!(nn.get_weight(1, 1, 0), &0.511301270, &0.000000001);
    compare_floats!(nn.get_weight(1, 1, 1), &0.561370121, &0.000000001);
    compare_floats!(nn.get_weight(0, 0, 0), &0.149780716, &0.000000001);
    compare_floats!(nn.get_weight(0, 0, 1), &0.19956143, &0.00000001);
    compare_floats!(nn.get_weight(0, 1, 0), &0.24975114, &0.00000001);
    compare_floats!(nn.get_weight(0, 1, 1), &0.29950229, &0.00000001);
}