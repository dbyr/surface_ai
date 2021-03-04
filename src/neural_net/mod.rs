use crate::perceptron::Perceptron;

struct Layer {
    neurons: Vec<Perceptron>
}

impl Layer {
    pub fn new(size: usize, outputs: usize) -> Layer {
        let mut p = Vec::with_capacity(size);
        for _ in 0..size {
            p.push(Perceptron::new_logistic(outputs));
        }
        Layer {
            neurons: p
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

    fn back_propogate(&mut self) {

    }
}