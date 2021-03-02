use crate::classification::{
    Classification,
    Classification::Unclassifiable
};

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
            weights: vec!(0f64; size),
            act_func: activation_func,
            reweight_func: reweight_func,
            bias: 0f64,
            learning_rate: 0.75
        }
    }

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