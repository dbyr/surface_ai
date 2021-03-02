use crate::classification::Classification;

pub trait Classifier<T> {
    fn train(&mut self, data: &Vec<T>);
    fn classify(&self, datum: &T) -> Classification;
}