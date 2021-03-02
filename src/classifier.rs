pub trait Classifier<T, C> {
    // returns true if training was successful
    fn train(&mut self, data: &Vec<T>, expect: &Vec<C>) -> bool;
    fn classify(&self, datum: &T) -> C;
}