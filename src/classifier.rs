pub trait Classifier<T, C> {
    // returns true if training was successful
    fn train(&mut self, data: &[T], expect: &[C]) -> bool;
    fn classify(&self, datum: &T) -> C;
}

pub trait Resettable {
    // should return the calling value to its
    // "just initialised" state
    // returns true if the reset was successful
    fn reset(&mut self) -> bool;
}