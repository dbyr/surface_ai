/// A trait for training classifiers that allows
/// the use and subsequent dump of data that is
/// only needed during training
/// should never be implemented directly, only through
/// implementing the ClassifierBuilder type
pub trait ClassifierTrainer<T, C>: ClassifierBuilder<T, C> {
    // returns a trained classifier
    fn build_once(self, data: &[T], expect: &[C]) -> Option<Self::Classifier>;
}

pub trait ClassifierBuilder<T, C> {
    type Classifier: Classifier<T, C>;

    // returns a trained classifier
    fn train(self, data: &[T], expect: &[C]) -> Self;
    fn build(&self) -> Option<Self::Classifier>;
}

impl<B, T, C> ClassifierTrainer<T, C> for B
where B: ClassifierBuilder<T, C> {
    #[inline]
    fn build_once(self, data: &[T], expect: &[C]) -> Option<Self::Classifier> {
        self.train(data, expect).build()
    }
}

pub trait Classifier<T, C> {
    fn classify(&self, datum: &T) -> C;
}

pub trait Resettable {
    // should return the calling value to its
    // "just initialised" state
    // returns true if the reset was successful
    fn reset(&mut self) -> bool;
}