use crate::perceptron::Perceptron;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative}
};
use crate::classifier::Classifier;

#[test]
fn test_perceptron_learning() {
    let data: Vec<Vec<f64>> = vec!(
        vec!(4.5, 4.8),
        vec!(4.7, 4.5),
        vec!(4.7, 4.1),
        vec!(4.9, 6.0),
        vec!(5.1, 5.0),
        vec!(5.1, 5.1),
        vec!(5.2, 3.4),
        vec!(5.2, 5.3),
        vec!(5.3, 4.4),
        vec!(5.3, 5.4),
        vec!(5.4, 5.8),
        vec!(5.5, 5.7),
        vec!(5.5, 5.6),
        vec!(5.6, 3.8),
        vec!(5.6, 5.0),
        vec!(5.6, 5.1),
        vec!(5.7, 5.4),
        vec!(5.7, 5.6),
        vec!(5.7, 5.8),
        vec!(5.7, 5.9),
        vec!(5.7, 6.0),
        vec!(5.8, 3.4),
        vec!(5.8, 4.3),
        vec!(5.8, 4.5),
        vec!(5.8, 5.6),
        vec!(5.9, 3.7),
        vec!(5.9, 4.3),
        vec!(5.9, 4.4),
        vec!(5.9, 4.7),
        vec!(5.9, 5.5),
        vec!(5.9, 5.6),
        vec!(5.9, 6.0),
        vec!(6.0, 4.1),
        vec!(6.0, 4.2),
        vec!(6.0, 4.3),
        vec!(6.0, 4.7),
        vec!(6.0, 6.6),
        vec!(6.1, 4.1),
        vec!(6.1, 4.3),
        vec!(6.1, 4.4),
        vec!(6.1, 4.5),
        vec!(6.1, 4.7),
        vec!(6.1, 5.9),
        vec!(6.1, 6.0),
        vec!(6.1, 6.2),
        vec!(6.1, 6.5),
        vec!(6.1, 6.7),
        vec!(6.1, 6.8),
        vec!(6.2, 4.4),
        vec!(6.2, 4.6),
        vec!(6.2, 4.7),
        vec!(6.2, 6.5),
        vec!(6.3, 6.5),
        vec!(6.6, 6.9)
    );
    let expect: Vec<Classification> = vec!(
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Negative(0f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Negative(0f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Positive(1f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Positive(1f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
        Negative(0f32),
        Negative(0f32),
        Negative(0f32),
        Positive(1f32),
        Positive(1f32),
        Positive(1f32),
    );

    assert_eq!(data.len(), expect.len());
    let mut p = Perceptron::new_linear(2);

    p.train(&data, &expect);
    for (i, expected) in expect.iter().enumerate() {
        assert!(p.classify(&data[i]).class_match(expected));
    }
    assert!(p.classify(&vec!(5.4, 5.0)).positive());
    assert!(p.classify(&vec!(7.0, 7.5)).positive());
    assert!(p.classify(&vec!(4.5, 3.8)).positive());
    assert!(p.classify(&vec!(7.0, 5.0)).negative());
    assert!(p.classify(&vec!(5.0, 2.5)).negative());
    assert!(p.classify(&vec!(5.4, 3.5)).negative());
}