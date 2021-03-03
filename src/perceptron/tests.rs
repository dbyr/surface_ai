use crate::perceptron::Perceptron;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative},
    average_certainty
};
use crate::classifier::Classifier;
use crate::test_methods::{
    expect_success_rate,
    cross_validation_testing
};

#[test]
fn test_linear_perceptron_learning() {
    let mut data = get_data();
    let mut expect = get_expect();

    assert_eq!(data.len(), expect.len());
    let mut p = Perceptron::new_linear(2);

    p.train(&data, &expect);
    assert!(expect_success_rate(&p, &data, &expect, 1.0));
    assert!(p.classify(&vec!(5.4, 5.0)).positive());
    assert!(p.classify(&vec!(7.0, 7.5)).positive());
    assert!(p.classify(&vec!(4.5, 3.8)).positive());
    assert!(p.classify(&vec!(7.0, 5.0)).negative());
    assert!(p.classify(&vec!(5.0, 2.5)).negative());
    assert!(p.classify(&vec!(5.4, 3.5)).negative());

    add_to_data(&mut data);
    add_to_expect(&mut expect);
    let mut p = Perceptron::new_linear(2);

    p.train(&data, &expect);
    assert!(expect_success_rate(&p, &data, &expect, 0.92));

    let mut p = Perceptron::new_linear(2);
    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9).unwrap();
    println!("Achieved {} strat result", strat_result);
    assert!(strat_result > 0.89);
}

#[test]
fn test_logistic_peceptron_learning() {
    let mut data = get_data();
    let mut expect = get_expect();

    let mut p = Perceptron::new_logistic(2);

    p.train(&data, &expect);
    println!("average certainty: {}", average_certainty(&mut data.iter().map(|x| p.classify(x))));
    assert!(expect_success_rate(&p, &data, &expect, 1f64));
    assert!(p.classify(&vec!(5.4, 5.0)).positive());
    assert!(p.classify(&vec!(7.0, 7.5)).positive());
    assert!(p.classify(&vec!(4.5, 3.8)).positive());
    assert!(p.classify(&vec!(7.0, 5.0)).negative());
    assert!(p.classify(&vec!(5.0, 2.5)).negative());
    assert!(p.classify(&vec!(5.4, 3.5)).negative());

    add_to_data(&mut data);
    add_to_expect(&mut expect);

    let mut p = Perceptron::new_logistic(2);

    p.train(&data, &expect);
    println!("average certainty: {}", average_certainty(&mut data.iter().map(|x| p.classify(x))));
    assert!(expect_success_rate(&p, &data, &expect, 0.92));

    let mut p = Perceptron::new_linear(2);
    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9).unwrap();
    println!("Achieved {} strat result", strat_result);
    assert!(strat_result > 0.89);
}



fn add_to_data(data: &mut Vec<Vec<f64>>) {
    data.append(&mut vec!(
        vec!(5.0, 4.0),
        vec!(5.3, 3.7),
        vec!(5.4, 4.2),
        vec!(5.5, 3.5),
        vec!(5.5, 3.9),
        vec!(5.9, 5.0),
        vec!(5.9, 5.5),
        vec!(6.0, 5.5),
        vec!(6.1, 5.6),
        vec!(6.3, 5.6),
        vec!(6.4, 5.7)
    ));
}

fn add_to_expect(expect: &mut Vec<Classification>) {
    expect.append(&mut vec!(
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64)
    ))
}

fn get_data() -> Vec<Vec<f64>> {
    vec!(
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
    )
}

fn get_expect() -> Vec<Classification> {
    vec!(
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
        Negative(0f64),
        Negative(0f64),
        Negative(0f64),
        Positive(1f64),
        Positive(1f64),
        Positive(1f64),
    )
}