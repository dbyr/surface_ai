use std::str::FromStr;

use crate::perceptron::Perceptron;
use crate::classification::{
    Classification,
    Classification::{Positive, Negative},
    average_certainty
};
use crate::classifier::{
    ClassifierBuilder,
    Classifier
};
use crate::test_methods::{
    expect_success_rate,
    cross_validation_testing
};
use crate::common::read_dataset_file;

// dataset downloaded from:
// https://archive.ics.uci.edu/ml/datasets/banknote+authentication
const BANKNOTE_FILE: &str = "./data/data_banknote_authentication.txt";

fn banknote_from_line(line: String) -> Option<(Vec<f64>, Classification)> {
    let parts: Vec<&str> = line.split(",").collect();
    let mut attrs = Vec::new();
    for i in 0..4 {
        attrs.push(f64::from_str(parts[i]).unwrap());
    }
    let out = if parts[4] == "0" {
        Negative(0f64)
    } else {
        Positive(1f64)
    };
    Some((attrs, out))
}

#[inline]
fn read_banknote_file() -> (Vec<Vec<f64>>, Vec<Classification>) {
    match read_dataset_file(BANKNOTE_FILE, &banknote_from_line, false) {
        Some(v) => v,
        None => panic!("Test file '{}' not found", BANKNOTE_FILE)
    }
}

#[test]
fn test_linear_banknotes() {
    let (mut data, mut expect) = read_banknote_file();

    let mut p = Perceptron::new_linear(4);
    p = p.train(&data, &expect).unwrap();
    assert!(expect_success_rate(&p, &data, &expect, 0.95, &|l, r| *l == *r));
    println!("{:?}", p);

    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9, &|l, r| *l == *r).unwrap();
    println!("Achieved {} strat result", strat_result);
    assert!(strat_result > 0.95);
}

#[test]
fn test_logistic_banknotes() {
    let (mut data, mut expect) = read_banknote_file();

    let mut p = Perceptron::new_logistic(4);
    p = p.train(&data, &expect).unwrap();
    assert!(expect_success_rate(&p, &data, &expect, 0.95, &|l, r| *l == *r));
    println!("{:?}", p);

    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9, &|l, r| *l == *r).unwrap();
    println!("Achieved {} strat result", strat_result);
    assert!(strat_result > 0.95);
}

#[test]
fn test_linear_perceptron_learning() {
    let mut data = get_data();
    let mut expect = get_expect();

    assert_eq!(data.len(), expect.len());
    let mut p = Perceptron::new_linear(2);

    p = p.train(&data, &expect).unwrap();
    assert!(expect_success_rate(&p, &data, &expect, 1.0, &|l, r| *l == *r));
    assert!(p.classify(&vec!(5.4, 5.0)).positive());
    assert!(p.classify(&vec!(7.0, 7.5)).positive());
    assert!(p.classify(&vec!(4.5, 3.8)).positive());
    assert!(p.classify(&vec!(7.0, 5.0)).negative());
    assert!(p.classify(&vec!(5.0, 2.5)).negative());
    assert!(p.classify(&vec!(5.4, 3.5)).negative());

    add_to_data(&mut data);
    add_to_expect(&mut expect);
    let mut p = Perceptron::new_linear(2);

    p = p.train(&data, &expect).unwrap();
    assert!(expect_success_rate(&p, &data, &expect, 0.92, &|l, r| *l == *r));

    let mut p = Perceptron::new_linear(2);
    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9, &|l, r| *l == *r).unwrap();
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
    assert!(expect_success_rate(&p, &data, &expect, 1f64, &|l, r| *l == *r));
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
    assert!(expect_success_rate(&p, &data, &expect, 0.92, &|l, r| *l == *r));

    let mut p = Perceptron::new_linear(2);
    let strat_result = cross_validation_testing(&mut p, &mut data, &mut expect, 0.9, &|l, r| *l == *r).unwrap();
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