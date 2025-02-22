use crate::classifier::{Classifier, Resettable};
use rand::Rng;

pub fn shuffle_class_data<T, C>(data: &mut Vec<T>, classes: &mut Vec<C>) {
    let mut rng = rand::thread_rng();
    let mut i = data.len();
    let mut swap_to;
    // method ripped from rand crate
    while i >= 2 {
        i -= 1;
        swap_to = rng.gen_range(0..i + 1);
        data.swap(i, swap_to);
        classes.swap(i, swap_to);
    }
}

// returns the average fidelity of the
// classifier model when being trained on
// train_on_portion fraction of the given
// training data, and tested on 1-train_on_portion
// fraction
pub fn cross_validation_testing<M, T, C>(
    classifier: &mut M,
    data: &mut Vec<T>,
    expect: &mut Vec<C>,
    train_on_portion: f64,
    are_equal: &dyn Fn(&C, &C) -> bool
) -> Option<f64>
where M: Classifier<T, C> + Resettable, T: std::fmt::Debug {
    if data.len() != expect.len() { return None; }
    if train_on_portion <= 0f64 || train_on_portion >= 1f64 { return None; }
    shuffle_class_data(data, expect);

    let end = data.len();
    let test_step = (end as f64 * (1f64 - train_on_portion)) as usize;
    let mut cur_test_start = 0;
    let mut cur_test_end = test_step;
    let mut data_test;
    let mut expect_test;
    let mut correctness_sum = 0;
    let mut count = 0;

    while cur_test_start < end {
        data_test = Vec::with_capacity(test_step);
        expect_test = Vec::with_capacity(test_step);

        // separate the test and training portions
        while cur_test_start < cur_test_end {
            data_test.push(data.pop().unwrap());
            expect_test.push(expect.pop().unwrap());
            cur_test_start += 1;
        }

        // train and test
        classifier.train(data, expect);
        correctness_sum += sum_successes(classifier, &data_test, &expect_test, are_equal)?;
        count += data_test.len();
        classifier.reset();
        
        // update the portion trackers and return the
        // test data back to the original vecs
        cur_test_end += test_step;
        if cur_test_end > end { cur_test_end = end; }
        data_test.append(data);
        *data = data_test;
        expect_test.append(expect);
        *expect = expect_test;
    }
    Some(correctness_sum as f64 / (count as f64))
}

// returns the average error of the
// classifier model when being trained on
// train_on_portion fraction of the given
// training data, and tested on 1-train_on_portion
// fraction
pub fn cross_validation_testing_continuous<M, T, C>(
    classifier: &mut M,
    data: &mut Vec<T>,
    expect: &mut Vec<C>,
    train_on_portion: f64,
    loss: &dyn Fn(&C, &C) -> f64
) -> Option<f64>
where M: Classifier<T, C> + Resettable, T: std::fmt::Debug {
    if data.len() != expect.len() { return None; }
    if train_on_portion <= 0f64 || train_on_portion >= 1f64 { return None; }
    shuffle_class_data(data, expect);

    let end = data.len();
    let test_step = (end as f64 * (1f64 - train_on_portion)) as usize;
    let mut cur_test_start = 0;
    let mut cur_test_end = test_step;
    let mut data_test;
    let mut expect_test;
    let mut loss_sum = 0f64;
    let mut count = 0;

    while cur_test_start < end {
        data_test = Vec::with_capacity(test_step);
        expect_test = Vec::with_capacity(test_step);

        // separate the test and training portions
        while cur_test_start < cur_test_end {
            data_test.push(data.pop().unwrap());
            expect_test.push(expect.pop().unwrap());
            cur_test_start += 1;
        }

        // train and test
        classifier.train(data, expect);
        loss_sum += data_test.iter().zip(expect_test.iter())
            .map(|(dat, exp)| loss(&classifier.classify(dat), exp))
            .sum::<f64>();
        count += data_test.len();
        classifier.reset();
        
        // update the portion trackers and return the
        // test data back to the original vecs
        cur_test_end += test_step;
        if cur_test_end > end { cur_test_end = end; }
        data_test.append(data);
        *data = data_test;
        expect_test.append(expect);
        *expect = expect_test;
    }
    Some(loss_sum as f64 / (count as f64))
}

fn sum_successes<T, C>(
    classifier: &dyn Classifier<T, C>,
    data: &[T],
    expect: &[C],
    are_equal: &dyn Fn(&C, &C) -> bool
) -> Option<usize> {
    if data.len() != expect.len() { return None; }

    let mut correct = 0;
    for (datum, expected) in data.iter().zip(expect.iter()) {
        if are_equal(&classifier.classify(datum), expected) {
            correct += 1;
        }
    }
    Some(correct)
}

// return true if at least the provided
// success rate was achieved
// success_rate is a number between
// 0 and 1 representing a percentage
pub fn expect_success_rate<T, C>(
    classifier: &dyn Classifier<T, C>,
    data: &Vec<T>,
    expect: &Vec<C>,
    expect_rate: f64,
    are_equal: &dyn Fn(&C, &C) -> bool
) -> bool {
    if data.len() != expect.len() { return false; }

    let actual_rate = sum_successes(classifier, data, expect, are_equal).unwrap() as f64 / (data.len() as f64);
    println!("achieved {}, expected {}", actual_rate, expect_rate);
    actual_rate >= expect_rate
}

#[macro_export]
macro_rules! compare_floats {
    ($l:expr, $r:expr) => {
        if !reasonably_equal($l, $r) {
            println!("Left: {}, Right: {}", $l, $r);
        }
        assert!(reasonably_equal($l, $r));
    };

    ($l:expr, $r:expr, $d:expr) => {
        if !within_difference($l, $r, $d) {
            println!("Left: {}, Right: {}", $l, $r);
        }
        assert!(within_difference($l, $r, $d));
    };
}