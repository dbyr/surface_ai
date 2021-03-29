use std::str::FromStr;

use crate::common::read_dataset_file;
use crate::neural_net::NeuralNet;
use crate::classifier::Classifier;
use crate::test_methods::{
    cross_validation_testing,
    expect_success_rate
};

use StarCat::{
    RedDwarf,
    BrownDwarf,
    WhiteDwarf,
    MainSeq,
    SuperGiant,
    HyperGiant,
    Other
};

// dataset downloaded from:
// https://www.kaggle.com/deepu1109/star-dataset
const STAR_FILE: &str = "./data/star_data.csv";

#[derive(Debug, PartialEq)]
enum StarCat {
    RedDwarf,
    BrownDwarf,
    WhiteDwarf,
    MainSeq,
    SuperGiant,
    HyperGiant,
    Other
}

impl From<u32> for StarCat {
    fn from(s: u32) -> Self {
        match s {
            0 => BrownDwarf,
            1 => RedDwarf,
            2 => WhiteDwarf,
            3 => MainSeq,
            4 => SuperGiant,
            5 => HyperGiant,
            _ => Other
        }
    }
}
impl From<StarCat> for u32 {
    fn from(s: StarCat) -> Self {
        match s {
            BrownDwarf => 0,
            RedDwarf => 1,
            WhiteDwarf => 2,
            MainSeq => 3,
            SuperGiant => 4,
            HyperGiant => 5,
            Other => 6
        }
    }
}
impl From<&StarCat> for u32 {
    fn from(s: &StarCat) -> Self {
        match s {
            BrownDwarf => 0,
            RedDwarf => 1,
            WhiteDwarf => 2,
            MainSeq => 3,
            SuperGiant => 4,
            HyperGiant => 5,
            Other => 6
        }
    }
}

impl From<Vec<f64>> for StarCat {
    fn from(v: Vec<f64>) -> Self {
        if v.len() != 6 {
            return Other;
        }
        let mut found = false;
        let mut u32val = 6;
        for i in 0..v.len() {
            if v[i].round() == 1f64 {
                if found {
                    return Other;
                } else {
                    found = true;
                }
                u32val = i;
            }
        }
        StarCat::from(u32val as u32)
    }
}
impl From<&Vec<f64>> for StarCat {
    fn from(v: &Vec<f64>) -> Self {
        if v.len() != 6 {
            return Other;
        }
        let mut found = false;
        let mut u32val = 6;
        for i in 0..v.len() {
            if v[i].round() == 1f64 {
                if found {
                    return Other;
                } else {
                    found = true;
                }
                u32val = i;
            }
        }
        StarCat::from(u32val as u32)
    }
}

impl From<StarCat> for Vec<f64> {
    fn from(s: StarCat) -> Self {
        let mut r = vec!(0f64; 6);
        let i = u32::from(&s) as usize;
        if i == 6 {
            return r;
        } else {
            r[i] = 1f64;
        }
        r
    }
}
impl From<&StarCat> for Vec<f64> {
    fn from(s: &StarCat) -> Self {
        let mut r = vec!(0f64; 6);
        let i = u32::from(s) as usize;
        if i == 6 {
            return r;
        } else {
            r[i] = 1f64;
        }
        r
    }
}

fn read_star_information(line: String) -> (Vec<f64>, Vec<f64>) {
    let parts: Vec<&str> = line.split(",").collect();
    let mut attrs = Vec::new();
    for i in 0..4 {
        attrs.push(f64::from_str(parts[i]).unwrap());
    }
    let out = Vec::<f64>::from(StarCat::from(u32::from_str(parts[4]).unwrap()));
    (attrs, out)
}

fn read_star_information_half(line: String) -> (Vec<f64>, Vec<f64>) {
    let parts: Vec<&str> = line.split(",").collect();
    let mut attrs = Vec::new();
    for i in 0..2 {
        attrs.push(f64::from_str(parts[i]).unwrap());
    }
    let out = Vec::<f64>::from(StarCat::from(u32::from_str(parts[4]).unwrap()));
    (attrs, out)
}

#[inline]
fn read_star_file(half: bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let reader = if half {
        read_star_information_half
    } else {
        read_star_information
    };
    match read_dataset_file(STAR_FILE, &reader, true) {
        Some(v) => v,
        None => panic!("Test file '{}' not found", STAR_FILE)
    }
}

#[test]
fn test_star_classification() {
    let (mut data, mut expect) = read_star_file(false);
    let mut nn = NeuralNet::new_regressor(4, 6);
    // println!("before: {:#?}", nn);
    nn.train(&data, &expect);
    // println!("after: {:#?}", nn);

    assert!(expect_success_rate(&nn, &data, &expect, 0.5, &|l, r| StarCat::from(l) == StarCat::from(r)));

    // let strat_result = cross_validation_testing(&mut nn, &mut data, &mut expect, 0.9, &|l, r| StarCat::from(l) == StarCat::from(r)).unwrap();
    // println!("Achieved {} strat result", strat_result);
    // assert!(strat_result > 0.5);
}

#[test]
fn test_sum_classification() {
    let (mut data, mut expected) = sum_data();
    let mut nn = NeuralNet::new_custom_regressor(2, 2, vec!(2));
    nn.train(&data, &expected);
    data.push(vec!(0.5, 0.5));
    expected.push(vec!(1.0, 0.0));
    data.push(vec!(0.75, 0.25));
    expected.push(vec!(1.0, 0.0));
    data.push(vec!(0.25, 0.75));
    expected.push(vec!(1.0, 0.0));
    data.push(vec!(0.33, 1.66));
    expected.push(vec!(1.0, 1.0));

    for i in 0..data.len() {
        let sum = sum_output(&nn.classify(&data[i]));
        assert_eq!(sum[0], expected[i][0]);
        assert_eq!(sum[1], expected[i][1]);
    }
}

fn sum_output(result: &Vec<f64>) -> Vec<f64> {
    vec!(result[0].round(), result[1].round())
}

fn sum_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let inputs = vec!(
        vec!(0f64, 0f64),
        vec!(0f64, 1f64),
        vec!(1f64, 0f64),
        vec!(1f64, 1f64)
    );
    let outputs = vec!(
        vec!(0f64, 0f64),
        vec!(1f64, 0f64),
        vec!(1f64, 0f64),
        vec!(1f64, 1f64)
    );
    (inputs, outputs)
}

// just test the to/from methods for the starcat enum
#[test]
fn test_from_vec() {
    assert_eq!(StarCat::from(vec!(1f64, 0f64, 0f64, 0f64, 0f64, 0f64)), BrownDwarf);
    assert_eq!(StarCat::from(vec!(0f64, 1f64, 0f64, 0f64, 0f64, 0f64)), RedDwarf);
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 1f64, 0f64, 0f64, 0f64)), WhiteDwarf);
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 0f64, 1f64, 0f64, 0f64)), MainSeq);
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 0f64, 0f64, 1f64, 0f64)), SuperGiant);
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 0f64, 0f64, 0f64, 1f64)), HyperGiant);
}

#[test]
fn test_to_vec() {
    assert_eq!(vec!(1f64, 0f64, 0f64, 0f64, 0f64, 0f64), Vec::<f64>::from(BrownDwarf));
    assert_eq!(vec!(0f64, 1f64, 0f64, 0f64, 0f64, 0f64), Vec::<f64>::from(RedDwarf));
    assert_eq!(vec!(0f64, 0f64, 1f64, 0f64, 0f64, 0f64), Vec::<f64>::from(WhiteDwarf));
    assert_eq!(vec!(0f64, 0f64, 0f64, 1f64, 0f64, 0f64), Vec::<f64>::from(MainSeq));
    assert_eq!(vec!(0f64, 0f64, 0f64, 0f64, 1f64, 0f64), Vec::<f64>::from(SuperGiant));
    assert_eq!(vec!(0f64, 0f64, 0f64, 0f64, 0f64, 1f64), Vec::<f64>::from(HyperGiant));
}