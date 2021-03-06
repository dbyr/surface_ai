use std::str::FromStr;

use crate::common::read_dataset_file;
use crate::neural_net::NeuralNet;
use crate::classifier::Classifier;
use crate::test_methods::cross_validation_testing;

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
            1 => RedDwarf,
            0 => BrownDwarf,
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
            RedDwarf => 1,
            BrownDwarf => 0,
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
            RedDwarf => 1,
            BrownDwarf => 0,
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
        let mut factor = 1f64;
        let mut cat = 0f64;
        for d in v.iter() {
            cat += factor * d.round();
            factor *= 2f64;
        }
        StarCat::from(cat as u32)
    }
}
impl From<&Vec<f64>> for StarCat {
    fn from(v: &Vec<f64>) -> Self {
        let mut factor = 1f64;
        let mut cat = 0f64;
        for d in v.iter() {
            cat += factor * d.round();
            factor *= 2f64;
        }
        StarCat::from(cat as u32)
    }
}

impl From<StarCat> for Vec<f64> {
    fn from(s: StarCat) -> Self {
        let mut r = vec!(0f64, 0f64, 0f64);
        let mut factor;
        let mut cat_val = u32::from(s);
        for i in (0..r.len()).rev() {
            factor = 2u32.pow(i as u32);
            if cat_val >= factor {
                r[i] = 1f64;
                cat_val -= factor;
            }
        }
        r
    }
}
impl From<&StarCat> for Vec<f64> {
    fn from(s: &StarCat) -> Self {
        let mut r = vec!(0f64, 0f64, 0f64);
        let mut factor;
        let mut cat_val = u32::from(s);
        for i in (0..r.len()).rev() {
            factor = 2u32.pow(i as u32);
            if cat_val >= factor {
                r[i] = 1f64;
                cat_val -= factor;
            }
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

#[inline]
fn read_star_file() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    read_dataset_file(STAR_FILE, &read_star_information, true)
}

#[test]
fn test_star_classification() {
    let (mut data, mut expect) = read_star_file();
    let mut nn = NeuralNet::new(4, 3);
    println!("before: {:#?}", nn);
    nn.train(&data, &expect);
    println!("after: {:#?}", nn);

    let strat_result = cross_validation_testing(&mut nn, &mut data, &mut expect, 0.9, &|l, r| StarCat::from(l) == StarCat::from(r)).unwrap();
    println!("Achieved {} strat result", strat_result);
    assert!(strat_result > 0.95);
}

// just test the to/from methods for the starcat enum
#[test]
fn test_from_vec() {
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 0f64)), BrownDwarf);
    assert_eq!(StarCat::from(vec!(1f64, 0f64, 0f64)), RedDwarf);
    assert_eq!(StarCat::from(vec!(0f64, 1f64, 0f64)), WhiteDwarf);
    assert_eq!(StarCat::from(vec!(1f64, 1f64, 0f64)), MainSeq);
    assert_eq!(StarCat::from(vec!(0f64, 0f64, 1f64)), SuperGiant);
    assert_eq!(StarCat::from(vec!(1f64, 0f64, 1f64)), HyperGiant);
}

#[test]
fn test_to_vec() {
    assert_eq!(vec!(0f64, 0f64, 0f64), Vec::<f64>::from(BrownDwarf));
    assert_eq!(vec!(1f64, 0f64, 0f64), Vec::<f64>::from(RedDwarf));
    assert_eq!(vec!(0f64, 1f64, 0f64), Vec::<f64>::from(WhiteDwarf));
    assert_eq!(vec!(1f64, 1f64, 0f64), Vec::<f64>::from(MainSeq));
    assert_eq!(vec!(0f64, 0f64, 1f64), Vec::<f64>::from(SuperGiant));
    assert_eq!(vec!(1f64, 0f64, 1f64), Vec::<f64>::from(HyperGiant));
}