use std::str::FromStr;
use crate::common::read_dataset_file;
use crate::test_methods::shuffle_class_data;

use super::NeuralNet;
use crate::classifier::Classifier;

const HOUSE_FILE: &str = "./data/housing.csv";

#[test]
fn test_house_price_prediction() {
    let (mut data, mut expected) = read_dataset_file(HOUSE_FILE, &read_house_line, true).unwrap();
    let mut nn = NeuralNet::new_regressor(9, 1);
    shuffle_class_data(&mut data, &mut expected);
    nn.train(&data[..2000], &expected[..2000]);
    println!("resulting net: {:#?}", nn);

    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[0], expected[0], nn.classify(&data[0]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[11], expected[11], nn.classify(&data[11]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[21], expected[21], nn.classify(&data[21]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[31], expected[31], nn.classify(&data[31]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[41], expected[41], nn.classify(&data[41]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[51], expected[51], nn.classify(&data[51]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[0], expected[0], nn.classify(&data[30]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[11], expected[11], nn.classify(&data[311]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[21], expected[21], nn.classify(&data[321]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[31], expected[31], nn.classify(&data[331]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[41], expected[41], nn.classify(&data[341]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[51], expected[51], nn.classify(&data[351]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[0], expected[0], nn.classify(&data[80]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[11], expected[11], nn.classify(&data[811]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[21], expected[21], nn.classify(&data[821]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[31], expected[31], nn.classify(&data[831]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[41], expected[41], nn.classify(&data[841]));
    println!("input = {:?}, expect = {:?}, actual = {:.2?}", &data[51], expected[51], nn.classify(&data[851]));
    assert!(false);
}

fn read_house_line(line: String) -> Option<(Vec<f64>, Vec<f64>)> {
    let parts: Vec<&str> = line.split(",").collect();
    let mut attrs = Vec::new();
    for i in 0..8 {
        match f64::from_str(parts[i]) {
            Ok(v) => attrs.push(v),
            Err(_) => return None
        }
    }
    attrs.push(match parts[9] {
        "<1H OCEAN" => 0f64,
        "INLAND" => 1f64,
        "ISLAND" => 2f64,
        "NEAR BAY" => 3f64,
        "NEAR OCEAN" => 4f64,
        _ => 5f64
    });
    let out = vec!(f64::from_str(parts[8]).unwrap());
    Some((attrs, out))
}