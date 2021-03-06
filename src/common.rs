#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::io::{BufRead, BufReader};

const EQUAL_THRESHOLD: f64 = 0.0000000001;

#[inline]
pub fn reasonably_equal(left: &f64, right: &f64) -> bool {
    (left - right).abs() < EQUAL_THRESHOLD
}

#[cfg(test)]
pub fn read_dataset_file<I, O>(
    filename: &str,
    read_line: &dyn Fn(String) -> (I, O),
    skip_first: bool
) -> (Vec<I>, Vec<O>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut lines = reader.lines();
    if skip_first {lines.next();}
    for l in lines {
        let line = l.unwrap();
        let (i, o) = read_line(line);
        inputs.push(i);
        outputs.push(o);
    }
    (inputs, outputs)
}