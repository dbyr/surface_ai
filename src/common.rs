#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::io::{BufRead, BufReader};

const EQUAL_THRESHOLD: f64 = 0.0000000001;

#[inline]
pub fn reasonably_equal(left: &f64, right: &f64) -> bool {
    (left - right).abs() < EQUAL_THRESHOLD
}

#[inline]
pub fn within_difference(left: &f64, right: &f64, diff: &f64) -> bool {
    (left - right).abs() < *diff
}

// read_line should return None if the data
// for any given line is invalid
#[cfg(test)]
pub fn read_dataset_file<I, O>(
    filename: &str,
    read_line: &dyn Fn(String) -> Option<(I, O)>,
    skip_first: bool
) -> Option<(Vec<I>, Vec<O>)> {
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => return None
    };
    let reader = BufReader::new(file);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut lines = reader.lines();
    if skip_first {lines.next();}
    for l in lines {
        let line = l.unwrap();

        // ignore broken data
        match read_line(line) {
            Some((i, o)) => {
                inputs.push(i);
                outputs.push(o);
            },
            None => continue
        }
    }
    Some((inputs, outputs))
}

pub fn get_normalising_factor(data: &[Vec<f64>]) -> Option<Vec<i8>> {
    let mut factors = vec!(0i8; data.get(0)?.len());
    for datum in data.iter() {
        for (f, attr) in datum.iter().enumerate() {
            while (*attr * 10f64.powi(factors[f] as i32)).abs() > 1f64 {
                factors[f] -= 1;
            }
        }
    }
    Some(factors)
}