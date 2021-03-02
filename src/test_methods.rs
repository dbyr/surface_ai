use crate::classifier::Classifier;

// pub fn stratified_testing<(
//     classifer: dyn Classifier,

// ) {

// }

// return true if at least the provided
// success rate was achieved
// success_rate is a number between
// 0 and 1 representing a percentage
pub fn expect_success_rate<T, C: Eq>(
    classifier: Box<&dyn Classifier<T, C>>,
    data: &Vec<T>,
    expect: &Vec<C>,
    success_rate: f64
) -> bool {
    if data.len() != expect.len() { return false; }

    let mut correct = 0;
    for (datum, expected) in data.iter().zip(expect.iter()) {
        if classifier.classify(datum) == *expected {
            correct += 1;
        }
    }
    let actual_rate = correct as f64 / (data.len() as f64);
    println!("achieved {}, expected {}", actual_rate, success_rate);
    actual_rate >= success_rate
}