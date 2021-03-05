const EQUAL_THRESHOLD: f64 = 0.0000000001;

#[inline]
pub fn reasonably_equal(left: &f64, right: &f64) -> bool {
    (left - right).abs() < EQUAL_THRESHOLD
}