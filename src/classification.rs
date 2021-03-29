use self::Classification::{Positive, Negative, Unclassifiable, Probs};
use crate::common::reasonably_equal;

#[derive(Debug, Clone)]
pub enum Classification {
    Positive(f64),
    Negative(f64),
    Unclassifiable(String),
    Probs(Vec<f64>)
}

impl Classification {

    // determines the reason for failed classification
    pub fn reason(self) -> String {
        match self {
            Unclassifiable(r) => r,
            _ => "Classification successful".to_string()
        }
    }
    pub fn view_reason(&self) -> String {
        match self {
            Unclassifiable(r) => r.clone(),
            _ => "Classification successful".to_string()
        }
    }

    pub fn class_match(&self, other: &Classification) -> bool {
        match self {
            Positive(_) => other.positive(),
            Negative(_) => other.negative(),
            Probs(my_probs) => {
                let other_probs = match other {
                    Probs(p) => p,
                    _ => return false
                };
                if my_probs.len() != other_probs.len() {
                    return false;
                }

                let mut my_largest = 0f64;
                let mut other_largest = 0f64;
                for (m, o) in my_probs.iter().zip(other_probs.iter()) {
                    if *m > my_largest && *o > other_largest {
                        my_largest = *m;
                        other_largest = *o;
                    } else if *m < my_largest && *o < other_largest {
                        continue;
                    } else {
                        return false;
                    }
                }
                true
            }
            _ => false
        }
    }
    pub fn exact_match(&self, other: &Classification) -> bool {
        match self {
            Positive(_) | Negative(_) => reasonably_equal(&self.certainty(), &other.certainty()),
            _ => false
        }
    }

    pub fn error(&self, other: &Classification) -> f64 {
        if let Unclassifiable(_) = self {
            0f64
        } else if let Unclassifiable(_) = other {
            0f64
        } else {
            self.certainty() - other.certainty()
        }
    }

    pub fn loss(&self, expected: &Classification) -> f64 {
        match self {
            Positive(_) | Negative(_) => self.error(expected).powi(2),
            Probs(vs) => {
                if let Probs(os) = expected {
                    if vs.len() != os.len() {
                        return 0f64;
                    }
                    for (a, e) in vs.iter().zip(os.iter()) {
                        if *e == 1f64 {
                            return -a.log10();
                        }
                    }
                }
                0f64
            }
            _ => 0f64
        }
    }

    pub fn certainty(&self) -> f64 {
        match self {
            Positive(v) | Negative(v) => *v,
            Probs(p) => {
                let mut largest = 0f64;
                for val in p.iter() {
                    if *val > largest {
                        largest = *val;
                    }
                }
                largest
            }
            Unclassifiable(_) => -1f64
        }
    }

    pub fn probs(&self) -> bool {
        match self {
            Probs(_) => true,
            _ => false
        }
    }

    pub fn positive(&self) -> bool {
        match self {
            Positive(_) => true,
            _ => false
        }
    }
    pub fn negative(&self) -> bool {
        match self {
            Negative(_) => true,
            _ => false
        }
    }

    pub fn thoughts(self) -> Vec<f64> {
        match self {
            Probs(v) => v,
            _ => panic!("Cannot get thoughts on non-probs classification")
        }
    }
}

pub fn average_certainty(classes: &mut dyn Iterator<Item=Classification>) -> f64 {
    let mut sum = 0f64;
    let mut count = 0;
    while let Some(class) = classes.next() {
        match class {
            Positive(c) => sum += c,
            Negative(c) => sum += 1f64 - c,
            _ => continue
        }
        count += 1;
    }
    ((sum / (count as f64)) * 2f64) - 1f64 // re-scale the number
}

impl PartialEq for Classification {
    fn eq(&self, other: &Self) -> bool {
        self.class_match(other)
    }
}

impl Eq for Classification {}