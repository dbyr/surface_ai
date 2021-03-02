use self::Classification::{Positive, Negative, Unclassifiable};

#[derive(Debug)]
pub enum Classification {
    Positive(f64),
    Negative(f64),
    Unclassifiable(String)
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
            _ => false
        }
    }

    pub fn certainty(&self) -> f64 {
        match self {
            Positive(v) | Negative(v) => *v,
            Unclassifiable(_) => -1f64
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
    sum / (count as f64)
}

impl PartialEq for Classification {
    fn eq(&self, other: &Self) -> bool {
        self.class_match(other)
    }
}

impl Eq for Classification {}