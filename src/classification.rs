use self::Classification::{Positive, Negative, Unclassifiable};

pub enum Classification {
    Positive(f32),
    Negative(f32),
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

    pub fn certainty(&self) -> f32 {
        match self {
            Positive(v) | Negative(v) => *v,
            Unclassifiable(_) => -1f32
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