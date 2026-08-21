//! Reversible serde representation for floating-point metric values.
//!
//! JSON has no native representation for IEEE-754 infinities or NaN. Serde
//! JSON otherwise serializes those values as `null`, which cannot be
//! deserialized back into `f64` and also conflates a perfect PSNR with missing
//! data. Finite values stay numeric; non-finite values use stable strings.

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy)]
struct ReversibleF64(f64);

impl Serialize for ReversibleF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_value(self.0, serializer)
    }
}

impl<'de> Deserialize<'de> for ReversibleF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ReversibleF64Visitor).map(Self)
    }
}

struct ReversibleF64Visitor;

impl Visitor<'_> for ReversibleF64Visitor {
    type Value = f64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a number or one of Infinity, -Infinity, and NaN")
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
        Ok(value)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(value as f64)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(value as f64)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        match value {
            "Infinity" => Ok(f64::INFINITY),
            "-Infinity" => Ok(f64::NEG_INFINITY),
            "NaN" => Ok(f64::NAN),
            _ => value
                .parse::<f64>()
                .map_err(|_| E::custom(format!("invalid floating-point value `{value}`"))),
        }
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_str(&value)
    }
}

fn serialize_value<S>(value: f64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if value.is_finite() {
        serializer.serialize_f64(value)
    } else if value.is_nan() {
        serializer.serialize_str("NaN")
    } else if value.is_sign_positive() {
        serializer.serialize_str("Infinity")
    } else {
        serializer.serialize_str("-Infinity")
    }
}

pub(crate) mod value {
    use super::*;

    pub(crate) fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_value(*value, serializer)
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        ReversibleF64::deserialize(deserializer).map(|value| value.0)
    }
}

pub(crate) mod option {
    use super::*;

    pub(crate) fn serialize<S>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(value) => serializer.serialize_some(&ReversibleF64(*value)),
            None => serializer.serialize_none(),
        }
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<ReversibleF64>::deserialize(deserializer).map(|value| value.map(|value| value.0))
    }
}

pub(crate) mod map {
    use super::*;
    use serde::ser::SerializeMap;

    pub(crate) fn serialize<S>(
        values: &BTreeMap<String, f64>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(values.len()))?;
        for (key, value) in values {
            map.serialize_entry(key, &ReversibleF64(*value))?;
        }
        map.end()
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<BTreeMap<String, f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        BTreeMap::<String, ReversibleF64>::deserialize(deserializer).map(|values| {
            values
                .into_iter()
                .map(|(key, value)| (key, value.0))
                .collect()
        })
    }
}

pub(crate) mod vec {
    use super::*;
    use serde::ser::SerializeSeq;

    pub(crate) fn serialize<S>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(values.len()))?;
        for value in values {
            sequence.serialize_element(&ReversibleF64(*value))?;
        }
        sequence.end()
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<ReversibleF64>::deserialize(deserializer)
            .map(|values| values.into_iter().map(|value| value.0).collect())
    }
}

macro_rules! array_module {
    ($module:ident, $length:literal) => {
        pub(crate) mod $module {
            use super::*;
            use serde::ser::SerializeSeq;

            pub(crate) fn serialize<S>(
                values: &[f64; $length],
                serializer: S,
            ) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut sequence = serializer.serialize_seq(Some($length))?;
                for value in values {
                    sequence.serialize_element(&ReversibleF64(*value))?;
                }
                sequence.end()
            }

            pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<[f64; $length], D::Error>
            where
                D: Deserializer<'de>,
            {
                <[ReversibleF64; $length]>::deserialize(deserializer)
                    .map(|values| values.map(|value| value.0))
            }
        }
    };
}

array_module!(array3, 3);

/// Serializes metric-detail values without losing IEEE-754 non-finite values.
pub fn map_to_json(values: &BTreeMap<String, f64>) -> Result<String, serde_json::Error> {
    serde_json::to_string(&SerializableMap(values))
}

struct SerializableMap<'a>(&'a BTreeMap<String, f64>);

impl Serialize for SerializableMap<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        map::serialize(self.0, serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Direction, MetricOutput};

    #[test]
    fn json_non_finite_values_roundtrip() {
        for (value, encoded) in [
            (f64::INFINITY, "\"Infinity\""),
            (f64::NEG_INFINITY, "\"-Infinity\""),
            (f64::NAN, "\"NaN\""),
            (1.25, "1.25"),
        ] {
            let json = serde_json::to_string(&ReversibleF64(value)).unwrap();
            assert_eq!(json, encoded);
            let decoded: ReversibleF64 = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.0.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn metric_output_preserves_perfect_score_and_detail() {
        let metric =
            MetricOutput::new("psnr:color", f64::INFINITY, "dB", Direction::HigherIsBetter)
                .with_detail("red_psnr", f64::INFINITY);

        let json = serde_json::to_string(&metric).unwrap();
        assert!(json.contains("\"score\":\"Infinity\""));
        assert!(json.contains("\"red_psnr\":\"Infinity\""));
        let decoded: MetricOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.score, f64::INFINITY);
        assert_eq!(decoded.details["red_psnr"], f64::INFINITY);
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Containers {
        #[serde(with = "vec")]
        values: Vec<f64>,
        #[serde(with = "array3")]
        triple: [f64; 3],
    }

    #[test]
    fn containers_roundtrip_non_finite_values() {
        let value = Containers {
            values: vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            triple: [f64::NAN, 1.0, f64::INFINITY],
        };

        let json = serde_json::to_string(&value).unwrap();
        assert!(!json.contains("null"));
        let decoded: Containers = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.values[0], f64::INFINITY);
        assert_eq!(decoded.values[1], f64::NEG_INFINITY);
        assert!(decoded.values[2].is_nan());
        assert!(decoded.triple[0].is_nan());
        assert_eq!(decoded.triple[2], f64::INFINITY);
    }
}
