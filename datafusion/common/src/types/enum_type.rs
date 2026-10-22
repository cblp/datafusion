// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! PostgreSQL-style enumerated types.

use crate::Result;
use crate::datatype::DataTypeExt;
use crate::error::{_internal_err, _plan_datafusion_err, _plan_err};
use crate::types::extension::DFExtensionType;
use arrow::array::{Array, AsArray, Int16Array, StringArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use arrow::util::display::{ArrayFormatter, DisplayIndex, FormatOptions, FormatResult};
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

/// An enumerated type: a static, ordered set of text labels, like
/// `CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')` in PostgreSQL.
///
/// Values are stored as [`DataType::Int16`] codes, where a code is the position of the label
/// in the declaration. Comparing and sorting codes therefore follows the declaration order of
/// the labels, as in PostgreSQL.
///
/// The type is identified by the `datafusion.enum` extension type name. Its metadata is a JSON
/// object with the type name and the labels, for example
/// `{"name":"mood","labels":["sad","ok","happy"]}`. As the metadata is part of the Arrow schema,
/// file formats that store the Arrow schema (such as Parquet and Arrow IPC) keep the type
/// definition next to the data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DFEnum {
    name: String,
    labels: Vec<String>,
}

impl DFEnum {
    /// The extension type name of enumerated types.
    pub const NAME: &'static str = "datafusion.enum";

    /// The storage type of the codes.
    pub const STORAGE_TYPE: DataType = DataType::Int16;

    /// Creates a new enumerated type named `name` with the `labels` in declaration order.
    pub fn try_new(name: impl Into<String>, labels: Vec<String>) -> Result<Self> {
        let name = name.into();
        if labels.len() > i16::MAX as usize + 1 {
            return _plan_err!(
                "Enum type {name} has {} labels, at most {} are supported",
                labels.len(),
                i16::MAX as usize + 1
            );
        }
        let mut seen = std::collections::HashSet::with_capacity(labels.len());
        for label in &labels {
            if !seen.insert(label.as_str()) {
                return _plan_err!("Enum label \"{label}\" used more than once");
            }
        }
        Ok(Self { name, labels })
    }

    /// Creates an enumerated type from the storage type and the serialized metadata of a field.
    pub fn try_from_metadata(
        data_type: &DataType,
        metadata: Option<&str>,
    ) -> Result<Self> {
        if data_type != &Self::STORAGE_TYPE {
            return _plan_err!(
                "Enum type requires storage type {}, got {data_type}",
                Self::STORAGE_TYPE
            );
        }
        let Some(metadata) = metadata else {
            return _plan_err!("Enum type requires metadata");
        };
        let invalid = || _plan_datafusion_err!("Invalid enum type metadata: {metadata}");
        let value: serde_json::Value =
            serde_json::from_str(metadata).map_err(|_| invalid())?;
        let name = value
            .get("name")
            .and_then(|name| name.as_str())
            .ok_or_else(invalid)?;
        let labels = value
            .get("labels")
            .and_then(|labels| labels.as_array())
            .ok_or_else(invalid)?
            .iter()
            .map(|label| label.as_str().map(str::to_owned).ok_or_else(invalid))
            .collect::<Result<Vec<_>>>()?;
        Self::try_new(name, labels)
    }

    /// Returns the enumerated type of `field`, or `None` if the field is not an enum.
    pub fn try_from_field(field: &Field) -> Result<Option<Self>> {
        if field.extension_type_name() != Some(Self::NAME) {
            return Ok(None);
        }
        Self::try_from_metadata(field.data_type(), field.extension_type_metadata())
            .map(Some)
    }

    /// The name of the type.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The labels in declaration order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the metadata to attach to a field of this type.
    pub fn field_metadata(&self) -> HashMap<String, String> {
        HashMap::from([
            (EXTENSION_TYPE_NAME_KEY.to_owned(), Self::NAME.to_owned()),
            (
                EXTENSION_TYPE_METADATA_KEY.to_owned(),
                self.serialize_metadata_json(),
            ),
        ])
    }

    /// Returns a nullable, unnamed field of this type.
    pub fn to_field_ref(&self) -> FieldRef {
        Arc::new(
            Self::STORAGE_TYPE
                .into_nullable_field()
                .with_metadata(self.field_metadata()),
        )
    }

    /// Returns the code of `label`.
    pub fn code(&self, label: &str) -> Result<i16> {
        self.labels
            .iter()
            .position(|l| l == label)
            .map(|code| code as i16)
            .ok_or_else(|| {
                _plan_datafusion_err!(
                    "invalid input value for enum {}: \"{label}\"",
                    self.name
                )
            })
    }

    /// Returns the label of `code`.
    pub fn label(&self, code: i16) -> Result<&str> {
        usize::try_from(code)
            .ok()
            .and_then(|code| self.labels.get(code))
            .map(String::as_str)
            .ok_or_else(|| {
                _plan_datafusion_err!("invalid code for enum {}: {code}", self.name)
            })
    }

    /// Converts an array of labels (`Utf8`, `LargeUtf8` or `Utf8View`) to an array of codes.
    pub fn encode(&self, array: &dyn Array) -> Result<Int16Array> {
        let codes: HashMap<&str, i16> = self
            .labels
            .iter()
            .enumerate()
            .map(|(code, label)| (label.as_str(), code as i16))
            .collect();
        let encode = |label: Option<&str>| {
            label
                .map(|label| {
                    codes.get(label).copied().ok_or_else(|| {
                        _plan_datafusion_err!(
                            "invalid input value for enum {}: \"{label}\"",
                            self.name
                        )
                    })
                })
                .transpose()
        };
        match array.data_type() {
            DataType::Utf8 => array.as_string::<i32>().iter().map(encode).collect(),
            DataType::LargeUtf8 => array.as_string::<i64>().iter().map(encode).collect(),
            DataType::Utf8View => array.as_string_view().iter().map(encode).collect(),
            other => _internal_err!("Cannot convert {other} to enum {}", self.name),
        }
    }

    /// Converts an array of codes to an array of labels.
    pub fn decode(&self, array: &Int16Array) -> Result<StringArray> {
        array
            .iter()
            .map(|code| code.map(|code| self.label(code)).transpose())
            .collect()
    }

    fn serialize_metadata_json(&self) -> String {
        serde_json::json!({ "name": self.name, "labels": self.labels }).to_string()
    }
}

impl DFExtensionType for DFEnum {
    fn storage_type(&self) -> DataType {
        Self::STORAGE_TYPE
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(self.serialize_metadata_json())
    }

    fn create_array_formatter<'fmt>(
        &self,
        array: &'fmt dyn Array,
        options: &FormatOptions<'fmt>,
    ) -> Result<Option<ArrayFormatter<'fmt>>> {
        let Some(codes) = array.as_primitive_opt() else {
            return _internal_err!("Wrong array type for enum {}", self.name);
        };
        let display_index = EnumValueDisplayIndex {
            codes,
            labels: self.labels.clone(),
            null_str: options.null(),
        };
        Ok(Some(ArrayFormatter::new(
            Box::new(display_index),
            options.safe(),
        )))
    }
}

/// Pretty printer for enum codes.
struct EnumValueDisplayIndex<'a> {
    codes: &'a Int16Array,
    labels: Vec<String>,
    null_str: &'a str,
}

impl DisplayIndex for EnumValueDisplayIndex<'_> {
    fn write(&self, idx: usize, f: &mut dyn Write) -> FormatResult {
        if self.codes.is_null(idx) {
            write!(f, "{}", self.null_str)?;
            return Ok(());
        }
        let code = self.codes.value(idx);
        match usize::try_from(code).ok().and_then(|c| self.labels.get(c)) {
            Some(label) => write!(f, "{label}")?,
            None => write!(f, "<invalid enum code {code}>")?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mood() -> DFEnum {
        DFEnum::try_new("mood", vec!["sad".into(), "ok".into(), "happy".into()]).unwrap()
    }

    #[test]
    fn metadata_roundtrip() -> Result<()> {
        let mood = mood();
        let field = mood.to_field_ref();
        assert_eq!(DFEnum::try_from_field(&field)?, Some(mood));
        assert_eq!(
            DFEnum::try_from_field(&Field::new("", DataType::Int16, true))?,
            None
        );
        Ok(())
    }

    #[test]
    fn rejects_duplicate_labels() {
        let err = DFEnum::try_new("e", vec!["a".into(), "a".into()]).unwrap_err();
        assert!(err.to_string().contains("used more than once"), "{err}");
    }

    #[test]
    fn encode_decode() -> Result<()> {
        let mood = mood();
        let labels = StringArray::from(vec![Some("happy"), None, Some("sad")]);
        let codes = mood.encode(&labels)?;
        assert_eq!(codes, Int16Array::from(vec![Some(2), None, Some(0)]));
        assert_eq!(mood.decode(&codes)?, labels);

        let err = mood.encode(&StringArray::from(vec!["angry"])).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid input value for enum mood: \"angry\""),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn pretty_print() -> Result<()> {
        let codes = Int16Array::from(vec![Some(1), None]);
        let formatter = mood()
            .create_array_formatter(&codes, &FormatOptions::default().with_null("NULL"))?
            .unwrap();
        assert_eq!(formatter.value(0).to_string(), "ok");
        assert_eq!(formatter.value(1).to_string(), "NULL");
        Ok(())
    }
}
