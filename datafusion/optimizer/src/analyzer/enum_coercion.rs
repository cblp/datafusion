// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Type coercion for enumerated types ([`DFEnum`]).
//!
//! Enum values are `Int16` codes, so most operators work on them unchanged. This module
//! resolves the operations that need the labels:
//!
//! - comparing an enum with text (`mood = 'happy'`, `mood IN (...)`, `BETWEEN`) converts the
//!   text to codes, so the comparison follows the declaration order of the labels,
//! - casting text to an enum (`'happy'::mood`, `INSERT` into an enum column) converts labels to
//!   codes,
//! - casting an enum to text converts codes to labels.
//!
//! Other operations that would treat the codes as numbers, such as arithmetic or comparing two
//! different enum types, are rejected.

use std::sync::Arc;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Field, FieldRef, Int16Type};
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::tree_node::Transformed;
use datafusion_common::types::DFEnum;
use datafusion_common::{DFSchema, Result, ScalarValue, internal_err, plan_err};
use datafusion_expr::expr::{Between, BinaryExpr, InList, ScalarFunction};
use datafusion_expr::{
    Cast, ColumnarValue, Expr, ExprSchemable, Operator, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TryCast, Volatility,
};

/// Rewrites the enum-related parts of `expr` (not of its children).
pub(crate) fn coerce_enum_expr(
    expr: Expr,
    schema: &DFSchema,
) -> Result<Transformed<Expr>> {
    if involves_enum(&expr, schema)? {
        rewrite_enum_expr(expr, schema).map(Transformed::yes)
    } else {
        Ok(Transformed::no(expr))
    }
}

/// Returns whether [`rewrite_enum_expr`] needs to rewrite `expr`.
fn involves_enum(expr: &Expr, schema: &DFSchema) -> Result<bool> {
    let is_enum = |expr: &Expr| enum_of(expr, schema).map(|e| e.is_some());
    match expr {
        Expr::Cast(Cast { expr, field }) | Expr::TryCast(TryCast { expr, field }) => {
            Ok(DFEnum::try_from_field(field)?.is_some() || is_enum(expr)?)
        }
        // Logical operators cannot take enums, and skipping them avoids computing the types
        // of deep AND / OR trees
        Expr::BinaryExpr(BinaryExpr {
            op: Operator::And | Operator::Or,
            ..
        }) => Ok(false),
        Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
            Ok(is_enum(left)? || is_enum(right)?)
        }
        Expr::Between(Between { expr, .. }) | Expr::InList(InList { expr, .. }) => {
            is_enum(expr)
        }
        _ => Ok(false),
    }
}

fn rewrite_enum_expr(expr: Expr, schema: &DFSchema) -> Result<Expr> {
    match expr {
        Expr::Cast(Cast { expr, field }) => match cast_enum(*expr, &field, schema)? {
            Ok(expr) => Ok(expr),
            Err(expr) => Ok(Expr::Cast(Cast::new_from_field(Box::new(expr), field))),
        },
        Expr::TryCast(TryCast { expr, field }) => {
            match cast_enum(*expr, &field, schema)? {
                Ok(expr) => Ok(expr),
                Err(expr) => Ok(Expr::TryCast(TryCast::new_from_field(
                    Box::new(expr),
                    field,
                ))),
            }
        }
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let (left, right) = match (enum_of(&left, schema)?, enum_of(&right, schema)?)
            {
                (None, None) => (left, right),
                (Some(enum_type), None) => {
                    check_comparison(op, &enum_type)?;
                    let right = to_enum(*right, &enum_type, schema)?;
                    (left, Box::new(right))
                }
                (None, Some(enum_type)) => {
                    check_comparison(op, &enum_type)?;
                    let left = to_enum(*left, &enum_type, schema)?;
                    (Box::new(left), right)
                }
                (Some(left_type), Some(right_type)) => {
                    check_comparison(op, &left_type)?;
                    if left_type != right_type {
                        return plan_err!(
                            "Cannot compare enum types {} and {}",
                            left_type.name(),
                            right_type.name()
                        );
                    }
                    (left, right)
                }
            };
            Ok(Expr::BinaryExpr(BinaryExpr::new(left, op, right)))
        }
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) => {
            let Some(enum_type) = enum_of(&expr, schema)? else {
                return Ok(Expr::Between(Between::new(expr, negated, low, high)));
            };
            let low = to_enum(*low, &enum_type, schema)?;
            let high = to_enum(*high, &enum_type, schema)?;
            Ok(Expr::Between(Between::new(
                expr,
                negated,
                Box::new(low),
                Box::new(high),
            )))
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) => {
            let Some(enum_type) = enum_of(&expr, schema)? else {
                return Ok(Expr::InList(InList::new(expr, list, negated)));
            };
            let list = list
                .into_iter()
                .map(|e| to_enum(e, &enum_type, schema))
                .collect::<Result<_>>()?;
            Ok(Expr::InList(InList::new(expr, list, negated)))
        }
        expr => Ok(expr),
    }
}

/// Returns the enum type of `expr`, if any.
fn enum_of(expr: &Expr, schema: &DFSchema) -> Result<Option<DFEnum>> {
    let (_, field) = expr.to_field(schema)?;
    DFEnum::try_from_field(&field)
}

fn check_comparison(op: Operator, enum_type: &DFEnum) -> Result<()> {
    match op {
        Operator::Eq
        | Operator::NotEq
        | Operator::Lt
        | Operator::LtEq
        | Operator::Gt
        | Operator::GtEq
        | Operator::IsDistinctFrom
        | Operator::IsNotDistinctFrom => Ok(()),
        _ => plan_err!(
            "Operator {op} is not supported for enum type {}",
            enum_type.name()
        ),
    }
}

/// Converts `expr` to a value of `enum_type`. Text is converted to codes.
fn to_enum(expr: Expr, enum_type: &DFEnum, schema: &DFSchema) -> Result<Expr> {
    if let Some(source_type) = enum_of(&expr, schema)? {
        if &source_type != enum_type {
            return plan_err!(
                "Cannot convert enum type {} to {}",
                source_type.name(),
                enum_type.name()
            );
        }
        return Ok(expr);
    }
    let data_type = expr.get_type(schema)?;
    match expr {
        Expr::Literal(value, _) if value.is_null() => Ok(enum_literal(enum_type, None)),
        Expr::Literal(value, _) if is_string(&data_type) => {
            let label = value.try_as_str().flatten().unwrap_or_default();
            Ok(enum_literal(enum_type, Some(enum_type.code(label)?)))
        }
        expr if data_type == DataType::Null => Ok(Expr::Cast(Cast::new_from_field(
            Box::new(expr),
            enum_type.to_field_ref(),
        ))),
        expr if is_string(&data_type) => {
            Ok(Expr::ScalarFunction(ScalarFunction::new_udf(
                Arc::new(ScalarUDF::new_from_impl(TextToEnum::new(enum_type.clone()))),
                vec![expr],
            )))
        }
        _ => plan_err!(
            "Cannot convert {data_type} to enum type {}",
            enum_type.name()
        ),
    }
}

/// Rewrites a cast of `expr` to `target` if the cast involves an enum. Returns `Err(expr)` if
/// it does not.
fn cast_enum(
    expr: Expr,
    target: &FieldRef,
    schema: &DFSchema,
) -> Result<std::result::Result<Expr, Expr>> {
    let source_type = enum_of(&expr, schema)?;
    let target_type = DFEnum::try_from_field(target)?;
    match (source_type, target_type) {
        (None, None) => Ok(Err(expr)),
        (_, Some(target_type)) => to_enum(expr, &target_type, schema).map(Ok),
        (Some(source_type), None) => {
            let target_data_type = target.data_type();
            if !is_string(target_data_type) {
                return plan_err!(
                    "Cannot cast enum type {} to {target_data_type}",
                    source_type.name()
                );
            }
            let text = match expr {
                Expr::Literal(ScalarValue::Int16(code), _) => Expr::Literal(
                    ScalarValue::Utf8(
                        code.map(|code| source_type.label(code).map(str::to_owned))
                            .transpose()?,
                    ),
                    None,
                ),
                expr => Expr::ScalarFunction(ScalarFunction::new_udf(
                    Arc::new(ScalarUDF::new_from_impl(EnumToText::new(source_type))),
                    vec![expr],
                )),
            };
            if target_data_type == &DataType::Utf8 {
                Ok(Ok(text))
            } else {
                Ok(Ok(Expr::Cast(Cast::new(
                    Box::new(text),
                    target_data_type.clone(),
                ))))
            }
        }
    }
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn enum_literal(enum_type: &DFEnum, code: Option<i16>) -> Expr {
    Expr::Literal(
        ScalarValue::Int16(code),
        Some(FieldMetadata::from(enum_type.field_metadata())),
    )
}

/// Converts labels to the codes of an enum type.
#[derive(Debug, PartialEq, Eq, Hash)]
struct TextToEnum {
    enum_type: DFEnum,
    signature: Signature,
}

impl TextToEnum {
    fn new(enum_type: DFEnum) -> Self {
        Self {
            enum_type,
            signature: Signature::string(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for TextToEnum {
    fn name(&self) -> &str {
        "text_to_enum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DFEnum::STORAGE_TYPE)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
        Ok(Arc::new(
            Field::new(self.name(), DFEnum::STORAGE_TYPE, nullable)
                .with_metadata(self.enum_type.field_metadata()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = args.args.as_slice() else {
            return internal_err!("text_to_enum expects one argument");
        };
        let array = arg.to_array(args.number_rows)?;
        let codes = self.enum_type.encode(array.as_ref())?;
        to_columnar_value(arg, Arc::new(codes))
    }
}

/// Converts the codes of an enum type to labels.
#[derive(Debug, PartialEq, Eq, Hash)]
struct EnumToText {
    enum_type: DFEnum,
    signature: Signature,
}

impl EnumToText {
    fn new(enum_type: DFEnum) -> Self {
        Self {
            enum_type,
            signature: Signature::exact(
                vec![DFEnum::STORAGE_TYPE],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for EnumToText {
    fn name(&self) -> &str {
        "enum_to_text"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = args.args.as_slice() else {
            return internal_err!("enum_to_text expects one argument");
        };
        let array = arg.to_array(args.number_rows)?;
        let Some(codes) = array.as_primitive_opt::<Int16Type>() else {
            return internal_err!("enum_to_text expects Int16 codes");
        };
        let labels = self.enum_type.decode(codes)?;
        to_columnar_value(arg, Arc::new(labels))
    }
}

/// Returns `result` as a scalar if the argument `arg` is a scalar.
fn to_columnar_value(
    arg: &ColumnarValue,
    result: Arc<dyn Array>,
) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Scalar(_) => Ok(ColumnarValue::Scalar(
            ScalarValue::try_from_array(&result, 0)?,
        )),
        ColumnarValue::Array(_) => Ok(ColumnarValue::Array(result)),
    }
}
