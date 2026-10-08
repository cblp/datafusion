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

//! Deterministic `sum` of floating point numbers.
//!
//! Floating point addition is not associative, so plain summation depends on
//! how rows are split into partitions and batches. These accumulators instead
//! compute the exact sum and round it once, using [`bitrep::SumF64`]:
//!
//! 1. Every finite `f64` is an integer multiple of 2⁻¹⁰⁷⁴, the smallest
//!    subnormal. It is converted to that integer, stored in a 2176-bit two's
//!    complement number, and added with integer arithmetic. Integer addition
//!    is exact, so the sum does not depend on the order of the values, and
//!    the width leaves room for 2⁶³ values of any magnitude without overflow.
//! 2. Non-finite values are tracked as flags: any NaN or both infinities
//!    give NaN, a single kind of infinity gives that infinity.
//! 3. Partial aggregates are merged by adding their integers.
//! 4. The final integer is rounded to the nearest `f64`, ties to even, as a
//!    single IEEE 754 addition would; sums beyond the `f64` range become ±inf.

use std::collections::HashSet;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float64Array, ListArray,
};
use arrow::datatypes::{DataType, Field, FieldRef, Float64Type};
use bitrep::SumF64;
use datafusion_common::cast::{
    as_fixed_size_binary_array, as_float64_array, as_list_array,
};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{Accumulator, EmitTo, GroupsAccumulator};

/// The state is [`SumF64::to_bytes`].
const STATE_BYTES: i32 = SumF64::BYTES as i32;

/// The state field of non-distinct floating point `sum`.
pub(super) fn state_field(name: String) -> FieldRef {
    Field::new(name, DataType::FixedSizeBinary(STATE_BYTES), true).into()
}

/// Non-null values of the argument that pass the filter, as `(row, value)`.
fn input_values<'a>(
    values: &'a [ArrayRef],
    opt_filter: Option<&'a BooleanArray>,
) -> Result<impl Iterator<Item = (usize, f64)> + 'a> {
    let values = as_float64_array(&values[0])?;
    Ok(values.iter().enumerate().filter_map(move |(row, value)| {
        let selected =
            opt_filter.is_none_or(|filter| filter.is_valid(row) && filter.value(row));
        Some((row, value.filter(|_| selected)?))
    }))
}

/// Non-null values of the argument.
fn non_null_values(values: &[ArrayRef]) -> Result<impl Iterator<Item = f64> + '_> {
    Ok(input_values(values, None)?.map(|(_, value)| value))
}

/// Non-null partial sums in the state, as `(row, sum)`.
fn partial_sums(
    states: &[ArrayRef],
) -> Result<impl Iterator<Item = Result<(usize, SumF64)>> + '_> {
    let states = as_fixed_size_binary_array(&states[0])?;
    Ok(states.iter().enumerate().filter_map(|(row, bytes)| {
        let sum = bytes?.try_into().ok().and_then(SumF64::from_bytes);
        Some(match sum {
            Some(sum) => Ok((row, sum)),
            None => internal_err!("invalid sum state"),
        })
    }))
}

fn encode_states<'a>(sums: impl IntoIterator<Item = &'a SumF64>) -> ArrayRef {
    let bytes: Vec<u8> = sums.into_iter().flat_map(SumF64::to_bytes).collect();
    Arc::new(FixedSizeBinaryArray::new(STATE_BYTES, bytes.into(), None))
}

/// `sum` of no values is NULL.
fn result(sum: &SumF64) -> Option<f64> {
    (sum.count() > 0).then(|| sum.value())
}

#[derive(Debug, Default)]
pub(super) struct FloatSumAccumulator {
    sum: SumF64,
}

impl Accumulator for FloatSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.sum.extend(non_null_values(values)?);
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        for state in partial_sums(states)? {
            self.sum.merge(&state?.1);
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::FixedSizeBinary(
            STATE_BYTES,
            Some(self.sum.to_bytes().to_vec()),
        )])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Float64(result(&self.sum)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// `sum(DISTINCT x)`. Adding up the distinct values in the iteration order of
/// a hash set would vary between runs.
#[derive(Debug, Default)]
pub(super) struct DistinctFloatSumAccumulator {
    /// Bit patterns of the distinct values.
    values: HashSet<u64>,
}

impl Accumulator for DistinctFloatSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.values
            .extend(non_null_values(values)?.map(f64::to_bits));
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        for values in as_list_array(&states[0])?.iter().flatten() {
            self.update_batch(&[values])?;
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let values = self.values.iter().map(|&bits| Some(f64::from_bits(bits)));
        let list = ListArray::from_iter_primitive::<Float64Type, _, _>([Some(values)]);
        Ok(vec![ScalarValue::List(Arc::new(list))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let sum: SumF64 = self.values.iter().copied().map(f64::from_bits).collect();
        Ok(ScalarValue::Float64(result(&sum)))
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.values.capacity() * size_of::<u64>()
    }
}

/// `sum` over sliding window frames, which removes values leaving the frame.
/// Removing a finite value from a [`SumF64`] is exact, but its flags for NaN
/// and infinities cannot be cleared, so non-finite values are counted instead
/// and added to the sum when evaluating.
#[derive(Debug, Default)]
pub(super) struct SlidingFloatSumAccumulator {
    finite_sum: SumF64,
    count: u64,
    nan_count: u64,
    pos_inf_count: u64,
    neg_inf_count: u64,
}

impl SlidingFloatSumAccumulator {
    fn non_finite_count(&mut self, x: f64) -> &mut u64 {
        match x {
            f64::INFINITY => &mut self.pos_inf_count,
            f64::NEG_INFINITY => &mut self.neg_inf_count,
            _ => &mut self.nan_count,
        }
    }
}

impl Accumulator for SlidingFloatSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for x in non_null_values(values)? {
            self.count += 1;
            if x.is_finite() {
                self.finite_sum.add(x);
            } else {
                *self.non_finite_count(x) += 1;
            }
        }
        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for x in non_null_values(values)? {
            self.count -= 1;
            if x.is_finite() {
                self.finite_sum.add(-x);
            } else {
                *self.non_finite_count(x) -= 1;
            }
        }
        Ok(())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }

    fn merge_batch(&mut self, _states: &[ArrayRef]) -> Result<()> {
        internal_err!("sliding sum is evaluated in a single partition")
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        internal_err!("sliding sum is evaluated in a single partition")
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut sum = self.finite_sum.clone();
        for (count, x) in [
            (self.nan_count, f64::NAN),
            (self.pos_inf_count, f64::INFINITY),
            (self.neg_inf_count, f64::NEG_INFINITY),
        ] {
            if count > 0 {
                sum.add(x);
            }
        }
        Ok(ScalarValue::Float64((self.count > 0).then(|| sum.value())))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// One [`SumF64`] per group, indexed by the group index.
#[derive(Debug, Default)]
pub(super) struct FloatSumGroupsAccumulator {
    sums: Vec<SumF64>,
}

impl GroupsAccumulator for FloatSumGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.sums.resize(total_num_groups, SumF64::new());
        for (row, value) in input_values(values, opt_filter)? {
            self.sums[group_indices[row]].add(value);
        }
        Ok(())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        self.sums.resize(total_num_groups, SumF64::new());
        for state in partial_sums(values)? {
            let (row, sum) = state?;
            self.sums[group_indices[row]].merge(&sum);
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let sums = emit_to.take_needed(&mut self.sums);
        Ok(Arc::new(sums.iter().map(result).collect::<Float64Array>()))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let sums = emit_to.take_needed(&mut self.sums);
        Ok(vec![encode_states(&sums)])
    }

    /// Each row becomes the state of a group holding just that row.
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let mut sums = vec![SumF64::new(); values[0].len()];
        for (row, value) in input_values(values, opt_filter)? {
            sums[row].add(value);
        }
        Ok(vec![encode_states(&sums)])
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.sums.capacity() * size_of::<SumF64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_sum(values: &[f64]) -> Result<f64> {
        let mut acc = FloatSumAccumulator::default();
        acc.update_batch(&[Arc::new(Float64Array::from(values.to_vec()))])?;
        match acc.evaluate()? {
            ScalarValue::Float64(Some(sum)) => Ok(sum),
            other => panic!("unexpected sum result: {other:?}"),
        }
    }

    /// Exact `2ⁿ`, including subnormals.
    fn pow2(n: i32) -> f64 {
        if n >= -1022 {
            f64::from_bits(((n + 1023) as u64) << 52)
        } else {
            f64::from_bits(1 << (n + 1074))
        }
    }

    /// Test cases of CPython's `testFsum` in `Lib/test/test_math.py`.
    #[test]
    fn is_correctly_rounded() -> Result<()> {
        let harmonic: Vec<f64> = (1..=1000).map(|n| 1.0 / f64::from(n)).collect();
        let alternating: Vec<f64> = (1..=1000)
            .map(|n| if n % 2 == 0 { 1.0 } else { -1.0 } / f64::from(n))
            .collect();
        let mut wide_range: Vec<f64> = (-1074..972)
            .step_by(2)
            .map(|n| pow2(n) - pow2(n + 50) + pow2(n + 52))
            .collect();
        wide_range.push(-pow2(1022));

        let cases: Vec<(Vec<f64>, f64)> = vec![
            (vec![0.0], 0.0),
            (vec![1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
            (vec![pow2(53), -0.5, -pow2(-54)], pow2(53) - 1.0),
            (vec![pow2(53), 1.0, pow2(-100)], pow2(53) + 2.0),
            (vec![pow2(53) + 10.0, 1.0, pow2(-100)], pow2(53) + 12.0),
            (vec![pow2(53) - 4.0, 0.5, pow2(-54)], pow2(53) - 3.0),
            (harmonic, 7.485470860550345),
            (alternating, -0.6926474305598203),
            (vec![1e16, 1.0, 1e-16], 10000000000000002.0),
            (
                vec![
                    1e16 - 2.0,
                    1.0 - pow2(-53),
                    -(1e16 - 2.0),
                    -(1.0 - pow2(-53)),
                ],
                0.0,
            ),
            (wide_range, 1.3305602063564798e292),
        ];
        for (values, expected) in cases {
            assert_eq!(exact_sum(&values)?, expected, "{values:?}");
        }
        Ok(())
    }

    #[test]
    fn non_finite_values() -> Result<()> {
        assert_eq!(exact_sum(&[1.0, f64::INFINITY, 2.0])?, f64::INFINITY);
        assert!(exact_sum(&[f64::INFINITY, f64::NEG_INFINITY])?.is_nan());
        assert!(exact_sum(&[1.0, f64::NAN])?.is_nan());
        assert_eq!(
            exact_sum(&[f64::INFINITY, f64::MAX, f64::MAX])?,
            f64::INFINITY
        );
        Ok(())
    }

    #[test]
    fn intermediate_overflow_does_not_depend_on_order() -> Result<()> {
        assert_eq!(exact_sum(&[1e308, 1e308, -1e308])?, 1e308);
        assert_eq!(exact_sum(&[1e308, -1e308, 1e308])?, 1e308);
        Ok(())
    }

    #[test]
    fn overflows_to_infinity_as_addition() -> Result<()> {
        let half_ulp_of_max = pow2(970);
        assert_eq!(exact_sum(&[f64::MAX, f64::MAX])?, f64::INFINITY);
        assert_eq!(exact_sum(&[-f64::MAX, -f64::MAX])?, f64::NEG_INFINITY);
        assert_eq!(
            exact_sum(&[f64::MAX, half_ulp_of_max])?,
            f64::MAX + half_ulp_of_max
        );
        assert_eq!(
            exact_sum(&[f64::MAX, half_ulp_of_max, -pow2(-1074)])?,
            f64::MAX
        );
        Ok(())
    }

    #[test]
    fn merges_partial_states() -> Result<()> {
        let mut merged = FloatSumAccumulator::default();
        for values in [vec![1e308, 1e308], vec![], vec![-1e308]] {
            let mut partial = FloatSumAccumulator::default();
            partial.update_batch(&[Arc::new(Float64Array::from(values))])?;
            let state = partial
                .state()?
                .iter()
                .map(ScalarValue::to_array)
                .collect::<Result<Vec<_>>>()?;
            merged.merge_batch(&state)?;
        }
        assert_eq!(merged.evaluate()?, ScalarValue::Float64(Some(1e308)));
        Ok(())
    }

    /// DataFusion may apply `FILTER` before calling the accumulator, so SQL
    /// queries do not reliably exercise `opt_filter`.
    #[test]
    fn groups_accumulator_applies_filter() -> Result<()> {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(4.0),
            None,
            Some(8.0),
        ]));
        let filter = BooleanArray::from(vec![
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            None,
        ]);

        let mut acc = FloatSumGroupsAccumulator::default();
        acc.update_batch(&[Arc::clone(&values)], &[0, 1, 0, 2, 2], Some(&filter), 3)?;
        let sums = acc.evaluate(EmitTo::All)?;
        assert_eq!(
            as_float64_array(&sums)?,
            &Float64Array::from(vec![Some(1.0), Some(2.0), None])
        );

        let states = acc.convert_to_state(&[values], Some(&filter))?;
        let mut merged = FloatSumGroupsAccumulator::default();
        merged.merge_batch(&states, &[0, 0, 0, 0, 0], 1)?;
        let sums = merged.evaluate(EmitTo::All)?;
        assert_eq!(as_float64_array(&sums)?, &Float64Array::from(vec![3.0]));
        Ok(())
    }
}
