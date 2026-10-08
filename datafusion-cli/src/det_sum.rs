//! `sum` with deterministic floating point summation.
//!
//! Floating point addition is not associative, so the result of the built-in
//! `sum` depends on how rows are split into partitions and batches. For
//! floating point input, [`det_sum`] instead computes the exact sum and rounds
//! it once, using [`bitrep::SumF64`]:
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

use {
    bitrep::SumF64,
    datafusion::{
        arrow::{
            array::{
                Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float64Array,
                ListArray,
            },
            datatypes::{DataType, Field, FieldRef, Float64Type},
        },
        common::{
            Result, ScalarValue,
            cast::{as_fixed_size_binary_array, as_float64_array, as_list_array},
            internal_err,
        },
        functions_aggregate::sum::Sum,
        logical_expr::{
            Accumulator, AggregateUDF, AggregateUDFImpl, Documentation, EmitTo, Expr,
            GroupsAccumulator, Operator, ReversedUDAF, SetMonotonicity, Signature,
            StatisticsArgs,
            expr::AggregateFunction,
            function::{AccumulatorArgs, StateFieldsArgs},
            utils::{AggregateOrderSensitivity, format_state_name},
        },
    },
    std::{collections::HashSet, sync::Arc},
};

/// The built-in `sum`, except that floating point input is summed exactly and
/// rounded once, so the result does not depend on the summation order.
pub fn det_sum() -> AggregateUDF {
    AggregateUDF::from(DetSumUDAF::default())
}

/// Delegates to the built-in [`Sum`] unless the input is `Float64`, to which
/// `sum` coerces all floating point types.
#[derive(Debug, Default, PartialEq, Eq, Hash)]
struct DetSumUDAF {
    builtin: Sum,
}

fn sums_floats(args: &AccumulatorArgs) -> bool {
    args.return_field.data_type() == &DataType::Float64
}

/// The state is [`SumF64::to_bytes`].
const STATE_BYTES: i32 = SumF64::BYTES as i32;

impl AggregateUDFImpl for DetSumUDAF {
    fn name(&self) -> &str {
        self.builtin.name()
    }

    fn signature(&self) -> &Signature {
        self.builtin.signature()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.builtin.return_type(arg_types)
    }

    /// `DISTINCT` keeps the built-in state, a list of the distinct values.
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        if args.return_type() == &DataType::Float64 && !args.is_distinct {
            let name = format_state_name(args.name, "sum");
            Ok(vec![
                Field::new(name, DataType::FixedSizeBinary(STATE_BYTES), true).into(),
            ])
        } else {
            self.builtin.state_fields(args)
        }
    }

    fn accumulator(&self, args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        match (sums_floats(&args), args.is_distinct) {
            (true, false) => Ok(Box::new(DetSumAccumulator::default())),
            (true, true) => Ok(Box::new(DistinctDetSumAccumulator::default())),
            (false, _) => self.builtin.accumulator(args),
        }
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        self.builtin.groups_accumulator_supported(args)
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        if sums_floats(&args) {
            Ok(Box::new(DetSumGroupsAccumulator::default()))
        } else {
            self.builtin.create_groups_accumulator(args)
        }
    }

    fn create_sliding_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        if sums_floats(&args) && !args.is_distinct {
            Ok(Box::new(SlidingDetSumAccumulator::default()))
        } else {
            self.builtin.create_sliding_accumulator(args)
        }
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        self.builtin.reverse_expr()
    }

    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
        self.builtin.order_sensitivity()
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.builtin.documentation()
    }

    fn set_monotonicity(&self, data_type: &DataType) -> SetMonotonicity {
        self.builtin.set_monotonicity(data_type)
    }

    /// The built-in rewrite of `sum(x + c)` to `sum(x) + c * count(x)` rounds
    /// differently from the exact sum of `x + c`.
    fn simplify_expr_op_literal(
        &self,
        agg_function: &AggregateFunction,
        arg: &Expr,
        op: Operator,
        lit: &Expr,
        arg_is_left: bool,
    ) -> Result<Option<Expr>> {
        if matches!(lit, Expr::Literal(value, _) if value.data_type().is_floating()) {
            return Ok(None);
        }
        self.builtin
            .simplify_expr_op_literal(agg_function, arg, op, lit, arg_is_left)
    }

    /// Floating point sums in statistics are not computed exactly.
    fn value_from_stats(&self, args: &StatisticsArgs) -> Option<ScalarValue> {
        if args.return_type == &DataType::Float64 {
            return None;
        }
        self.builtin.value_from_stats(args)
    }
}

fn decode_state(bytes: &[u8]) -> Result<SumF64> {
    let Some(sum) = bytes.try_into().ok().and_then(SumF64::from_bytes) else {
        return internal_err!("invalid sum state");
    };
    Ok(sum)
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
struct DetSumAccumulator {
    sum: SumF64,
}

impl Accumulator for DetSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.sum
            .extend(as_float64_array(&values[0])?.iter().flatten());
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        for bytes in as_fixed_size_binary_array(&states[0])?.iter().flatten() {
            self.sum.merge(&decode_state(bytes)?);
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

/// `sum(DISTINCT x)`. The built-in one adds up a hash set in its iteration
/// order, which varies between runs.
#[derive(Debug, Default)]
struct DistinctDetSumAccumulator {
    /// Bit patterns of the distinct values.
    values: HashSet<u64>,
}

impl Accumulator for DistinctDetSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let values = as_float64_array(&values[0])?;
        self.values
            .extend(values.iter().flatten().map(f64::to_bits));
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
/// and infinities cannot be cleared, so non-finite values are counted here.
#[derive(Debug, Default)]
struct SlidingDetSumAccumulator {
    finite_sum: SumF64,
    count: u64,
    nan_count: u64,
    pos_inf_count: u64,
    neg_inf_count: u64,
}

impl SlidingDetSumAccumulator {
    fn non_finite_count(&mut self, x: f64) -> Option<&mut u64> {
        if x.is_nan() {
            Some(&mut self.nan_count)
        } else if x == f64::INFINITY {
            Some(&mut self.pos_inf_count)
        } else if x == f64::NEG_INFINITY {
            Some(&mut self.neg_inf_count)
        } else {
            None
        }
    }
}

impl Accumulator for SlidingDetSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for x in as_float64_array(&values[0])?.iter().flatten() {
            self.count += 1;
            match self.non_finite_count(x) {
                Some(count) => *count += 1,
                None => self.finite_sum.add(x),
            }
        }
        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for x in as_float64_array(&values[0])?.iter().flatten() {
            self.count -= 1;
            match self.non_finite_count(x) {
                Some(count) => *count -= 1,
                None => self.finite_sum.add(-x),
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
        let sum =
            if self.nan_count > 0 || (self.pos_inf_count > 0 && self.neg_inf_count > 0) {
                f64::NAN
            } else if self.pos_inf_count > 0 {
                f64::INFINITY
            } else if self.neg_inf_count > 0 {
                f64::NEG_INFINITY
            } else {
                self.finite_sum.value()
            };
        Ok(ScalarValue::Float64((self.count > 0).then_some(sum)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

/// One [`SumF64`] per group, indexed by the group index.
#[derive(Debug, Default)]
struct DetSumGroupsAccumulator {
    sums: Vec<SumF64>,
}

/// Non-null input values that pass the filter, as `(row, value)`.
fn selected_values<'a>(
    values: &'a Float64Array,
    opt_filter: Option<&'a BooleanArray>,
) -> impl Iterator<Item = (usize, f64)> + 'a {
    values.iter().enumerate().filter_map(move |(row, value)| {
        let selected =
            opt_filter.is_none_or(|filter| filter.is_valid(row) && filter.value(row));
        Some((row, value.filter(|_| selected)?))
    })
}

impl GroupsAccumulator for DetSumGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.sums.resize(total_num_groups, SumF64::new());
        for (row, value) in selected_values(as_float64_array(&values[0])?, opt_filter) {
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
        let states = as_fixed_size_binary_array(&values[0])?;
        for (bytes, &group) in states.iter().zip(group_indices) {
            if let Some(bytes) = bytes {
                self.sums[group].merge(&decode_state(bytes)?);
            }
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
        let values = as_float64_array(&values[0])?;
        let mut sums = vec![SumF64::new(); values.len()];
        for (row, value) in selected_values(values, opt_filter) {
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
    use {
        super::*,
        datafusion::{
            arrow::{array::Int64Array, record_batch::RecordBatch},
            common::cast::as_int64_array,
            datasource::MemTable,
            execution::FunctionRegistry,
            functions_aggregate::expr_fn::{count, sum},
            prelude::*,
        },
        std::collections::BTreeMap,
    };

    fn exact_sum(values: &[f64]) -> Result<f64> {
        let mut acc = DetSumAccumulator::default();
        acc.update_batch(&[Arc::new(Float64Array::from(values.to_vec()))])?;
        match acc.evaluate()? {
            ScalarValue::Float64(Some(sum)) => Ok(sum),
            other => panic!("unexpected det_sum result: {other:?}"),
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
        let mut merged = DetSumAccumulator::default();
        for values in [vec![1e308, 1e308], vec![], vec![-1e308]] {
            let mut partial = DetSumAccumulator::default();
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

    /// `γₖ = k·u / (1 − k·u)`: summing `n` floats in any order differs from the
    /// exact sum by at most `γₙ₋₁·Σ|x|` (Higham, "Accuracy and Stability of
    /// Numerical Algorithms", ch. 4).
    fn gamma(k: i64) -> f64 {
        let ku = k as f64 * f64::EPSILON / 2.0;
        ku / (1.0 - ku)
    }

    #[tokio::test]
    async fn det_sum_approximately_equals_sum_on_random_data() -> Result<()> {
        let config = SessionConfig::new().with_target_partitions(4);
        let ctx = SessionContext::new_with_config(config);

        // Same-sign values make `Σ|x|` equal to `|sum|`, so the tolerance
        // stays relative to the result instead of growing with cancellation.
        let batches = ctx
            .sql(
                "SELECT
                    CASE WHEN random() < 0.1 THEN
                        NULL
                    ELSE
                        random() * 1e6
                    END
                    AS x
                FROM generate_series(1, 100000)",
            )
            .await?
            .aggregate(
                vec![],
                vec![
                    sum(col("x")).alias("builtin"),
                    det_sum().call(vec![col("x")]).alias("deterministic"),
                    count(col("x")),
                    sum(abs(col("x"))),
                ],
            )?
            .collect()
            .await?;

        let batch = &batches[0];
        let sum = as_float64_array(batch.column(0))?.value(0);
        let det_sum = as_float64_array(batch.column(1))?.value(0);
        let n = as_int64_array(batch.column(2))?.value(0);
        let abs_sum = as_float64_array(batch.column(3))?.value(0);

        // `sum` is within `gamma(n - 1) * Σ|x|` of the exact sum, and the
        // correctly rounded `det_sum` within half an ulp of it.
        let tolerance = gamma(n - 1) * abs_sum + f64::EPSILON / 2.0 * det_sum.abs();
        assert!(
            (sum - det_sum).abs() <= tolerance,
            "|{sum} - {det_sum}| > {tolerance}"
        );
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

        let mut acc = DetSumGroupsAccumulator::default();
        acc.update_batch(&[Arc::clone(&values)], &[0, 1, 0, 2, 2], Some(&filter), 3)?;
        let sums = acc.evaluate(EmitTo::All)?;
        assert_eq!(
            as_float64_array(&sums)?,
            &Float64Array::from(vec![Some(1.0), Some(2.0), None])
        );

        let states = acc.convert_to_state(&[values], Some(&filter))?;
        let mut merged = DetSumGroupsAccumulator::default();
        merged.merge_batch(&states, &[0, 0, 0, 0, 0], 1)?;
        let sums = merged.evaluate(EmitTo::All)?;
        assert_eq!(as_float64_array(&sums)?, &Float64Array::from(vec![3.0]));
        Ok(())
    }

    /// `GROUP BY` goes through [`DetSumGroupsAccumulator`], and skipping
    /// partial aggregation through its `convert_to_state`.
    #[tokio::test]
    async fn group_by_matches_per_group_sums() -> Result<()> {
        let data =
            SessionContext::new_with_config(SessionConfig::new().with_batch_size(1000))
                .sql(
                    "SELECT
                        v % 100 AS g,
                        CASE WHEN random() < 0.1 THEN
                            NULL
                        ELSE
                            (random() - 0.5) * pow(10, random() * 20)
                        END
                        AS x
                    FROM generate_series(1, 10000) t(v)",
                )
                .await?
                .collect()
                .await?;

        let mut sums: BTreeMap<i64, (SumF64, SumF64)> = BTreeMap::new();
        for batch in &data {
            let groups = as_int64_array(batch.column(0))?;
            let values = as_float64_array(batch.column(1))?;
            for (&group, value) in groups.values().iter().zip(values) {
                let (all, positive) = sums.entry(group).or_default();
                if let Some(x) = value {
                    all.add(x);
                    if x > 0.0 {
                        positive.add(x);
                    }
                }
            }
        }
        let expected: Vec<_> = sums
            .iter()
            .map(|(&group, (all, positive))| (group, result(all), result(positive)))
            .collect();

        for skip_partial_aggregation in [false, true] {
            let mut config = SessionConfig::new().with_target_partitions(4);
            if skip_partial_aggregation {
                let execution = &mut config.options_mut().execution;
                execution.skip_partial_aggregation_probe_rows_threshold = 0;
                execution.skip_partial_aggregation_probe_ratio_threshold = 0.0;
            }
            let ctx = SessionContext::new_with_config(config);
            ctx.register_udaf(det_sum());
            let partitions = data.chunks(3).map(<[_]>::to_vec).collect();
            ctx.register_table(
                "t",
                Arc::new(MemTable::try_new(data[0].schema(), partitions)?),
            )?;

            let batches = ctx
                .sql(
                    "SELECT g, sum(x), sum(x) FILTER (WHERE x > 0)
                    FROM t GROUP BY g ORDER BY g",
                )
                .await?
                .collect()
                .await?;
            let mut actual = vec![];
            for batch in &batches {
                let groups = as_int64_array(batch.column(0))?;
                let all = as_float64_array(batch.column(1))?;
                let positive = as_float64_array(batch.column(2))?;
                for ((&group, all), positive) in
                    groups.values().iter().zip(all).zip(positive)
                {
                    actual.push((group, all, positive));
                }
            }
            assert_eq!(
                actual, expected,
                "skip_partial_aggregation = {skip_partial_aggregation}"
            );
        }
        Ok(())
    }

    /// Runs `sum` with and without `DISTINCT` over `data` shuffled, cut into
    /// batches of `batch_size` rows and spread over `partitions` partitions.
    async fn shuffled_det_sum(
        data: &[RecordBatch],
        partitions: usize,
        batch_size: usize,
    ) -> Result<Vec<ScalarValue>> {
        let config = SessionConfig::new()
            .with_target_partitions(partitions)
            .with_batch_size(batch_size);
        let ctx = SessionContext::new_with_config(config);
        ctx.register_udaf(det_sum());

        let schema = data[0].schema();
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(schema.clone(), vec![data.to_vec()])?),
        )?;
        let shuffled = ctx
            .sql("SELECT x FROM t ORDER BY random()")
            .await?
            .collect()
            .await?;

        let mut partitioned = vec![vec![]; partitions];
        for (i, batch) in shuffled.into_iter().enumerate() {
            partitioned[i % partitions].push(batch);
        }
        ctx.register_table(
            "shuffled",
            Arc::new(MemTable::try_new(schema, partitioned)?),
        )?;

        let result = ctx
            .sql(
                "SELECT sum(x), sum(DISTINCT x), sum(DISTINCT -x), count(DISTINCT x)
                FROM shuffled",
            )
            .await?
            .collect()
            .await?;
        result[0]
            .columns()
            .iter()
            .map(|column| ScalarValue::try_from_array(column, 0))
            .collect()
    }

    #[tokio::test]
    async fn det_sum_does_not_depend_on_summation_order() -> Result<()> {
        // Mixed signs and magnitudes spanning 20 orders make plain floating
        // point summation highly order-dependent.
        let data = SessionContext::new()
            .sql(
                "SELECT
                    CASE WHEN random() < 0.1 THEN
                        NULL
                    ELSE
                        (random() - 0.5) * pow(10, random() * 20)
                    END
                    AS x
                FROM generate_series(1, 10000)",
            )
            .await?
            .collect()
            .await?;

        let expected = shuffled_det_sum(&data, 1, 8192).await?;
        for (partitions, batch_size) in [(1, 8192), (1, 1), (2, 1000), (4, 100), (8, 7)] {
            let actual = shuffled_det_sum(&data, partitions, batch_size).await?;
            assert_eq!(
                actual, expected,
                "partitions = {partitions}, batch_size = {batch_size}"
            );
        }
        Ok(())
    }

    /// Runs `sql` with the built-in `sum` and with [`det_sum`].
    async fn builtin_and_det_sum(
        sql: &str,
    ) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
        let builtin = SessionContext::new().sql(sql).await?.collect().await?;
        let ctx = SessionContext::new();
        ctx.register_udaf(det_sum());
        let deterministic = ctx.sql(sql).await?.collect().await?;
        Ok((builtin, deterministic))
    }

    #[tokio::test]
    async fn replaces_builtin_sum() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_udaf(det_sum());
        assert_eq!(*ctx.udaf("sum")?, det_sum());
        Ok(())
    }

    #[tokio::test]
    async fn sums_other_types_as_builtin() -> Result<()> {
        let sql = "SELECT
                sum(i), sum(DISTINCT i), sum(u), sum(d),
                arrow_typeof(sum(i)), arrow_typeof(sum(u)), arrow_typeof(sum(d))
            FROM (VALUES
                (1, CAST(1 AS INT UNSIGNED), CAST(1.25 AS DECIMAL(10, 2))),
                (1, CAST(2 AS INT UNSIGNED), CAST(2.50 AS DECIMAL(10, 2))),
                (3, CAST(3 AS INT UNSIGNED), CAST(3.75 AS DECIMAL(10, 2)))
            ) AS t(i, u, d)";
        let (builtin, deterministic) = builtin_and_det_sum(sql).await?;
        assert_eq!(builtin, deterministic);
        Ok(())
    }

    #[tokio::test]
    async fn sums_all_float_types() -> Result<()> {
        // DataFusion rejects `sum(x)` and `sum(CAST(x AS FLOAT))` in one query
        // as duplicate names, even with aliases.
        for x in ["x", "CAST(x AS FLOAT)"] {
            let sql =
                format!("SELECT sum({x}) FROM (VALUES (1.0), (1e16), (-1e16)) AS t(x)");
            let (_, deterministic) = builtin_and_det_sum(&sql).await?;
            assert_eq!(
                as_float64_array(deterministic[0].column(0))?.value(0),
                1.0,
                "{x}"
            );
        }
        Ok(())
    }

    /// A single `sum(DISTINCT x)` is rewritten to a plain `sum` over
    /// `GROUP BY x`; distinct sums of different arguments are not.
    #[tokio::test]
    async fn distinct_sum_is_exact() -> Result<()> {
        let sql = "SELECT sum(DISTINCT x), sum(DISTINCT -x)
            FROM (VALUES (1e16), (1.0), (-1e16), (1.0), (1e16)) AS t(x)";
        let (_, deterministic) = builtin_and_det_sum(sql).await?;
        assert_eq!(as_float64_array(deterministic[0].column(0))?.value(0), 1.0);
        assert_eq!(as_float64_array(deterministic[0].column(1))?.value(0), -1.0);
        Ok(())
    }

    /// The built-in rewrite of `sum(x + c)` to `sum(x) + c * count(x)` would
    /// round differently.
    #[tokio::test]
    async fn sum_of_x_plus_literal_is_exact() -> Result<()> {
        let values = [1e16, 1.0, -1e16];
        let sql = "SELECT sum(x), sum(x + 1.0), sum(x + 2.0)
            FROM (VALUES (1e16), (1.0), (-1e16)) AS t(x)";
        let (_, deterministic) = builtin_and_det_sum(sql).await?;
        for (column, c) in deterministic[0].columns().iter().zip([0.0, 1.0, 2.0]) {
            let expected = exact_sum(&values.map(|x| x + c))?;
            assert_eq!(as_float64_array(column)?.value(0), expected, "c = {c}");
        }
        Ok(())
    }

    /// Sliding frames add values entering the frame and remove values leaving
    /// it; the result must still be the exact sum of the frame.
    #[tokio::test]
    async fn window_frames_are_summed_exactly() -> Result<()> {
        let values = [
            1e16,
            1.0,
            -1e16,
            f64::INFINITY,
            2.0,
            f64::NAN,
            3.0,
            f64::NEG_INFINITY,
            1e308,
            1e308,
            -1e308,
            4.0,
            5.0,
            6.0,
        ];
        let batch = RecordBatch::try_from_iter([
            (
                "i",
                Arc::new(Int64Array::from_iter_values(0..values.len() as i64))
                    as ArrayRef,
            ),
            (
                "x",
                Arc::new(Float64Array::from(values.to_vec())) as ArrayRef,
            ),
        ])?;
        let ctx = SessionContext::new();
        ctx.register_udaf(det_sum());
        ctx.register_batch("t", batch)?;

        let batches = ctx
            .sql(
                "SELECT sum(x) OVER (ORDER BY i ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                FROM t ORDER BY i",
            )
            .await?
            .collect()
            .await?;
        let actual: Vec<u64> = as_float64_array(batches[0].column(0))?
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let expected = (0..values.len())
            .map(|i| Ok(exact_sum(&values[i.saturating_sub(2)..=i])?.to_bits()))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
