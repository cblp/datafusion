//! `det_sum`: a deterministic floating point sum.
//!
//! Floating point addition is not associative, so the result of `sum` depends
//! on how rows are split into partitions and batches. `det_sum` instead
//! computes the exact sum and rounds it once:
//!
//! 1. Every finite `f64` is an integer multiple of 2⁻¹⁰⁷⁴, the smallest
//!    subnormal. It is converted to that integer, stored in a 2176-bit two's
//!    complement number, and added with integer arithmetic. Integer addition
//!    is exact, so the sum does not depend on the order of the values, and
//!    the width leaves room for 2⁶⁴ values of any magnitude without overflow.
//! 2. Non-finite values are summed separately: any NaN or both infinities
//!    give NaN, a single kind of infinity gives that infinity.
//! 3. Partial aggregates are merged by adding their integers.
//! 4. The final integer is rounded to the nearest `f64`, ties to even, as a
//!    single IEEE 754 addition would; sums beyond the `f64` range become ±inf.

use {
    datafusion::{
        arrow::{
            array::{Array, ArrayRef, ListArray},
            datatypes::{DataType, UInt64Type},
        },
        common::{
            Result, ScalarValue,
            cast::{as_float64_array, as_list_array, as_uint64_array},
            internal_err,
        },
        logical_expr::{Accumulator, AggregateUDF, Volatility, create_udaf},
    },
    std::sync::Arc,
};

/// The exact sum of the input rounded to the nearest `f64`, so it does not
/// depend on the summation order.
pub fn det_sum() -> AggregateUDF {
    create_udaf(
        "det_sum",
        vec![DataType::Float64],
        Arc::new(DataType::Float64),
        Volatility::Immutable,
        Arc::new(|_| Ok(Box::new(ExactSumAccumulator::default()))),
        Arc::new(vec![
            DataType::new_list(DataType::UInt64, true),
            DataType::Float64,
        ]),
    )
}

/// Holds any sum of up to 2⁶⁴ finite `f64` values plus a sign bit:
/// 1074 fraction bits + 1024 integer bits + 64 + 1 ≤ 34 · 64.
const LIMBS: usize = 34;

/// A two's complement fixed-point number whose least significant bit is 2⁻¹⁰⁷⁴,
/// the smallest subnormal `f64`. Every finite `f64` is exactly representable,
/// so sums are exact and do not depend on the order of additions.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FixedPoint([u64; LIMBS]);

impl Default for FixedPoint {
    fn default() -> Self {
        Self([0; LIMBS])
    }
}

impl FixedPoint {
    fn from_f64(x: f64) -> Self {
        let bits = x.to_bits();
        let biased_exponent = (bits >> 52) & 0x7ff;
        let fraction = bits & ((1 << 52) - 1);
        // |x| = mantissa · 2^(shift − 1074)
        let (mantissa, shift) = if biased_exponent == 0 {
            (fraction, 0)
        } else {
            (fraction | (1 << 52), biased_exponent - 1)
        };

        let mut result = Self::default();
        let limb = (shift / 64) as usize;
        let wide = u128::from(mantissa) << (shift % 64);
        result.0[limb] = wide as u64;
        result.0[limb + 1] = (wide >> 64) as u64;
        if x.is_sign_negative() {
            result.negated()
        } else {
            result
        }
    }

    fn add(&mut self, other: &Self) {
        let mut carry = false;
        for (limb, &other_limb) in self.0.iter_mut().zip(&other.0) {
            let (sum, carry1) = limb.overflowing_add(other_limb);
            let (sum, carry2) = sum.overflowing_add(u64::from(carry));
            *limb = sum;
            carry = carry1 || carry2;
        }
    }

    fn negated(&self) -> Self {
        let mut result = Self(self.0.map(|limb| !limb));
        let mut one = Self::default();
        one.0[0] = 1;
        result.add(&one);
        result
    }

    fn is_negative(&self) -> bool {
        self.0[LIMBS - 1] >> 63 == 1
    }

    fn bit(&self, i: usize) -> bool {
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    fn bit_len(&self) -> usize {
        self.0
            .iter()
            .rposition(|&limb| limb != 0)
            .map_or(0, |i| i * 64 + 64 - self.0[i].leading_zeros() as usize)
    }

    /// Rounds to the nearest `f64`, ties to even, as IEEE 754 addition does.
    fn to_f64(self) -> f64 {
        if self.is_negative() {
            return -self.negated().to_f64();
        }

        let len = self.bit_len();
        if len <= 53 {
            // A subnormal or the smallest normal exponent: the value in units
            // of 2⁻¹⁰⁷⁴ is exactly the bit pattern.
            return f64::from_bits(self.0[0]);
        }

        let shift = len - 53;
        let mut mantissa = (shift..len)
            .rev()
            .fold(0, |mantissa, i| (mantissa << 1) | u64::from(self.bit(i)));
        let half = self.bit(shift - 1);
        let more_than_half = (0..shift - 1).any(|i| self.bit(i));
        if half && (more_than_half || mantissa & 1 == 1) {
            mantissa += 1;
        }
        // The mantissa includes the implicit leading bit, so adding it to the
        // exponent field bumps the exponent by one; so does a rounding carry.
        let bits = ((shift as u64) << 52) + mantissa;
        f64::from_bits(bits.min(f64::INFINITY.to_bits()))
    }
}

#[derive(Debug, Default)]
struct ExactSumAccumulator {
    finite_sum: FixedPoint,
    /// Sum of the non-finite inputs: 0, ±inf or NaN.
    special_sum: f64,
    seen_any: bool,
}

impl Accumulator for ExactSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        for x in as_float64_array(&values[0])?.iter().flatten() {
            self.seen_any = true;
            if x.is_finite() {
                self.finite_sum.add(&FixedPoint::from_f64(x));
            } else {
                self.special_sum += x;
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let finite_sums = as_list_array(&states[0])?;
        let special_sums = as_float64_array(&states[1])?;
        for i in 0..finite_sums.len() {
            if finite_sums.is_null(i) {
                continue;
            }
            let limbs = finite_sums.value(i);
            let Ok(limbs) = as_uint64_array(&limbs)?.values().as_ref().try_into() else {
                return internal_err!("det_sum state must have {LIMBS} limbs");
            };
            self.seen_any = true;
            self.finite_sum.add(&FixedPoint(limbs));
            self.special_sum += special_sums.value(i);
        }
        Ok(())
    }

    /// The limbs of `finite_sum`, NULL when no values were seen, and `special_sum`.
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let limbs = self.seen_any.then(|| self.finite_sum.0.map(Some));
        let limbs = ListArray::from_iter_primitive::<UInt64Type, _, _>([limbs]);
        Ok(vec![
            ScalarValue::List(Arc::new(limbs)),
            ScalarValue::Float64(Some(self.special_sum)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let sum = if self.special_sum != 0.0 {
            self.special_sum
        } else {
            self.finite_sum.to_f64()
        };
        Ok(ScalarValue::Float64(self.seen_any.then_some(sum)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        datafusion::{
            arrow::{array::Float64Array, record_batch::RecordBatch},
            common::cast::as_int64_array,
            datasource::MemTable,
            prelude::*,
        },
    };

    fn exact_sum(values: &[f64]) -> Result<f64> {
        let mut acc = ExactSumAccumulator::default();
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
    fn matches_python_fsum() -> Result<()> {
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
        let mut merged = ExactSumAccumulator::default();
        for values in [vec![1e308, 1e308], vec![], vec![-1e308]] {
            let mut partial = ExactSumAccumulator::default();
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
        ctx.register_udaf(det_sum());

        // Same-sign values make `Σ|x|` equal to `|sum|`, so the tolerance
        // stays relative to the result instead of growing with cancellation.
        let batches = ctx
            .sql(
                "WITH t AS (
                    SELECT
                        CASE WHEN random() < 0.1 THEN
                            NULL
                        ELSE
                            random() * 1e6
                        END
                        AS x
                    FROM generate_series(1, 100000)
                )
                SELECT sum(x), det_sum(x), count(x), sum(abs(x)) FROM t",
            )
            .await?
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

    /// Runs `det_sum` over `data` shuffled, cut into batches of `batch_size`
    /// rows and spread over `partitions` partitions.
    async fn shuffled_det_sum(
        data: &[RecordBatch],
        partitions: usize,
        batch_size: usize,
    ) -> Result<ScalarValue> {
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
            .sql("SELECT det_sum(x) FROM shuffled")
            .await?
            .collect()
            .await?;
        ScalarValue::try_from_array(result[0].column(0), 0)
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
}
