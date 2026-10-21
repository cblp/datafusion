<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Late materialization benchmark

Compares `datafusion-cli` builds, with and without filter pushdown and
`datafusion.optimizer.enable_parquet_late_materialization`, and DuckDB on
ClickBench queries that read many columns through a filter or a TopK.

## Requirements

- ClickBench `hits.parquet` (single file, ~14 GB), for example from
  `./bench.sh data clickbench_1`.
- One or more `datafusion-cli` binaries built with `--release`.
- Optionally a `duckdb` binary.
- Python 3.11+ (standard library only).

`--late-materialization` takes a comma-separated list of modes: `off`; `on`,
the physical rule (`enable_parquet_late_materialization`, variants `_lm`); and
`join`, the logical semi join rule (`enable_late_materialization_join`,
variants `_lj`). Builds without an option report its variants as `ERR`.

## Usage

Reproduce the comparison of the late materialization prototype with the
upstream behavior and DuckDB:

```shell
cargo build --release -p datafusion-cli
python3 benchmarks/late_materialization/run.py \
  --hits benchmarks/data/hits.parquet \
  --datafusion main=target/release/datafusion-cli \
  --late-materialization off,on,join \
  --duckdb "$(which duckdb)" \
  --out /tmp/late_materialization
```

Compare two DataFusion builds on the filter queries, without late
materialization:

```shell
python3 benchmarks/late_materialization/run.py \
  --hits benchmarks/data/hits.parquet \
  --datafusion release="$(which datafusion-cli)" \
  --datafusion main=target/release/datafusion-cli \
  --queries c1_google,c4_google,all_google,c1_ru,c4_ru,all_ru \
  --out /tmp/filters
```

Calibrate `late_materialization_min_saved_bytes_ratio`: disable the threshold
with `--set` (any DataFusion setting, repeatable) and compare queries whose late
columns are a growing share of the bytes:

```shell
python3 benchmarks/late_materialization/run.py \
  --hits benchmarks/data/hits.parquet \
  --datafusion main=target/release/datafusion-cli \
  --late-materialization off,on \
  --set datafusion.optimizer.late_materialization_min_saved_bytes_ratio=0 \
  --queries topk_google_ints,topk_google_title,topk_google_title_referer,topk_google \
  --out /tmp/calibration
```

The `ParquetFetchByRowIdExec` line of each `_lm` log shows the fetch I/O
estimate: `estimated_page_requests` / `estimated_page_bytes` and
`estimated_chunk_requests` / `estimated_chunk_bytes`.

Print the summary of an earlier run:

```shell
python3 benchmarks/late_materialization/run.py --summarize /tmp/late_materialization/results.json
```

The summary shows the median and the min-max range in seconds;
`ERR(k/n)` means `k` of `n` runs failed, were killed for memory or timed out
(see `<out>/logs/`).

## Method

- Every query runs as `EXPLAIN ANALYZE`: it executes fully without printing
  the result, and the log keeps the metrics of each operator.
- Each run is a fresh process. Variants are interleaved within a round, so
  thermal throttling and background noise spread evenly. Warm the OS cache
  first (for example `cat hits.parquet > /dev/null`) if hot runs are wanted.
- `--partitions` sets both `target_partitions` and DuckDB `threads`; on
  machines with efficiency cores set it to the number of performance cores.
- Each run is limited by `memguard.sh` (`--mem-limit-mb`, `--timeout`),
  because `ulimit -v` is not enforced on macOS.
- Compare numbers only within one series: the same variant can differ by
  tens of percent between series.

## Profiling

On macOS, build with symbols and sample a running query:

```shell
cargo build --profile profiling -p datafusion-cli
target/profiling/datafusion-cli -f queries.sql &
sample $! 20 -file profile.txt
```

`profile.txt` ends with the functions sorted by the number of samples on
top of the stack.
