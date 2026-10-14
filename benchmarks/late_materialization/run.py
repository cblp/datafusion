#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Compare datafusion-cli builds and DuckDB on wide-table ClickBench queries.

Every query runs as `EXPLAIN ANALYZE`, which executes it fully without
printing the result. Variants are interleaved within each round (A, B, C, A,
B, C, ...) to spread thermal throttling and noise evenly. Each run is a fresh
process under `memguard.sh`, and its full output is kept in `<out>/logs/`.

See README.md for examples.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MEMGUARD = Path(__file__).resolve().parent / "memguard.sh"

GOOGLE = """"URL" LIKE '%google%'"""
RU = """"URL" LIKE '%.ru%'"""
WIDE_COLUMNS = '"URL", "Title", "Referer", "EventTime"'

QUERIES = {
    # Many output columns through a filter on one column
    "c1_google": f'SELECT "URL" FROM hits WHERE {GOOGLE}',
    "c4_google": f"SELECT {WIDE_COLUMNS} FROM hits WHERE {GOOGLE}",
    "all_google": f"SELECT * FROM hits WHERE {GOOGLE}",
    "c1_ru": f'SELECT "URL" FROM hits WHERE {RU}',
    "c4_ru": f"SELECT {WIDE_COLUMNS} FROM hits WHERE {RU}",
    "all_ru": f"SELECT * FROM hits WHERE {RU}",
    # TopK over all columns; topk_google is ClickBench Q23 (q23.sql in DataFusion)
    "topk_google": f'SELECT * FROM hits WHERE {GOOGLE} ORDER BY "EventTime" LIMIT 10',
    "topk_ru": f'SELECT * FROM hits WHERE {RU} ORDER BY "EventTime" LIMIT 10',
    "topk_nofilter": 'SELECT * FROM hits ORDER BY "EventTime" LIMIT 10',
    # The sort key is not correlated with the file order, so the TopK dynamic
    # filter cannot prune row groups
    "topk_watchid_google": f'SELECT * FROM hits WHERE {GOOGLE} ORDER BY "WatchID" LIMIT 10',
}
DEFAULT_QUERIES = ["topk_google", "topk_ru", "topk_nofilter", "topk_watchid_google"]

# Output lines that mean the query failed. Metrics such as
# `predicate_evaluation_errors=0` must not match.
ERROR_RE = re.compile(
    r"(?m)^(Error|.*Internal error|.*Execution error|.*Parquet error|memguard:)"
)


@dataclass(frozen=True)
class Variant:
    name: str
    binary: str
    duckdb: bool = False
    pushdown: bool = False
    late_materialization: bool = False


def parse_named(value: str) -> tuple[str, str]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, path


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants = []
    for (name, binary), pushdown, late in itertools.product(
        args.datafusion, args.pushdown, args.late_materialization
    ):
        suffix = f"pd_{'on' if pushdown else 'off'}"
        if late:
            suffix += "_lm"
        variants.append(Variant(f"{name}_{suffix}", binary, False, pushdown, late))
    if args.duckdb:
        variants.append(Variant("duckdb", args.duckdb, duckdb=True))
    return variants


def datafusion_sql(variant: Variant, hits: str, query: str, partitions: int) -> str:
    pushdown = "true" if variant.pushdown else "false"
    lines = [
        "SET datafusion.execution.parquet.binary_as_string = true;",
        f"SET datafusion.execution.target_partitions = {partitions};",
        f"SET datafusion.execution.parquet.pushdown_filters = {pushdown};",
        f"SET datafusion.execution.parquet.reorder_filters = {pushdown};",
        "SET datafusion.explain.format = 'indent';",
    ]
    # Only set when enabled, so that builds without the option still run.
    if variant.late_materialization:
        lines.append(
            "SET datafusion.optimizer.enable_parquet_late_materialization = true;"
        )
    lines += [
        f"CREATE EXTERNAL TABLE hits STORED AS PARQUET LOCATION '{hits}';",
        f"EXPLAIN ANALYZE {query};",
    ]
    return "\n".join(lines) + "\n"


def duckdb_sql(hits: str, query: str, threads: int) -> str:
    return (
        f"SET threads = {threads};\n"
        f"CREATE VIEW hits AS SELECT * FROM read_parquet('{hits}', binary_as_string = true);\n"
        ".timer on\n"
        f"EXPLAIN ANALYZE {query};\n"
    )


def run_once(
    variant: Variant, query: str, args: argparse.Namespace, log: Path
) -> float | None:
    """Run one query, save the full output to `log`, return the seconds."""
    if variant.duckdb:
        sql = duckdb_sql(args.hits, query, args.partitions)
        command = [variant.binary]
        time_re = r"Run Time \(s\): real ([\d.]+)"
    else:
        sql = datafusion_sql(variant, args.hits, query, args.partitions)
        command = [variant.binary, "--format", "csv"]
        time_re = r"Elapsed ([\d.]+) seconds"
    guard = [str(MEMGUARD), str(args.mem_limit_mb), str(args.timeout)]
    result = subprocess.run(guard + command, input=sql, capture_output=True, text=True)
    output = result.stdout + result.stderr
    log.write_text(f"{sql}\n----\n{output}")
    times = re.findall(time_re, output)
    if result.returncode != 0 or ERROR_RE.search(output) or not times:
        return None
    # The last timing belongs to EXPLAIN ANALYZE; earlier ones are SET/CREATE.
    return float(times[-1])


def summarize(results: list[dict]) -> str:
    variants = list(dict.fromkeys(r["variant"] for r in results))
    queries = list(dict.fromkeys(r["query"] for r in results))
    rows = [["query"] + variants]
    for query in queries:
        row = [query]
        for variant in variants:
            times = [
                r["seconds"]
                for r in results
                if r["query"] == query and r["variant"] == variant
            ]
            if not times:
                row.append("")
            elif None in times:
                row.append(f"ERR({times.count(None)}/{len(times)})")
            else:
                row.append(
                    f"{statistics.median(times):.2f} ({min(times):.2f}-{max(times):.2f})"
                )
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hits", help="path to ClickBench hits.parquet")
    parser.add_argument(
        "--datafusion",
        metavar="NAME=PATH",
        type=parse_named,
        action="append",
        default=[],
        help="datafusion-cli binary to compare; repeatable",
    )
    parser.add_argument("--duckdb", metavar="PATH", help="duckdb binary to compare")
    parser.add_argument(
        "--pushdown",
        default="off,on",
        help="pushdown_filters and reorder_filters values: off, on or off,on",
    )
    parser.add_argument(
        "--late-materialization",
        default="off",
        help="enable_parquet_late_materialization values: off, on or off,on",
    )
    parser.add_argument(
        "--queries",
        default=",".join(DEFAULT_QUERIES),
        help=f"comma-separated query names, or 'all'; known: {', '.join(QUERIES)}",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--partitions",
        type=int,
        default=8,
        help="target_partitions for DataFusion and threads for DuckDB "
        "(default: 8, the performance cores of the machine used)",
    )
    parser.add_argument("--mem-limit-mb", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300, help="seconds per run")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--summarize",
        type=Path,
        metavar="RESULTS_JSON",
        help="only print the summary of an earlier results.json",
    )
    args = parser.parse_args()

    if args.summarize:
        print(summarize(json.loads(args.summarize.read_text())))
        return 0

    if not args.hits:
        parser.error("--hits is required")
    if not args.datafusion and not args.duckdb:
        parser.error("give at least one --datafusion or --duckdb")

    def on_off(name: str, value: str) -> list[bool]:
        flags = {"off": False, "on": True}
        try:
            return [flags[v] for v in value.split(",")]
        except KeyError:
            parser.error(f"{name} must be off, on or off,on")

    args.pushdown = on_off("--pushdown", args.pushdown)
    args.late_materialization = on_off(
        "--late-materialization", args.late_materialization
    )
    names = list(QUERIES) if args.queries == "all" else args.queries.split(",")
    unknown = [name for name in names if name not in QUERIES]
    if unknown:
        parser.error(f"unknown queries: {', '.join(unknown)}")

    variants = build_variants(args)
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results = []
    for round_index in range(args.rounds):
        for query_name in names:
            for variant in variants:
                log = logs / f"{variant.name}_{query_name}_r{round_index}.txt"
                seconds = run_once(variant, QUERIES[query_name], args, log)
                results.append(
                    {
                        "round": round_index,
                        "query": query_name,
                        "variant": variant.name,
                        "seconds": seconds,
                    }
                )
                print(round_index, query_name, variant.name, seconds, flush=True)
                # Rewritten after every run, so an interrupted series is kept.
                results_path.write_text(json.dumps(results, indent=1))

    print()
    print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
