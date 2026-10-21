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

//! Late materialization of a TopK over a single Parquet file, as a semi join
//! on the file row index.
//!
//! ```text
//! Sort: EventTime, fetch=10
//!   Filter: URL LIKE '%google%'
//!     TableScan: hits
//! ```
//!
//! becomes
//!
//! ```text
//! Projection: <original columns>
//!   Sort: EventTime, fetch=10                        -- restore the order
//!     LeftSemi Join: full.rid = top.rid
//!       SubqueryAlias: full                          -- all columns
//!         Projection: <original columns>, file_row_index() AS rid
//!           TableScan: hits
//!       SubqueryAlias: top                           -- the n winning row ids
//!         Projection: rid
//!           Sort: EventTime, fetch=10
//!             Projection: EventTime, file_row_index() AS rid
//!               Filter: URL LIKE '%google%'
//!                 TableScan: hits
//! ```
//!
//! The physical join builds on the `n` row ids and pushes its dynamic filter
//! on `rid` into the scan of all columns, which turns it into a row selection
//! (see `ParquetSource::row_index_predicate`), so only the `n` rows are read.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion_catalog::default_table_source::source_as_provider;
use datafusion_catalog_listing::ListingTable;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Column, Result, TableReference};
use datafusion_datasource_parquet::file_format::ParquetFormat;
use datafusion_expr::logical_plan::{LogicalPlan, LogicalPlanBuilder, TableScan};
use datafusion_expr::{Expr, JoinType, SortExpr, col};
use datafusion_optimizer::optimizer::ApplyOrder;
use datafusion_optimizer::{OptimizerConfig, OptimizerRule};

/// Name of the row id column added on both sides of the join.
const ROW_ID: &str = "__late_materialization_rid";
/// Alias of the scan of all columns.
const FULL: &str = "__late_materialization_full";
/// Alias of the narrow TopK.
const TOP: &str = "__late_materialization_top";

/// Rewrites `Sort(fetch) <- Filter* <- TableScan` over a single Parquet file
/// into a semi join on the file row index. Enabled by
/// `datafusion.optimizer.enable_late_materialization_join`.
#[derive(Debug, Default)]
pub struct LateMaterializationJoin {}

impl LateMaterializationJoin {
    /// Create a new [`LateMaterializationJoin`] rule.
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for LateMaterializationJoin {
    fn name(&self) -> &str {
        "late_materialization_join"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let options = config.options();
        if !options.optimizer.enable_late_materialization_join {
            return Ok(Transformed::no(plan));
        }
        match try_rewrite(&plan, options.optimizer.late_materialization_max_rows)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

fn try_rewrite(plan: &LogicalPlan, max_rows: usize) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::Sort(sort) = plan else {
        return Ok(None);
    };
    let Some(fetch) = sort.fetch.filter(|fetch| *fetch <= max_rows) else {
        return Ok(None);
    };

    // Only filters between the TopK and the scan.
    let mut needed = BTreeSet::new();
    let mut current = sort.input.as_ref();
    let scan = loop {
        match current {
            LogicalPlan::Filter(filter) => {
                needed.extend(filter.predicate.column_refs().into_iter().cloned());
                current = filter.input.as_ref();
            }
            LogicalPlan::TableScan(scan) => break scan,
            _ => return Ok(None),
        }
    };
    if scan.fetch.is_some() || !is_single_parquet_file(scan) {
        return Ok(None);
    }
    let output = sort.input.schema();
    if output.field_with_unqualified_name(ROW_ID).is_ok() {
        return Ok(None);
    }
    for filter in &scan.filters {
        needed.extend(filter.column_refs().into_iter().cloned());
    }
    let sort_columns: BTreeSet<Column> = sort
        .expr
        .iter()
        .flat_map(|sort_expr| sort_expr.expr.column_refs().into_iter().cloned())
        .collect();
    needed.extend(sort_columns.iter().cloned());
    if output
        .columns()
        .iter()
        .all(|column| needed.contains(column))
    {
        // Nothing to read late.
        return Ok(None);
    }

    let row_id = || datafusion_functions::core::expr_fn::file_row_index().alias(ROW_ID);

    // The n winning row ids.
    let top = LogicalPlanBuilder::from(sort.input.as_ref().clone())
        .project(
            sort_columns
                .iter()
                .cloned()
                .map(Expr::Column)
                .chain([row_id()]),
        )?
        .sort_with_limit(sort.expr.clone(), Some(fetch))?
        .project([col(ROW_ID)])?
        .alias(TOP)?
        .build()?;

    // All the output columns, of the rows with those row ids.
    let full_scan = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        Arc::clone(&scan.source),
        None,
    )?
    .build()?;
    let full = LogicalPlanBuilder::from(full_scan)
        .project(
            output
                .columns()
                .into_iter()
                .map(Expr::Column)
                .chain([row_id()]),
        )?
        .alias(FULL)?
        .build()?;

    let full_column = |name: &str| Column::new(Some(TableReference::bare(FULL)), name);
    let sort_exprs = sort
        .expr
        .iter()
        .map(|sort_expr| {
            let expr = sort_expr
                .expr
                .clone()
                .transform(|expr| match expr {
                    Expr::Column(column) => {
                        Ok(Transformed::yes(Expr::Column(full_column(&column.name))))
                    }
                    other => Ok(Transformed::no(other)),
                })?
                .data;
            Ok(SortExpr::new(expr, sort_expr.asc, sort_expr.nulls_first))
        })
        .collect::<Result<Vec<_>>>()?;

    let rewritten = LogicalPlanBuilder::from(full)
        .join(
            top,
            JoinType::LeftSemi,
            (
                vec![full_column(ROW_ID)],
                vec![Column::new(Some(TableReference::bare(TOP)), ROW_ID)],
            ),
            None,
        )?
        .sort_with_limit(sort_exprs, Some(fetch))?
        .project(output.iter().map(|(qualifier, field)| {
            Expr::Column(full_column(field.name()))
                .alias_qualified(qualifier.cloned(), field.name())
        }))?
        .build()?;
    Ok(Some(rewritten))
}

/// Whether `scan` reads a listing table of exactly one Parquet file: the file
/// row index identifies a row only within one file.
fn is_single_parquet_file(scan: &TableScan) -> bool {
    let Ok(provider) = source_as_provider(&scan.source) else {
        return false;
    };
    let Some(table) = provider.downcast_ref::<ListingTable>() else {
        return false;
    };
    table
        .options()
        .format
        .downcast_ref::<ParquetFormat>()
        .is_some()
        && table.options().table_partition_cols.is_empty()
        && matches!(table.table_paths().as_slice(), [path] if !path.is_collection())
}
