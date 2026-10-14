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

//! Late materialization of a TopK over a single Parquet file.
//!
//! For a plan such as
//!
//! ```text
//! SortPreservingMergeExec: [EventTime ASC], fetch=10
//!   SortExec: TopK(fetch=10), expr=[EventTime ASC]
//!     FilterExec: URL LIKE '%google%'
//!       DataSourceExec: projection=[<105 columns>]
//! ```
//!
//! [`ParquetLateMaterialization`] reads only the sort and filter columns plus
//! the file row index to find the winning rows, and
//! [`ParquetFetchByRowIdExec`] then decodes the remaining columns for just
//! those rows:
//!
//! ```text
//! ParquetFetchByRowIdExec: fetch=10
//!   SortPreservingMergeExec: [EventTime ASC], fetch=10
//!     SortExec: TopK(fetch=10), expr=[EventTime ASC]
//!       FilterExec: URL LIKE '%google%'
//!         DataSourceExec: projection=[EventTime, URL, file_row_index()]
//! ```

use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{Result, internal_err};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion_physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
    ScalarFunctionExpr,
};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, common,
    validate_child_count,
};
use datafusion_session::PhysicalOptimizerRule;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};

use crate::metadata::DFParquetMetadata;
use crate::source::ParquetSource;
use crate::{ParquetAccessPlan, RowGroupAccess};

/// Name of the row index column added to the narrow and the fetch scans.
const ROW_INDEX_COLUMN: &str = "__late_materialization_row_index";

/// Physical optimizer rule that rewrites a TopK over a single Parquet file
/// into a narrow TopK followed by [`ParquetFetchByRowIdExec`].
///
/// Enabled by `datafusion.optimizer.enable_parquet_late_materialization`.
/// It must run before `FilterPushdown(Post)`, so that the TopK dynamic filter
/// is pushed into the narrow scan.
#[derive(Debug, Default)]
pub struct ParquetLateMaterialization {}

impl ParquetLateMaterialization {
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for ParquetLateMaterialization {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_parquet_late_materialization {
            return Ok(plan);
        }
        let max_rows = config.optimizer.late_materialization_max_rows;
        plan.transform_down(|node| match try_rewrite_topk(&node, max_rows)? {
            // The rewritten subtree contains a TopK over a Parquet scan again:
            // do not visit it, or the rule would rewrite it forever.
            Some(rewritten) => {
                Ok(Transformed::new(rewritten, true, TreeNodeRecursion::Jump))
            }
            None => Ok(Transformed::no(node)),
        })
        .data()
    }

    fn name(&self) -> &str {
        "ParquetLateMaterialization"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The parts of a matched `[SortPreservingMergeExec] -> SortExec(TopK) ->
/// (FilterExec | RepartitionExec(RoundRobinBatch))* -> DataSourceExec(parquet)`
/// chain.
struct TopKChain<'a> {
    merge: Option<&'a SortPreservingMergeExec>,
    sort: &'a SortExec,
    /// Nodes between the TopK and the scan, top to bottom.
    between: Vec<Between<'a>>,
    scan: &'a DataSourceExec,
}

enum Between<'a> {
    Filter(&'a FilterExec),
    /// A round-robin repartition, which does not reference any column.
    Repartition(&'a Arc<dyn ExecutionPlan>),
}

fn match_topk_chain(node: &Arc<dyn ExecutionPlan>) -> Option<TopKChain<'_>> {
    let (merge, sort_node) = match node.downcast_ref::<SortPreservingMergeExec>() {
        Some(merge) => (Some(merge), merge.input()),
        None => (None, node),
    };
    let sort = sort_node.downcast_ref::<SortExec>()?;
    sort.fetch()?;
    if merge.is_none() && sort.preserve_partitioning() {
        // A partitioned TopK without a merge above it is not a complete TopK.
        return None;
    }

    let mut between = vec![];
    let mut current = sort.input();
    loop {
        if let Some(filter) = current.downcast_ref::<FilterExec>() {
            if filter.projection().is_some() || filter.fetch().is_some() {
                return None;
            }
            between.push(Between::Filter(filter));
            current = filter.input();
        } else if let Some(repartition) = current.downcast_ref::<RepartitionExec>() {
            if !matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_)) {
                return None;
            }
            between.push(Between::Repartition(current));
            current = repartition.input();
        } else {
            let scan = current.downcast_ref::<DataSourceExec>()?;
            return Some(TopKChain {
                merge,
                sort,
                between,
                scan,
            });
        }
    }
}

fn try_rewrite_topk(
    node: &Arc<dyn ExecutionPlan>,
    max_rows: usize,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(chain) = match_topk_chain(node) else {
        return Ok(None);
    };
    let fetch = chain
        .merge
        .and_then(|m| m.fetch())
        .or_else(|| chain.sort.fetch());
    let Some(fetch) = fetch.filter(|fetch| *fetch <= max_rows) else {
        return Ok(None);
    };
    let Some((config, _)) = chain.scan.downcast_to_file_source::<ParquetSource>() else {
        return Ok(None);
    };
    // The file row index is only unique within one file.
    let paths: BTreeSet<_> = config
        .file_groups
        .iter()
        .flat_map(|group| group.iter())
        .map(|file| file.object_meta.location.clone())
        .collect();
    if paths.len() != 1 || !config.table_partition_cols().is_empty() {
        return Ok(None);
    }

    // Columns the narrow pipeline needs, in scan output order.
    let scan_schema = chain.scan.schema();
    if scan_schema.index_of(ROW_INDEX_COLUMN).is_ok() {
        // Already rewritten.
        return Ok(None);
    }
    let mut needed = BTreeSet::new();
    for sort_expr in chain.sort.expr().iter() {
        needed.extend(collect_columns(&sort_expr.expr).iter().map(|c| c.index()));
    }
    for node in &chain.between {
        if let Between::Filter(filter) = node {
            needed.extend(
                collect_columns(filter.predicate())
                    .iter()
                    .map(|c| c.index()),
            );
        }
    }
    if needed.len() >= scan_schema.fields().len() {
        // Nothing left to materialize late.
        return Ok(None);
    }

    let narrow_exprs = needed.iter().map(|&index| {
        let name = scan_schema.field(index).name();
        ProjectionExpr::new(Arc::new(Column::new(name, index)), name)
    });
    let Some(narrow_scan) = scan_with_row_index(chain.scan, narrow_exprs)? else {
        return Ok(None);
    };
    let all_exprs = scan_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            ProjectionExpr::new(Arc::new(Column::new(field.name(), index)), field.name())
        });
    let Some(fetch_scan) = scan_with_row_index(chain.scan, all_exprs)? else {
        return Ok(None);
    };
    let Some(fetch_config) = fetch_scan
        .downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
        .cloned()
    else {
        return Ok(None);
    };

    // Rebuild the nodes between the TopK and the scan, and the TopK, on top of
    // the narrow scan.
    let mut narrow: Arc<dyn ExecutionPlan> = narrow_scan;
    for node in chain.between.iter().rev() {
        narrow = match node {
            Between::Filter(filter) => {
                let predicate = reassign_expr_columns(
                    Arc::clone(filter.predicate()),
                    &narrow.schema(),
                )?;
                Arc::new(FilterExec::try_new(predicate, narrow)?)
            }
            Between::Repartition(repartition) => Arc::clone(*repartition)
                .replace_children(
                    vec![narrow],
                    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
                )?,
        };
    }
    let narrow_ordering = reassign_ordering(chain.sort.expr(), &narrow.schema())?;
    narrow = Arc::new(
        SortExec::new(narrow_ordering.clone(), narrow)
            .with_fetch(chain.sort.fetch())
            .with_preserve_partitioning(chain.sort.preserve_partitioning()),
    );
    if let Some(merge) = chain.merge {
        narrow = Arc::new(
            SortPreservingMergeExec::new(narrow_ordering, narrow)
                .with_fetch(merge.fetch()),
        );
    }

    let row_index = narrow.schema().index_of(ROW_INDEX_COLUMN)?;
    Ok(Some(Arc::new(ParquetFetchByRowIdExec::try_new(
        narrow,
        row_index,
        fetch_config,
        node.schema(),
        chain.sort.expr().clone(),
        fetch,
    )?)))
}

/// Push `exprs` plus `file_row_index()` into `scan`.
fn scan_with_row_index(
    scan: &DataSourceExec,
    exprs: impl Iterator<Item = ProjectionExpr>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let scan: Arc<dyn ExecutionPlan> = Arc::new(scan.clone());
    let row_index = Arc::new(ScalarFunctionExpr::try_new(
        datafusion_functions::core::file_row_index(),
        vec![],
        &scan.schema(),
        Arc::new(ConfigOptions::default()),
    )?);
    let exprs = exprs.chain(std::iter::once(ProjectionExpr::new(
        row_index,
        ROW_INDEX_COLUMN,
    )));
    let projection = ProjectionExec::try_new(exprs, Arc::clone(&scan))?;
    scan.try_swapping_with_projection(&projection)
}

fn reassign_ordering(
    ordering: &LexOrdering,
    schema: &arrow::datatypes::Schema,
) -> Result<LexOrdering> {
    let exprs = ordering
        .iter()
        .map(|sort_expr| {
            Ok(PhysicalSortExpr::new(
                reassign_expr_columns(Arc::clone(&sort_expr.expr), schema)?,
                sort_expr.options,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    match LexOrdering::new(exprs) {
        Some(ordering) => Ok(ordering),
        None => internal_err!("TopK ordering must not be empty"),
    }
}

/// Reads the rows of a single Parquet file whose file row indexes are produced
/// by `input`, in the order `input` produces them.
///
/// `input` yields at most `fetch` rows and has a file row index column. Its
/// other columns are discarded: every output column is read from the file.
#[derive(Debug, Clone)]
pub struct ParquetFetchByRowIdExec {
    input: Arc<dyn ExecutionPlan>,
    /// Index of the file row index column in `input`'s schema.
    row_index: usize,
    /// Scan of all output columns plus the file row index (last column). Its
    /// file groups are replaced at execution time.
    fetch_config: FileScanConfig,
    /// Output schema: the fetch scan's schema without the row index column.
    schema: SchemaRef,
    fetch: usize,
    metrics: ExecutionPlanMetricsSet,
    cache: Arc<PlanProperties>,
}

impl ParquetFetchByRowIdExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        row_index: usize,
        fetch_config: FileScanConfig,
        schema: SchemaRef,
        ordering: LexOrdering,
        fetch: usize,
    ) -> Result<Self> {
        let eq_properties =
            EquivalenceProperties::new_with_orderings(Arc::clone(&schema), [ordering]);
        let cache = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            row_index,
            fetch_config,
            schema,
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
        })
    }
}

impl DisplayAs for ParquetFetchByRowIdExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ParquetFetchByRowIdExec: fetch={}, columns={}",
                    self.fetch,
                    self.schema.fields().len()
                )
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "fetch={}", self.fetch)?;
                write!(f, "columns={}", self.schema.fields().len())
            }
        }
    }
}

impl ExecutionPlan for ParquetFetchByRowIdExec {
    fn name(&self) -> &'static str {
        "ParquetFetchByRowIdExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        validate_child_count!(self, children);
        // The output properties do not depend on the child.
        Ok(Arc::new(Self {
            input: children.swap_remove(0),
            metrics: ExecutionPlanMetricsSet::new(),
            ..Self::clone(&self)
        }))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "ParquetFetchByRowIdExec has one partition, got {partition}"
            );
        }
        let input = self.input.execute(0, Arc::clone(&context))?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let fetched_rows = MetricBuilder::new(&self.metrics).counter("fetched_rows", 0);
        let fetch = FetchByRowId {
            row_index: self.row_index,
            fetch_config: self.fetch_config.clone(),
            schema: Arc::clone(&self.schema),
            fetched_rows,
        };
        let stream = futures::stream::once(async move {
            let batch = fetch.run(input, context).await?;
            baseline.record_output(batch.num_rows());
            baseline.done();
            Ok(batch)
        })
        .boxed();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

struct FetchByRowId {
    row_index: usize,
    fetch_config: FileScanConfig,
    schema: SchemaRef,
    fetched_rows: Count,
}

impl FetchByRowId {
    async fn run(
        self,
        input: SendableRecordBatchStream,
        context: Arc<TaskContext>,
    ) -> Result<RecordBatch> {
        let input_schema = input.schema();
        let winners = concat_batches(&input_schema, &common::collect(input).await?)?;
        let Some(row_ids) = winners
            .column(self.row_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .cloned()
        else {
            return internal_err!("file row index must be Int64");
        };
        if row_ids.null_count() > 0 {
            return internal_err!("file row index must not be null");
        }
        if row_ids.is_empty() {
            return Ok(RecordBatch::new_empty(Arc::clone(&self.schema)));
        }

        let file = self.fetch_config.file_groups[0].files()[0].clone();
        let store = context
            .runtime_env()
            .object_store(&self.fetch_config.object_store_url)?;
        let metadata = DFParquetMetadata::new(store.as_ref(), &file.object_meta)
            .with_file_metadata_cache(Some(
                context
                    .runtime_env()
                    .cache_manager
                    .get_file_metadata_cache(),
            ))
            .fetch_metadata()
            .await?;

        let sorted_ids: BTreeSet<u64> =
            row_ids.values().iter().map(|&id| id as u64).collect();
        let access_plan = access_plan_for_rows(&sorted_ids, &metadata)?;
        let file = PartitionedFile::new(
            file.object_meta.location.to_string(),
            file.object_meta.size,
        )
        .with_extension(access_plan);
        let config = FileScanConfigBuilder::from(self.fetch_config)
            .with_file_groups(vec![FileGroup::new(vec![file])])
            .with_limit(None)
            .build();
        let scan = DataSourceExec::from_data_source(config);
        let stream = scan.execute(0, Arc::clone(&context))?;
        let fetched_schema = stream.schema();
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        let fetched = concat_batches(&fetched_schema, &batches)?;
        self.fetched_rows.add(fetched.num_rows());

        // Map every winning row id to its position in the fetched batch, then
        // gather the fetched rows in the order of the winners.
        let fetched_row_index = fetched.num_columns() - 1;
        let Some(fetched_ids) = fetched
            .column(fetched_row_index)
            .as_any()
            .downcast_ref::<Int64Array>()
        else {
            return internal_err!("fetched file row index must be Int64");
        };
        let positions: HashMap<i64, u32> = fetched_ids
            .values()
            .iter()
            .enumerate()
            .map(|(position, &id)| (id, position as u32))
            .collect();
        let indices = row_ids
            .values()
            .iter()
            .map(|id| match positions.get(id) {
                Some(position) => Ok(*position),
                None => internal_err!("row {id} was not fetched"),
            })
            .collect::<Result<UInt32Array>>()?;
        let ordered = take_record_batch(&fetched, &indices)?;
        let columns = ordered.columns()[..fetched_row_index].to_vec();
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

/// Build an access plan that selects exactly `row_ids` (file row indexes).
fn access_plan_for_rows(
    row_ids: &BTreeSet<u64>,
    metadata: &parquet::file::metadata::ParquetMetaData,
) -> Result<ParquetAccessPlan> {
    let row_groups = metadata.row_groups();
    let mut plan = ParquetAccessPlan::new_none(row_groups.len());
    let mut ids = row_ids.iter().copied().peekable();
    let mut row_group_start = 0u64;
    for (index, row_group) in row_groups.iter().enumerate() {
        let row_group_rows = row_group.num_rows() as u64;
        let row_group_end = row_group_start + row_group_rows;
        let mut selectors = vec![];
        let mut position = row_group_start;
        while let Some(id) = ids.next_if(|id| *id < row_group_end) {
            if id > position {
                selectors.push(RowSelector::skip((id - position) as usize));
            }
            selectors.push(RowSelector::select(1));
            position = id + 1;
        }
        if !selectors.is_empty() {
            if row_group_end > position {
                selectors.push(RowSelector::skip((row_group_end - position) as usize));
            }
            plan.set(
                index,
                RowGroupAccess::Selection(RowSelection::from(selectors)),
            );
        }
        row_group_start = row_group_end;
    }
    if let Some(id) = ids.next() {
        return internal_err!(
            "row {id} is beyond the end of the file ({row_group_start} rows)"
        );
    }
    Ok(plan)
}
