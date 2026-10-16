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

use arrow::array::{
    Array, ArrayRef, BinaryViewArray, Int64Array, RecordBatch, StringViewArray,
    UInt32Array,
};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{Result, internal_err};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file::FileSource;
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
    RecordOutput, Time,
};
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::statistics::{StatisticsArgs, StatisticsContext};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, common,
    validate_child_count,
};
use datafusion_session::{PhysicalOptimizerContext, PhysicalOptimizerRule};
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

impl ParquetLateMaterialization {
    fn optimize_with_statistics(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
        statistics: &StatisticsContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_parquet_late_materialization {
            return Ok(plan);
        }
        let limits = Limits {
            max_rows: config.optimizer.late_materialization_max_rows,
            min_saved_bytes_ratio: config
                .optimizer
                .late_materialization_min_saved_bytes_ratio,
        };
        plan.transform_down(|node| {
            match try_rewrite_topk(&node, &limits, statistics)? {
                // The rewritten subtree contains a TopK over a Parquet scan
                // again: do not visit it, or the rule would rewrite it forever.
                Some(rewritten) => {
                    Ok(Transformed::new(rewritten, true, TreeNodeRecursion::Jump))
                }
                None => Ok(Transformed::no(node)),
            }
        })
        .data()
    }
}

impl PhysicalOptimizerRule for ParquetLateMaterialization {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.optimize_with_statistics(plan, config, &StatisticsContext::new())
    }

    fn optimize_with_context(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        context: &dyn PhysicalOptimizerContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match context.statistics_context() {
            Some(statistics) => {
                self.optimize_with_statistics(plan, context.config_options(), statistics)
            }
            None => self.optimize(plan, context.config_options()),
        }
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

/// When a TopK is worth rewriting.
struct Limits {
    /// Largest `LIMIT n`: `ParquetFetchByRowIdExec` reads up to `n` row groups.
    max_rows: usize,
    /// Smallest share of the scanned bytes in the columns read late.
    min_saved_bytes_ratio: f64,
}

fn try_rewrite_topk(
    node: &Arc<dyn ExecutionPlan>,
    limits: &Limits,
    statistics: &StatisticsContext,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(chain) = match_topk_chain(node) else {
        return Ok(None);
    };
    let fetch = chain
        .merge
        .and_then(|m| m.fetch())
        .or_else(|| chain.sort.fetch());
    let Some(fetch) = fetch.filter(|fetch| *fetch <= limits.max_rows) else {
        return Ok(None);
    };
    let Some((config, source)) = chain.scan.downcast_to_file_source::<ParquetSource>()
    else {
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
    // The narrow scan also reads the columns of its own predicate (filters
    // pushed into the scan), though it does not output them.
    let mut read_early = needed.clone();
    if let Some(predicate) = source.filter() {
        read_early.extend(
            collect_columns(&predicate)
                .iter()
                .filter_map(|column| scan_schema.index_of(column.name()).ok()),
        );
    }
    let saved = saved_bytes_ratio(chain.scan, &read_early, statistics)?;
    if saved < limits.min_saved_bytes_ratio {
        // The columns read late are too small to pay for a second scan.
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

/// Share of the bytes scanned by `scan` that are in columns other than
/// `read_early` (indexes into the scan output), from the scan statistics. Columns
/// without a byte size count as the average size of the others, or all as the
/// same size if none has one.
fn saved_bytes_ratio(
    scan: &DataSourceExec,
    read_early: &BTreeSet<usize>,
    statistics: &StatisticsContext,
) -> Result<f64> {
    let statistics = statistics.compute(scan, &StatisticsArgs::new())?;
    let sizes: Vec<Option<f64>> = statistics
        .column_statistics
        .iter()
        .map(|column| column.byte_size.get_value().map(|size| *size as f64))
        .collect();
    let known: Vec<f64> = sizes.iter().flatten().copied().collect();
    let default_size = if known.is_empty() {
        1.0
    } else {
        known.iter().sum::<f64>() / known.len() as f64
    };
    let sizes: Vec<f64> = sizes
        .into_iter()
        .map(|size| size.unwrap_or(default_size))
        .collect();
    let total: f64 = sizes.iter().sum();
    if total <= 0.0 {
        return Ok(0.0);
    }
    let saved: f64 = sizes
        .iter()
        .enumerate()
        .filter(|(index, _)| !read_early.contains(index))
        .map(|(_, size)| size)
        .sum();
    Ok(saved / total)
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
        let fetch = FetchByRowId {
            row_index: self.row_index,
            fetch_config: self.fetch_config.clone(),
            schema: Arc::clone(&self.schema),
            elapsed_compute: baseline.elapsed_compute().clone(),
            fetch_time: MetricBuilder::new(&self.metrics)
                .subset_time("fetch_time", partition),
            fetched_rows: MetricBuilder::new(&self.metrics)
                .counter("fetched_rows", partition),
            fetch_bytes_scanned: MetricBuilder::new(&self.metrics)
                .bytes_counter("fetch_bytes_scanned", partition),
            io: FetchIoMetrics::new(&self.metrics, partition),
        };
        let stream = futures::stream::once(async move {
            let batch = fetch.run(input, context).await?;
            let batch = batch.record_output(&baseline);
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
    /// Time spent in this operator's own work: gathering the row ids,
    /// planning the fetch and reordering the rows. Excludes waiting for the
    /// input and reading the file.
    elapsed_compute: Time,
    /// Wall-clock time of the fetch, from loading the metadata to reordering
    /// the rows, including the scan that reads them.
    fetch_time: Time,
    fetched_rows: Count,
    /// Bytes the fetch scan actually read, to compare with the estimates.
    fetch_bytes_scanned: Count,
    io: FetchIoMetrics,
}

/// Estimated object store I/O of the fetch, as metrics.
///
/// The fetch can read either only the pages that hold the wanted rows (needs
/// the offset index) or whole column chunks. Pages read fewer bytes, chunks
/// need fewer requests: on a remote store with a high cost per request, chunks
/// can be cheaper. Both are estimated from the Parquet metadata, assuming the
/// requests of one row group are coalesced like `object_store` does, so the
/// choice can be made from data later; the fetch itself always reads pages
/// when the offset index is available.
struct FetchIoMetrics {
    row_groups: Count,
    page_requests: Count,
    page_bytes: Count,
    chunk_requests: Count,
    chunk_bytes: Count,
}

impl FetchIoMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let builder = || MetricBuilder::new(metrics);
        Self {
            row_groups: builder().counter("fetched_row_groups", partition),
            page_requests: builder().counter("estimated_page_requests", partition),
            page_bytes: builder().bytes_counter("estimated_page_bytes", partition),
            chunk_requests: builder().counter("estimated_chunk_requests", partition),
            chunk_bytes: builder().bytes_counter("estimated_chunk_bytes", partition),
        }
    }

    fn record(&self, estimate: &FetchIoEstimate) {
        self.row_groups.add(estimate.row_groups);
        self.page_requests.add(estimate.pages.requests);
        self.page_bytes.add(estimate.pages.bytes);
        self.chunk_requests.add(estimate.chunks.requests);
        self.chunk_bytes.add(estimate.chunks.bytes);
    }
}

impl FetchByRowId {
    async fn run(
        self,
        input: SendableRecordBatchStream,
        context: Arc<TaskContext>,
    ) -> Result<RecordBatch> {
        let input_schema = input.schema();
        let input_batches = common::collect(input).await?;
        let fetch_timer = self.fetch_time.timer();
        let compute_timer = self.elapsed_compute.timer();
        let winners = concat_batches(&input_schema, &input_batches)?;
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
        compute_timer.done();

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

        let compute_timer = self.elapsed_compute.timer();
        let sorted_ids: BTreeSet<u64> =
            row_ids.values().iter().map(|&id| id as u64).collect();
        let access_plan = access_plan_for_rows(&sorted_ids, &metadata)?;
        let fetched_columns: Vec<&str> = self
            .schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        self.io
            .record(&estimate_fetch_io(&sorted_ids, &metadata, &fetched_columns));
        let file = PartitionedFile::new(
            file.object_meta.location.to_string(),
            file.object_meta.size,
        )
        .with_extension(access_plan);
        // The fetch scan is a copy of the narrow scan's source, which would
        // share its metrics: give it its own, so that the narrow scan reports
        // only its own reads and `fetch_bytes_scanned` only the fetch's.
        let Some(source) = self
            .fetch_config
            .file_source()
            .downcast_ref::<ParquetSource>()
        else {
            return internal_err!("fetch scan must read Parquet");
        };
        let mut source = source.clone();
        source.metrics = ExecutionPlanMetricsSet::new();
        let config = FileScanConfigBuilder::from(self.fetch_config)
            .with_source(Arc::new(source))
            .with_file_groups(vec![FileGroup::new(vec![file])])
            .with_limit(None)
            .build();
        compute_timer.done();

        let scan = DataSourceExec::from_data_source(config);
        let stream = scan.execute(0, Arc::clone(&context))?;
        let fetched_schema = stream.schema();
        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        if let Some(bytes) = scan
            .metrics()
            .and_then(|metrics| metrics.sum_by_name("bytes_scanned"))
        {
            self.fetch_bytes_scanned.add(bytes.as_usize());
        }

        let _compute_timer = self.elapsed_compute.timer();
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
        // `take` keeps the view arrays pointing into the buffers of every page
        // read; copy out only the bytes of the selected rows.
        let columns = ordered.columns()[..fetched_row_index]
            .iter()
            .map(compact_view_array)
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
        fetch_timer.done();
        Ok(batch)
    }
}

/// Copy a `Utf8View` / `BinaryView` array into buffers holding only its own
/// values; other arrays are returned as they are.
fn compact_view_array(array: &ArrayRef) -> ArrayRef {
    if let Some(strings) = array.as_any().downcast_ref::<StringViewArray>() {
        Arc::new(strings.gc())
    } else if let Some(binaries) = array.as_any().downcast_ref::<BinaryViewArray>() {
        Arc::new(binaries.gc())
    } else {
        Arc::clone(array)
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

/// `object_store` merges requested ranges closer than this into one request
/// (`OBJECT_STORE_COALESCE_DEFAULT`).
const COALESCE_GAP_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Default, PartialEq)]
struct IoEstimate {
    requests: usize,
    bytes: usize,
}

#[derive(Debug, PartialEq)]
struct FetchIoEstimate {
    row_groups: usize,
    /// Read only the pages that hold the rows (falls back to whole chunks for
    /// columns without an offset index).
    pages: IoEstimate,
    /// Read whole column chunks.
    chunks: IoEstimate,
}

/// Estimate the object store requests and bytes to fetch `row_ids` for the
/// top-level `columns`, per row group, after coalescing.
fn estimate_fetch_io(
    row_ids: &BTreeSet<u64>,
    metadata: &parquet::file::metadata::ParquetMetaData,
    columns: &[&str],
) -> FetchIoEstimate {
    let schema = metadata.file_metadata().schema_descr();
    let leaves: Vec<usize> = (0..schema.num_columns())
        .filter(|&leaf| {
            let column = schema.column(leaf);
            column
                .path()
                .parts()
                .first()
                .is_some_and(|root| columns.contains(&root.as_str()))
        })
        .collect();
    let page_index = metadata.page_index();

    let mut estimate = FetchIoEstimate {
        row_groups: 0,
        pages: IoEstimate::default(),
        chunks: IoEstimate::default(),
    };
    let mut ids = row_ids.iter().copied().peekable();
    let mut row_group_start = 0u64;
    for (row_group_index, row_group) in metadata.row_groups().iter().enumerate() {
        let row_group_end = row_group_start + row_group.num_rows() as u64;
        let mut rows = vec![];
        while let Some(id) = ids.next_if(|id| *id < row_group_end) {
            rows.push((id - row_group_start) as i64);
        }
        row_group_start = row_group_end;
        if rows.is_empty() {
            continue;
        }
        estimate.row_groups += 1;

        let mut chunk_ranges = vec![];
        let mut page_ranges = vec![];
        for &leaf in &leaves {
            let column = row_group.column(leaf);
            let (start, length) = column.byte_range();
            chunk_ranges.push((start, start + length));
            let locations = page_index
                .and_then(|index| index.page_locations(row_group_index, leaf))
                .filter(|locations| !locations.is_empty());
            let Some(locations) = locations else {
                page_ranges.push((start, start + length));
                continue;
            };
            // The dictionary page precedes the first data page.
            let first_data_page = locations[0].offset as u64;
            if first_data_page > start {
                page_ranges.push((start, first_data_page));
            }
            let mut last_page = None;
            for &row in &rows {
                let page = locations
                    .partition_point(|location| location.first_row_index <= row)
                    .saturating_sub(1);
                if last_page != Some(page) {
                    let location = &locations[page];
                    let offset = location.offset as u64;
                    page_ranges
                        .push((offset, offset + location.compressed_page_size as u64));
                    last_page = Some(page);
                }
            }
        }
        add_coalesced(&mut estimate.chunks, chunk_ranges);
        add_coalesced(&mut estimate.pages, page_ranges);
    }
    estimate
}

/// Add the requests and bytes of `ranges` coalesced like `object_store` does.
fn add_coalesced(estimate: &mut IoEstimate, mut ranges: Vec<(u64, u64)>) {
    ranges.sort_unstable();
    let mut current: Option<(u64, u64)> = None;
    for (start, end) in ranges {
        current = match current {
            Some((current_start, current_end))
                if start <= current_end + COALESCE_GAP_BYTES =>
            {
                Some((current_start, current_end.max(end)))
            }
            Some((current_start, current_end)) => {
                estimate.requests += 1;
                estimate.bytes += (current_end - current_start) as usize;
                Some((start, end))
            }
            None => Some((start, end)),
        };
    }
    if let Some((start, end)) = current {
        estimate.requests += 1;
        estimate.bytes += (end - start) as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_merges_close_ranges() {
        let mut estimate = IoEstimate::default();
        let mib = COALESCE_GAP_BYTES;
        add_coalesced(
            &mut estimate,
            vec![
                (10 * mib, 11 * mib),
                (0, 100),
                (200, mib + 300),
                (5 * mib, 6 * mib),
            ],
        );
        // [0, 100) and [200, mib + 300) merge (gap 100); the other two stay
        // apart (gaps over 1 MiB).
        assert_eq!(
            estimate,
            IoEstimate {
                requests: 3,
                bytes: (mib + 300 + mib + mib) as usize,
            }
        );
    }

    #[test]
    fn coalesce_empty() {
        let mut estimate = IoEstimate::default();
        add_coalesced(&mut estimate, vec![]);
        assert_eq!(estimate, IoEstimate::default());
    }
}
