# Add a PostgreSQL-style ENUM type to DataFusion

## Task

Support user-defined ENUM types, compatible with PostgreSQL:

- declare them with `CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')`;
- use them in table columns, literals and casts;
- compare and sort values in declaration order, not alphabetically;
- store them in Parquet so that a file can be read back with its ENUM types without declaring them again.

## Results

Branch https://github.com/apache/datafusion/compare/main...cblp:datafusion:enum

The MVP works end to end.

### ENUM type

- ENUM columns in tables, `INSERT`, filters, `ORDER BY`, `IN`, `BETWEEN` and casts to and from text.
- Values are stored as small integers, so comparisons are fast and Parquet statistics still let the engine skip data.
- Invalid values give the same error as PostgreSQL (`invalid input value for enum mood: "angry"`).
- ENUM types survive a round trip through Parquet: another session reads the file with the right type and labels.
- Labels, not codes, appear in query output, the CLI and the test framework.

### General engine changes

DataFusion lacked some general mechanisms that ENUM needs, so they were added to the engine itself. Other types can use them too:

- `CREATE TYPE` statement and a per-session registry of user-defined types. Before, DataFusion had no way to declare a named type in SQL.
- Type names from the registry can be used in column definitions and casts.
- `INSERT ... VALUES` now keeps extension types (types with extra metadata, like ENUM or UUID) of the target table. Before, it reduced them to the underlying storage type.
- The Parquet reader now keeps extension types by default. Before, it dropped them. This also changes how existing files with other extension types (e.g. UUID) are read, so it needs discussion with the DataFusion maintainers.

These changes address known DataFusion issues, fully or as a draft:

- [#26517](https://github.com/apache/datafusion/issues/26517) (open): the Parquet reader drops extension types. Our change implements the fix proposed there.
- [#12644](https://github.com/apache/datafusion/issues/12644) (open) and [#7923](https://github.com/apache/datafusion/issues/7923) (closed in favour of #12644): user-defined types. `CREATE TYPE` and the type registry are a first step; only ENUM is supported so far.
- [#21626](https://github.com/apache/datafusion/issues/21626) (open): mapping SQL type names such as `UUID` to extension types. The new type lookup in the planner can be reused for it.
- [#24970](https://github.com/apache/datafusion/issues/24970) and [#17425](https://github.com/apache/datafusion/issues/17425) (closed): `INSERT` and `VALUES` dropping extension types. Our change covers the remaining case, `INSERT ... VALUES` into an extension-type column.

### Quality

All existing DataFusion tests pass (about 3,000 in the affected crates), new tests cover the feature, and the linter is clean.

Known limitations, planned as follow-ups: ENUM values lose their labels in `CASE`/`COALESCE`, `MIN`/`MAX` and `UNION` ([#24860](https://github.com/apache/datafusion/issues/24860)); ENUM columns are not supported in CSV external tables; there is no `DROP TYPE` or `ALTER TYPE` yet.
