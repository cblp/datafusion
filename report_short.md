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

DataFusion lacked some general mechanisms that ENUM needs. They were added to the engine itself, so other types can use them too.

Already merged into DataFusion as separate pull requests:

- [#26854](https://github.com/apache/datafusion/pull/26854): the Parquet reader keeps extension types (types with extra metadata, like ENUM or UUID). Before, its default `skip_metadata` setting dropped them.
- [#26856](https://github.com/apache/datafusion/pull/26856): the Parquet scan keeps extension types of columns that it replaces with constants.
- [#26939](https://github.com/apache/datafusion/pull/26939): `MIN` and `MAX` keep the extension type of their argument.
- [#26893](https://github.com/apache/datafusion/pull/26893): `DESCRIBE`, `SHOW COLUMNS` and `information_schema.columns` can show extension type names instead of storage types.

Still only in the branch:

- `CREATE TYPE` statement and a per-session registry of user-defined types. Before, DataFusion had no way to declare a named type in SQL.
- Type names from the registry can be used in column definitions and casts.
- `INSERT ... VALUES` casts values to the extension type of the target column. DataFusion already keeps the column's extension type there, but it cast the values only to the storage type, so text labels were not converted to ENUM values.

These changes address known DataFusion issues, fully or as a draft:

- [#26851](https://github.com/apache/datafusion/issues/26851) (closed by #26854): the Parquet reader drops extension types. This was also one of the items of the open Variant epic [#26517](https://github.com/apache/datafusion/issues/26517).
- [#26855](https://github.com/apache/datafusion/issues/26855), [#26916](https://github.com/apache/datafusion/issues/26916) and [#26866](https://github.com/apache/datafusion/issues/26866): closed by #26856, #26939 and #26893.
- [#12644](https://github.com/apache/datafusion/issues/12644) (open) and [#7923](https://github.com/apache/datafusion/issues/7923) (closed in favour of #12644): user-defined types. `CREATE TYPE` and the type registry are a first step; only ENUM is supported so far.
- [#24970](https://github.com/apache/datafusion/issues/24970) and [#17425](https://github.com/apache/datafusion/issues/17425) (closed): `INSERT` and `VALUES` dropping extension types. The branch covers the remaining case: values in `INSERT ... VALUES` are converted to the extension type, not only to its storage type.

### Quality

All existing DataFusion tests pass (more than 4,000 tests in the affected crates and 547 SQL test files), new tests cover the feature, and the linter is clean.

Known limitations, planned as follow-ups: ENUM values lose their labels in `CASE ... ELSE` and `COALESCE` ([#24860](https://github.com/apache/datafusion/issues/24860)); `DESCRIBE` shows the storage type `Int16` instead of the ENUM name; ENUM columns are not supported in CSV external tables; there is no `DROP TYPE` or `ALTER TYPE` yet.
