# ENUM-тип в стиле PostgreSQL для DataFusion

Ветка: `enum`, перебазирована на `main` от 4 ноября 2026 (ee55effcc).

## Задача

Добавить пользовательский тип ENUM, совместимый с PostgreSQL:

- объявление через `CREATE TYPE ... AS ENUM (...)`;
- использование в колонках таблиц, литералах и кастах;
- сравнение и сортировка в порядке объявления меток, а не по алфавиту;
- хранение в Parquet, чтобы тип восстанавливался при чтении файла без повторного `CREATE TYPE`.

## Пример

```sql
CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy');
CREATE TABLE person (name VARCHAR, current_mood mood);
INSERT INTO person VALUES ('Moe', 'happy'), ('Larry', 'sad');

SELECT * FROM person WHERE current_mood > 'sad' ORDER BY current_mood;  -- порядок объявления
SELECT 'ok'::mood, current_mood::text FROM person;

COPY person TO 'p.parquet' STORED AS PARQUET;
-- в другой сессии CREATE TYPE не нужен, определение типа хранится в файле:
SELECT * FROM 'p.parquet' WHERE current_mood = 'happy';
```

## Устройство

- **Хранение.** Значение хранится как код `Int16`. Список меток и имя типа
  лежат в метаданных поля Arrow (`ARROW:extension:name` /
  `ARROW:extension:metadata`, JSON). Так тип переживает запись в Parquet и IPC
  без изменений в форматах.
- **Сравнения.** Строковый литерал переводится в код на этапе планирования
  (`Filter: current_mood = Int16(2)`). Сравнение выполняется по целым числам,
  а pruning по статистике Parquet продолжает работать.
- **Ошибки как в Postgres.** Неизвестная метка даёт
  `invalid input value for enum mood: "angry"`.
- **Вывод.** Метки (а не коды) показываются в `DataFrame::to_string`,
  datafusion-cli и sqllogictest.

## Что изменено

### Новые файлы

- [enum_type.rs](datafusion/common/src/types/enum_type.rs): `DFEnum` —
  чтение и запись метаданных поля, перевод меток в коды и обратно,
  форматирование при выводе. В `datafusion-common` добавлена зависимость
  `serde_json` (она уже собиралась транзитивно).
- [enum_coercion.rs](datafusion/optimizer/src/analyzer/enum_coercion.rs):
  правила приведения типов для ENUM, вызываются из `TypeCoercionRewriter`.
  Обрабатывают сравнения, `IN`, `BETWEEN`, `CAST`/`TRY_CAST`; запрещают
  арифметику, сравнение разных ENUM-типов и касты в числа. Узлы `AND`/`OR`
  сразу отсекаются при проверке на ENUM, без вычисления типов операндов:
  операнды логических операторов не могут быть ENUM, а лишнее вычисление типа
  на каждом таком узле удвоило бы стоимость анализа длинных цепочек `AND`/`OR`.
  Сравнения внутри цепочки проверяются отдельно, как самостоятельные узлы.
- [enum_type.slt](datafusion/sqllogictest/test_files/enum_type.slt):
  sqllogictest-сценарии (DDL, вставка, фильтры, сортировка, касты, ошибки,
  запись и чтение Parquet).

### DDL и реестр типов

- Новый вариант `DdlStatement::CreateType` ([ddl.rs](datafusion/expr/src/logical_plan/ddl.rs),
  обход в [tree_node.rs](datafusion/expr/src/logical_plan/tree_node.rs)).
- Разбор `CREATE TYPE` в [statement.rs](datafusion/sql/src/statement.rs).
- Реестр пользовательских типов хранится в `SessionState`
  ([session_state.rs](datafusion/core/src/execution/session_state.rs)),
  выполнение DDL — в [context/mod.rs](datafusion/core/src/execution/context/mod.rs).
- Планировщик получает тип через новый метод
  `ContextProvider::get_user_defined_type` ([planner.rs](datafusion/expr/src/planner.rs)),
  имя типа в колонках и кастах разрешается в [sql/src/planner.rs](datafusion/sql/src/planner.rs).
- В proto `CreateType` явно помечен как несериализуемый
  ([proto/src/logical_plan/mod.rs](datafusion/proto/src/logical_plan/mod.rs)).

### Правка существующего поведения

**`VALUES` при `INSERT`** ([builder.rs](datafusion/expr/src/logical_plan/builder.rs)).
Метаданные целевой таблицы в узле `VALUES` сохраняются уже в `main`
([#21714](https://github.com/apache/datafusion/pull/21714)), но значения
приводились только к типу хранения: `CAST('happy' AS Int16)`. Строка не
превращалась в код ENUM, и `INSERT` падал. Теперь для полей с extension-типом
значения приводятся к самому extension-типу (с метаданными), и это приведение
обрабатывает правило анализатора для ENUM.

### Изменения, уже влитые в `main`

Часть общих правок, нужных для ENUM, оформлена отдельными PR и уже влита.
В ветке их больше нет:

- [#26854](https://github.com/apache/datafusion/pull/26854): при
  `skip_metadata = true` (по умолчанию) чтение Parquet сохраняет ключи
  `ARROW:extension:*`, поскольку это часть типа поля. Раньше эта правка была
  в ветке.
- [#26856](https://github.com/apache/datafusion/pull/26856): Parquet-скан
  сохраняет метаданные полей у колонок, которые он заменяет литералами.
- [#26939](https://github.com/apache/datafusion/pull/26939): `MIN`/`MAX`
  сохраняют метаданные аргумента, в том числе когда значение берётся из
  статистики Parquet. Поэтому `MIN`/`MAX` по ENUM теперь выводят метки.
- [#26893](https://github.com/apache/datafusion/pull/26893): `DESCRIBE`,
  `SHOW COLUMNS` и `information_schema.columns` показывают имя extension-типа,
  если тип зарегистрирован в сессии и реализует `DFExtensionType::display_name`.
  `DFEnum` этот метод пока не реализует.

### Вывод

- [print_format.rs](datafusion-cli/src/print_format.rs) — метки в datafusion-cli.
- [normalize.rs](datafusion/sqllogictest/src/engines/datafusion_engine/normalize.rs) —
  метки в sqllogictest.

## Связь с известными issues

Изменения в общей части движка закрывают или дают черновик решения для
нескольких известных задач:

- [#26851](https://github.com/apache/datafusion/issues/26851) (закрыт
  [#26854](https://github.com/apache/datafusion/pull/26854)). `skip_metadata`
  стирал extension-типы при чтении Parquet. Это был и один из пунктов
  открытого эпика Variant [#26517](https://github.com/apache/datafusion/issues/26517).
- [#26855](https://github.com/apache/datafusion/issues/26855),
  [#26916](https://github.com/apache/datafusion/issues/26916) и
  [#26866](https://github.com/apache/datafusion/issues/26866) (закрыты
  #26856, #26939 и #26893): потеря метаданных в Parquet-скане и в `MAX`/`MIN`,
  имена extension-типов в `DESCRIBE` и `information_schema`.
- [#12644](https://github.com/apache/datafusion/issues/12644) (открыт, общий
  issue про extension-типы и пользовательские типы). `CREATE TYPE`, реестр типов
  в сессии и разрешение имён типов в планировщике — первый шаг к
  пользовательским типам на уровне SQL. Пока поддерживается только ENUM.
- [#7923](https://github.com/apache/datafusion/issues/7923) (закрыт в пользу
  #12644). Предложение по пользовательским типам с синтаксисом
  `CREATE TYPE <name> AS <representation>`. Наша реализация разбирает тот же
  оператор, только для варианта `AS ENUM`.
- [#21626](https://github.com/apache/datafusion/issues/21626) (закрыт
  [#21714](https://github.com/apache/datafusion/pull/21714)). Встроенные имена
  `UUID` и `JSON` теперь отображаются на канонические extension-типы в
  `convert_data_type_to_field`. Наш `ContextProvider::get_user_defined_type`
  возвращает `Field` с метаданными для имени, объявленного через `CREATE TYPE`,
  и вызывается в том же месте для остальных имён.
- [#20755](https://github.com/apache/datafusion/issues/20755) (открыт).
  Подключаемые правила для кастов к extension-типам и из них. Касты ENUM у нас
  сделаны правилом анализатора, то есть обходным путём, описанным в самом
  issue. Общего механизма мы не добавили.
- [#24970](https://github.com/apache/datafusion/issues/24970) и
  [#17425](https://github.com/apache/datafusion/issues/17425) (оба закрыты).
  Потеря extension-метаданных при `INSERT` и в литералах `VALUES`. Наша правка
  `VALUES` закрывает оставшийся случай: значения в `INSERT ... VALUES` сразу
  приводятся к extension-типу целевой колонки, а не к его типу хранения.

## Ограничения MVP

- Метаданные ENUM теряются в `CASE` с веткой `ELSE` и в `COALESCE`/`NVL`
  (известная проблема, [#24860](https://github.com/apache/datafusion/issues/24860)).
  Значения верные, но выводятся числами. `MIN`/`MAX`, `UNION ALL` и `CASE` без
  `ELSE` метки сохраняют.
- `DESCRIBE` и `information_schema.columns` показывают тип хранения `Int16`
  вместо имени ENUM-типа: `DFEnum` не реализует `display_name`.
- `CREATE EXTERNAL TABLE (m mood) STORED AS CSV` не работает: строки из CSV
  не превращаются в коды.
- Нет `DROP TYPE` и `ALTER TYPE ... ADD VALUE`.
- Имена типов только без схемы.
- Типы не переживают `SessionStateBuilder::new_from_existing`.
- Внутренние функции `text_to_enum` и `enum_to_text` не сериализуются в proto.
- `TRY_CAST` к ENUM ведёт себя как `CAST`: неизвестная метка даёт ошибку,
  а не `NULL`.
- Сравнение ENUM с текстовой колонкой (не литералом) разрешено и выполняется
  через перевод в коды во время выполнения. Postgres в этом случае выдаёт
  ошибку.
- Чтение Parquet из нескольких файлов (проверено вручную):
  - файлы с одним и тем же ENUM-типом читаются вместе, метки и сравнения
    работают;
  - если у колонки в разных файлах разные ENUM-типы, объединение схем
    завершается ошибкой `conflicting metadata ... ARROW:extension:metadata`;
  - если в одном файле колонка ENUM, а в другом обычный `Int16`, схемы
    объединяются в ENUM, и числа из второго файла молча выводятся как метки.

## Дальнейшие планы

1. Сохранять метаданные полей в `CASE ... ELSE` и `COALESCE`
   (зависит от [#24860](https://github.com/apache/datafusion/issues/24860)
   или отдельного исправления).
2. Реализовать `display_name` для `DFEnum`, чтобы `DESCRIBE` показывал имя
   ENUM-типа.
3. Поддержать ENUM-колонки во внешних CSV/JSON-таблицах: перевод строк в коды
   при чтении.
4. `DROP TYPE`, `ALTER TYPE ... ADD VALUE`, имена типов со схемой.
5. Переносить реестр типов в `SessionStateBuilder::new_from_existing`.
6. Сериализация `text_to_enum`/`enum_to_text` и `CreateType` в proto.
7. `TRY_CAST` с возвратом `NULL`; запрет сравнения ENUM с текстовой колонкой,
   как в Postgres.
