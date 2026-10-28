# ENUM-тип в стиле PostgreSQL для DataFusion

Ветка: `enum`.

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

### Две правки существующего поведения

Без них MVP не работал:

1. **`VALUES` при `INSERT`** ([builder.rs](datafusion/expr/src/logical_plan/builder.rs)).
   Раньше строки заранее приводились к `Int16` по схеме таблицы, и метаданные
   терялись. Теперь для полей с extension-типом `VALUES` приводит значения сразу
   к этому типу (с метаданными). Побочный эффект: в одном снапшоте
   [params.rs](datafusion/sql/tests/cases/params.rs) из плана для UUID пропал
   лишний `CAST`.
2. **Чтение Parquet** ([file_format.rs](datafusion/datasource-parquet/src/file_format.rs)).
   Опция `skip_metadata` (включена по умолчанию) стирала все метаданные полей.
   Теперь ключи `ARROW:extension:*` сохраняются, поскольку это часть типа поля.

### Вывод

- [print_format.rs](datafusion-cli/src/print_format.rs) — метки в datafusion-cli.
- [normalize.rs](datafusion/sqllogictest/src/engines/datafusion_engine/normalize.rs) —
  метки в sqllogictest.

## Связь с известными issues

Изменения в общей части движка частично закрывают или дают черновик решения
для нескольких известных задач:

- [#26517](https://github.com/apache/datafusion/issues/26517) (открыт, эпик
  Variant). Один из пунктов эпика: `clear_metadata` при `skip_metadata = true`
  стирает `ARROW:extension:*`, поэтому Variant теряет тип при чтении
  Parquet-файлов. В эпике говорится о таблицах из нескольких файлов, но на деле
  тип теряется при любом их числе: потеря воспроизведена на одном файле в
  datafusion 55.0.0. Предложенное там исправление («сохранять ключи `ARROW:extension:*`,
  остальное стирать: это идентичность типа, а не пометка writer'а») совпадает с
  нашей правкой и решает проблему полностью. `clear_metadata` обрабатывает
  только поля верхнего уровня, а метаданные вложенных полей (внутри
  `Struct`/`List`) никогда не стирала, так что вложенные extension-типы
  сохранялись и раньше. Отдельного issue на этот пункт нет, текст для него
  подготовлен.
- [#12644](https://github.com/apache/datafusion/issues/12644) (открыт, общий
  issue про extension-типы и пользовательские типы). `CREATE TYPE`, реестр типов
  в сессии и разрешение имён типов в планировщике — первый шаг к
  пользовательским типам на уровне SQL. Пока поддерживается только ENUM.
- [#7923](https://github.com/apache/datafusion/issues/7923) (закрыт в пользу
  #12644). Предложение по пользовательским типам с синтаксисом
  `CREATE TYPE <name> AS <representation>`. Наша реализация разбирает тот же
  оператор, только для варианта `AS ENUM`.
- [#21626](https://github.com/apache/datafusion/issues/21626) (открыт).
  Отображение SQL-типов (`UUID`, `JSON`) на канонические extension-типы через
  возврат `Field` вместо `DataType` из планировщика. Наш
  `ContextProvider::get_user_defined_type` уже возвращает `Field` с
  метаданными для имени типа. Тот же путь можно использовать и для встроенных
  имён.
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

- Метаданные ENUM теряются в `CASE`/`COALESCE`, `MIN`/`MAX` и `UNION`
  (известная проблема, [#24860](https://github.com/apache/datafusion/issues/24860)). Значения верные, но выводятся числами.
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
- `skip_metadata` теперь сохраняет ключи `ARROW:extension:*` для **всех**
  extension-типов, а не только для ENUM. Это изменение поведения для
  существующих пользователей:
  - колонки других extension-типов (например, `arrow.uuid` или GeoArrow) из
    Parquet раньше читались как обычный `FixedSizeBinary`/`Binary`, а теперь
    приходят с extension-типом, что может изменить планы и приведение типов;
  - может вернуться проблема, от которой защищает `skip_metadata`: при чтении
    нескольких файлов, где у колонки в одном файле есть extension-тип, а в
    другом нет (или у ENUM разные списки меток), объединение схем, вероятно,
    завершится ошибкой. Это не проверялось.

  Альтернатива — не менять `skip_metadata` и поддерживать ENUM только при
  `skip_metadata = false` — хуже:
  - опция по умолчанию включена, поэтому ENUM не переживал бы чтение Parquet
    с настройками по умолчанию;
  - ошибки бы не было: колонка молча читалась бы как `Int16`, пользователь
    видел бы коды вместо меток, сравнение с `'happy'` падало бы на приведении
    строки к числу, а сравнение с `'2'` молча срабатывало бы по коду;
  - выключение опции возвращает все метаданные, включая метаданные уровня
    схемы, а с ними и конфликты при чтении нескольких файлов.

  Более узкий вариант: при `skip_metadata` сохранять метаданные только у полей
  с `ARROW:extension:name = "datafusion.enum"`. Тогда для существующих данных
  ничего не меняется, а вопрос о сохранении extension-типов в целом можно
  вынести в отдельное обсуждение в upstream. Довод в пользу текущего, широкого
  варианта: в эпике [#26517](https://github.com/apache/datafusion/issues/26517)
  предлагают именно его.

## Дальнейшие планы

1. Сохранять метаданные полей в `CASE`/`COALESCE`, агрегатах и `UNION`
   (зависит от [#24860](https://github.com/apache/datafusion/issues/24860)
   или отдельного исправления).
2. Поддержать ENUM-колонки во внешних CSV/JSON-таблицах: перевод строк в коды
   при чтении.
3. `DROP TYPE`, `ALTER TYPE ... ADD VALUE`, имена типов со схемой.
4. Переносить реестр типов в `SessionStateBuilder::new_from_existing`.
5. Сериализация `text_to_enum`/`enum_to_text` и `CreateType` в proto.
6. `TRY_CAST` с возвратом `NULL`; запрет сравнения ENUM с текстовой колонкой,
   как в Postgres.
