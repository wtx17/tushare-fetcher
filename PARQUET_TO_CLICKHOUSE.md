# 将 Parquet 目录同步到 ClickHouse

`sync_parquet_to_clickhouse.py` 递归读取指定目录内全部 `*.parquet`，将其完整镜像到指定的 `数据库.表名`。每次成功运行后，目标表的行值和重复次数与此次归档快照一致。

## daily_basic

在远端 `qdata` 环境中运行：

```bash
cd /home/jovyan/work/pkgs/tushare-fetcher
python sync_parquet_to_clickhouse.py \
  data/daily_basic zhangruiqi.daily_basic \
  --env-file /home/jovyan/work/vars/.env \
  --username-env ZRQ --password-env PSWD_ZRQ \
  --cluster cb_cluster --order-by trade_date ts_code
```

更换前两个位置参数即可导入其他目录和目标表；排序列也应随新数据调整，省略 `--order-by` 时使用 `ORDER BY tuple()`。连接默认读取 `MINGHU_CLICKHOUSE_HOST`、`MINGHU_CLICKHOUSE_PORT`，也可用 `--host`、`--port`、`--secure` 指定。凭据只从环境变量或指定 `.env` 读取。

依赖：`clickhouse-connect`、`pyarrow`、`duckdb`、`python-dotenv`。当前实现使用 Linux/macOS 的 `fcntl` 文件锁，Python 3.11 及以上；远端 `qdata` 已具备依赖。

## 表结构与幂等性

- 先在集群各节点创建本轮独立的 `MergeTree` 本地表，再创建引用它们的 `Distributed` 表。全部数据经分布式入口同步写入，最终业务入口固定为指定的 `数据库.表名`。
- 物理表名形如 `daily_basic__data_<run_id>`，暂存分布式表名为 `daily_basic__loading_<run_id>`。这使发布期间旧、新快照各自完整；不要手工修改或向这些镜像表写数据。
- 每次导入完整快照，校验通过后用 `RENAME`（首次）或 `EXCHANGE`（后续）发布分布式入口，再清理旧快照。重复执行不会追加重复行，源文件的增加、修改、删除均会反映到目标表。
- 保留原始列、NULL、浮点精度及重复行。日期字符串仍为字符串，不做去重、PIT 或前向填充。文件必须具有相同 schema；暂不支持嵌套类型，遇到不支持的类型直接报错。
- 写入后回读全部列，与源 Parquet 做全行、多重集比较；同值重复行的数量也必须相等。缺少或多出任意一行均不发布。
- 目标库须已存在且在所有节点使用 `Atomic`；账号须有目标库建表、读写、交换、删表以及 `ON CLUSTER` 权限。集群须保证普通 `MergeTree` 本地表可被其 `Distributed` 表一致读写；本脚本不配置复制拓扑。
- 表名切换在每个节点上原子执行，集群各节点完成切换的时刻可能不同。验证或加载失败时保留原有业务表；若发布请求断线，报告标记 `publish_unconfirmed` 并保留相关表供核查，不盲目重复交换。

## 文件一致性与报告

导入期间共享 tushare-fetcher 的 `data/.sync.lock`；发布前再次检查文件列表及每个文件的 SHA-256。普通目录默认使用该目录的 `.sync.lock`，也可用 `--lock-root` 指定包含源目录的写入锁目录。同一机器上对同一集群目标表的同步也互斥；归档的其他写入程序应遵守同一个锁。

没有任何 Parquet 文件的目录会报错，避免路径写错造成清表。存在 schema 正确但全部为零行的 Parquet 文件时，空快照可以正常同步。

默认报告为归档根目录下的 `_clickhouse_sync/<数据库>.<表名>.json`；本例为 `data/_clickhouse_sync/zhangruiqi.daily_basic.json`，可用 `--report` 改写位置。报告包含源文件数、行数、schema、归档摘要、物理表名、全行比较结果及运行状态。

校验会将数据库内容流式写入临时 Parquet，并在需要时落盘处理。可用 `--temp-dir` 选择有足够空间的目录，用 `--memory-limit` 调整校验内存（默认 `1GB`）；`--batch-rows` 控制写入批次（默认 `100000`）。

本地测试：

```bash
python -m pytest -q tests/test_sync_parquet_to_clickhouse.py
```
