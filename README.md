# tushare-sync

从 Tushare API 拉取财务、股东、行业分类等 A 股数据，本地存储为分区 Parquet 文件。

## 环境准备

```bash
# 使用 quant_data conda 环境
conda activate quant_data

# 设置 Tushare Token
export TUSHARE_TOKEN="你的token"
```

## 快速上手

```bash
# 首次全量回填（2016~至今）
python sync_tushare.py backfill --start-date 20160720 --end-date 20260720

# 后续增量更新
python sync_tushare.py update --as-of 20260720

# 离线校验本地数据完整性
python sync_tushare.py verify

# 快速连通性测试
python sync_tushare.py smoke
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--apis fina_indicator,income` | 只同步指定数据集（默认 `all`） |
| `--output-dir ./data` | 输出目录 |
| `--workers 1` | 并发拉取数 （api限制，默认为1，不能并发）|
| `--force` | 强制重新下载已校验的分区 |
| `--allow-shrink` | 允许新数据行数少于旧数据 |
| `--log-level DEBUG` | 调试日志 |

## 支持的数据集

| 名称 | 说明 |
|---|---|
| `fina_indicator` | 财务指标（EPS、ROE 等） |
| `forecast` | 业绩预告 |
| `express` | 业绩快报 |
| `income` | 利润表 |
| `balancesheet` | 资产负债表 |
| `cashflow` | 现金流量表 |
| `stk_holdernumber` | 股东人数变化 |
| `stk_holdertrade` | 股东增减持 |
| `index_member_all` | 申万行业成分 |
| `ci_index_member` | 中信行业成分 |

## 本地数据结构

```
data/
├── _catalog.json              # 归档元信息
├── fina_indicator/
│   ├── _manifest.json
│   └── period/20260331/data.parquet
├── stk_holdernumber/
│   ├── _manifest.json
│   └── ann_year/2026/data.parquet
└── index_member_all/
    ├── _manifest.json
    └── data.parquet
```

## 设计要点

- **Schema 来自文档**：字段定义从 `使用说明/` 目录的 Markdown 解析，与 Tushare API 文档同步。
  - 只有写在 `_models.py` 中的数据集会被解析
- **崩溃可恢复**：每个分区完成后原子更新 manifest，中断后重跑会自动跳过已完成分区。
- **防误覆盖**：拒绝用更少行数的分区覆盖已有数据（常见于 API 限流），除非加 `--force`。
- **离线校验**：`verify` 命令检查校验和、行数、schema、日期约束，无需网络。
