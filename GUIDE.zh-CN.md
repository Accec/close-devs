# Close-Devs 使用指南

## 项目概览

Close-Devs 是一个面向长期仓库维护的 Python 3.11+ 系统，核心始终只有 3 个 agent：

- `MaintenanceAgent`
- `StaticReviewAgent`
- `DynamicDebugAgent`

系统基于 `asyncio`、`LangGraph` 和 `Tortoise ORM`，用于持续扫描仓库、执行静态分析与动态调试、生成安全补丁、验证补丁，并把运行历史沉淀到数据库中。

## 项目结构

当前源码已经扁平化到 `src/` 下：

- `src/agents`：三个核心 agent
- `src/core`：编排、CLI、模型、调度、调度器
- `src/tools`：补丁、命令执行、静态工具等复用能力
- `src/workflows`：工作流入口
- `src/memory`：数据库和历史记录
- `src/github`：GitHub PR 发布和渲染
- `src/repo`：仓库扫描与增量变更检测
- `src/reports`：报告序列化与 Markdown 渲染

入口文件：

- `main.py`

默认配置：

- `config/default.toml`

GitHub Actions 工作流：

- `.github/workflows/close-devs-pr.yml`

## 环境要求

- Python `3.11+`
- Git
- Docker，如果你要使用默认的 PostgreSQL 本地运行方式

可选但建议安装的外部工具：

- `ruff`
- `mypy`
- `bandit`
- `pytest`

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

## CLI 命令

### 单次完整维护闭环

```bash
python main.py run-once --config config/default.toml --repo .
```

这条命令会顺序完成：

1. 扫描仓库
2. 执行静态审查
3. 执行动态调试
4. 生成维护补丁建议
5. 验证补丁
6. 输出报告

### 仅扫描仓库

```bash
python main.py scan --config config/default.toml --repo .
```

### 仅静态审查

```bash
python main.py review --config config/default.toml --repo .
```

### 仅动态调试

```bash
python main.py debug --config config/default.toml --repo .
```

### 查看最新报告

```bash
python main.py report --config config/default.toml --repo .
python main.py report --config config/default.toml --repo . --show
```

## GitHub PR 工作流

Close-Devs 的 PR 处理是两阶段的。

### 第一阶段：分析 PR

```bash
python main.py pr-review \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --repo "$GITHUB_WORKSPACE"
```

这一阶段负责：

- 读取 PR 上下文
- 扫描变更文件
- 并发执行静态审查和动态调试
- 生成维护补丁建议
- 验证补丁
- 写出 `report.json` 和 `artifacts/publish_context.json`

### 第二阶段：发布 PR 结果

```bash
python main.py pr-publish \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --publish-context "reports/<run_id>/artifacts/publish_context.json" \
  --repo "$GITHUB_WORKSPACE"
```

这一阶段负责：

- 解析 GitHub Actions artifact 的真实链接
- 更新同一条稳定的 Close-Devs 顶层评论
- 在满足条件时发布保守型 inline comments
- 当 safe autofix 可发布时，创建或更新 companion PR

## 报告与产物

每次运行都会在 `reports/<run_id>/` 下生成：

- `summary.md`
- `findings.json`
- `report.json`
- `patch.diff`
- `artifacts/`

PR 工作流额外会在 `artifacts/` 中生成：

- `publish_context.json`
- `review_payload.json`，在执行 `pr-publish` 之后生成

## 配置说明

`config/default.toml` 里最关键的配置段有：

- `[app]`：仓库路径、报告路径、扫描规则
- `[llm]`：`mock` 或 `openai_compatible`
- `[static_review]`：`ruff`、`mypy`、`bandit` 命令
- `[dynamic_debug]`：smoke 命令和测试命令
- `[github]`：分支前缀、token 环境变量、评论模式、artifact 保留时长
- `[pr_workflow]`：inline comment 数量上限、companion PR 开关、安全修复策略
- `[database]`：数据库后端和 DSN

默认行为：

- 默认数据库是 PostgreSQL
- `DATABASE_URL` 会优先覆盖配置文件中的 DSN
- `auto_apply_patch = false`
- 默认 LLM provider 是 `mock`

## 数据库说明

默认主路径：

- PostgreSQL

兼容路径：

- SQLite，适合轻量本地调试和测试

迁移命令：

```bash
aerich upgrade
```

## 三个 Agent 的边界

### `StaticReviewAgent`

- 只做静态分析
- 不运行代码
- 不写补丁

### `DynamicDebugAgent`

- 只做运行、测试、日志、traceback 分析
- 不做静态规范裁决
- 不写补丁

### `MaintenanceAgent`

- 只负责补丁生成和 safe autofix
- 不给最终静态审查结论
- 不做动态运行诊断

## 推荐本地开发流程

如果你是在开发 Close-Devs 自己，建议这样走：

1. 运行 `python main.py run-once --config config/default.toml --repo .`
2. 查看最新报告
3. 执行 `pytest`
4. 修改配置或工作流
5. 重新运行

## 常见问题

### PostgreSQL 连接失败

- 确认 Docker 已启动
- 执行 `docker compose up -d postgres`
- 检查 `DATABASE_URL` 是否覆盖了默认配置

### `pr-publish` 找不到 artifact

- 确认先执行过 `pr-review`
- 确认 workflow 已上传对应报告目录
- 检查 `publish_context.json` 路径是否正确

### GitHub 没有收到评论

- 检查 `GITHUB_TOKEN`
- 检查 PR 是否来自 fork
- 查看工作流日志，确认是否降级成了 `artifact_only`

## 建议阅读顺序

1. `README.md`
2. `GUIDE.en.md` 或 `GUIDE.zh-CN.md`
3. `config/default.toml`
4. `src/core/orchestrator.py`
5. `.github/workflows/close-devs-pr.yml`
