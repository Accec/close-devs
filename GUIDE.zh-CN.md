# Close-Devs 使用指南

## 1. Close-Devs 是什么

Close-Devs 是一个面向长期仓库维护的 Python 3.11+ 系统，核心始终只有
3 个 agent：

- `StaticReviewAgent`
- `DynamicDebugAgent`
- `MaintenanceAgent`

它不是一次性脚本，而是一个可持续运行的仓库维护系统。Close-Devs 会持续
扫描仓库、执行静态审查和动态调试、生成安全补丁、验证补丁、持久化历史，
并支持本地运行和 GitHub PR 流程。

需要特别明确的一点：

- `Orchestrator` 是确定性 supervisor
- 它不是第 4 个核心 agent

## 2. 架构与运行时模型

当前运行模型是：

- supervisor 负责宏观编排
- 3 个 agent 各自运行自治 session
- 每次运行都有隔离环境
- 持久化以 PostgreSQL 为主，SQLite 为兼容路径

一次标准维护闭环大致会经历：

1. 扫描仓库并生成变更集
2. 并发执行 `StaticReviewAgent` 和 `DynamicDebugAgent`
3. 合并 findings 和 handoff
4. 由 `MaintenanceAgent` 生成补丁建议
5. 在 validation workspace 中再次跑静态和动态验证
6. 写出报告、补丁、环境信息、skill 信息和历史记录

当前实现已经是 agentic runtime，而不是单次 helper 调用。每个 agent 都会
在预算内自己决定：

- 先读哪些文件
- 先调用哪些工具
- 什么时候继续探索
- 什么时候可以 finalize

当前系统的核心实现基础：

- `asyncio`
- `LangGraph`
- `Tortoise ORM`
- `Aerich`

## 3. 三个 Agent 的职责边界

### `StaticReviewAgent`

负责：

- 静态工具检查
- 语义代码审查
- 架构/正确性观察
- 结构化 `finding` 和 `handoff`

能做：

- 读文件
- 搜索代码
- 看 diff
- 生成 AST 摘要
- 跑 `ruff`、`mypy`、`bandit` 一类静态工具

不能做：

- 写文件
- 修改仓库
- 以 debugger 身份执行写入型运行步骤

### `DynamicDebugAgent`

负责：

- 跑测试和复现命令
- 收集 stdout/stderr
- 分析 traceback 和运行错误
- 生成运行态 fix request

能做：

- 跑测试命令
- 解析 traceback
- 读相关文件和日志
- 在预算内多轮定位问题

不能做：

- 写文件
- 改仓库
- 代替静态审查做最终规范裁决

### `MaintenanceAgent`

负责：

- 消费上游 findings 和 handoff
- 读取相关文件
- 生成安全补丁建议
- 在 maintenance workspace 中准备代码修改

能做：

- 读文件
- 看 diff
- 生成 safe patch
- 在维护工作区内写文件

不能做：

- 代替静态 agent 做最终静态裁决
- 代替动态 agent 做最终运行诊断
- 默认获得 push/commit 权限

当前只有 `MaintenanceAgent` 允许写代码，而且默认只写到 report-local 的
maintenance workspace。只有在 `auto_apply_patch = true` 时，才会显式回写
原始仓库。

## 4. 执行环境与隔离运行时

每次 run 都会在：

`reports/<run_id>/runtime/`

下创建一套隔离运行时，当前目录结构固定为：

- `base_workspace/<repo_name>`
- `maintenance_workspace/<repo_name>`
- `validation_workspace/<repo_name>`
- `.venv`

含义如下：

- `base_workspace`：初始分析工作区
- `maintenance_workspace`：维护 agent 的可写工作区
- `validation_workspace`：补丁验证工作区
- `.venv`：本次 run 专属虚拟环境

这意味着 Close-Devs 默认不会优先相信宿主环境。静态分析、测试、验证都应
优先跑在 report-local 的 workspace 和 venv 上。

### 依赖自动探测

当前依赖探测顺序固定为：

1. `src/requirements.txt`
2. `requirements.txt`
3. `requirements-dev.txt`
4. `requirements-test.txt`
5. `pyproject.toml:project.dependencies`

如果命中依赖源，Close-Devs 会：

1. 创建 report-local venv
2. 升级 `pip`、`setuptools`、`wheel`
3. 安装项目依赖
4. 安装分析辅助工具，如 `ruff`、`mypy`、`bandit`、`pytest`

如果安装失败，当前策略不是直接 fail fast，而是把环境标记为 `degraded`，
并把错误写进报告与环境产物。

## 5. 安装与首次运行

### PostgreSQL 主路径

如果你要使用默认的接近生产/CI 的本地运行方式，使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d postgres
aerich upgrade
python main.py run-once --config config/default.toml --repo .
```

### SQLite 轻量本地路径

如果你不想启 Docker，只想快速本地跑通：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export DATABASE_URL=sqlite:////tmp/close_devs_local.db
python main.py run-once --config config/default.toml --repo .
```

这是可行的，因为 Close-Devs 会优先读取 `DATABASE_URL`，并自动从 DSN 推断
SQLite 后端。

### 首次运行成功后你应该看到什么

一次成功的首次运行通常会：

- 创建数据库中的 run 记录
- 创建 `reports/<run_id>/`
- 创建 `reports/<run_id>/runtime/.venv`
- 输出 `summary.md`、`report.json`、`findings.json`
- 在有补丁时输出 `patch.diff`

## 6. 配置说明

主要配置文件在：

- `config/default.toml`

### `[app]`

负责控制：

- 目标仓库根目录
- state 目录
- reports 目录
- 扫描间隔
- 日志级别
- 是否输出 agent activity 日志
- 是否自动回写 patch 到原仓库
- include / exclude 规则

重点字段：

- `repo_root`
- `reports_dir`
- `log_level`
- `log_agent_activity`
- `auto_apply_patch`
- `rules_path`
- `include`
- `exclude`

### `[llm]`

控制所有 agent 默认使用的模型配置，除非某个 agent 有单独 override。

重点字段：

- `provider = "mock" | "openai_compatible"`
- `model`
- `base_url`
- `api_key_env`
- `timeout_seconds`
- `temperature`
- `system_prompt`

如果配置的 API key 环境变量不存在，Close-Devs 会记录 warning，并自动回退
到 `mock`。

### `[static_review]`

控制静态工具行为：

- `max_complexity`
- `ruff_command`
- `mypy_command`
- `bandit_command`

### `[dynamic_debug]`

控制运行命令行为：

- `smoke_commands`
- `test_commands`
- `timeout_seconds`

### `[database]`

控制持久化后端：

- `backend = "postgres" | "sqlite"`
- `url`
- `url_env`
- `echo`

本地、CI、一次性调试时，最常见的覆盖方式就是设置 `DATABASE_URL`。

### `[environment]`

控制 report-local 隔离环境：

- `enabled`
- `scope`
- `install_mode`
- `install_fail_policy`
- `python_executable`
- `bootstrap_tools`

当前默认含义是：

- 隔离环境启用
- 覆盖所有分析流程
- 依赖安装自动探测
- 安装失败时标记 `degraded`，而不是直接中断

### `[skills]`

控制 repo skill system 和 shadow self-upgrade：

- `enabled`
- `repo_root`
- `shadow_evaluation_enabled`
- `min_shadow_runs`
- `promotion_margin`

### `[skills.static]`、`[skills.dynamic]`、`[skills.maintenance]`

当前每个 agent skill 段主要控制：

- `baseline`
- `auto_upgrade`

### `[agents.static]`、`[agents.dynamic]`、`[agents.maintenance]`

这些配置定义每个 agent 的硬上限和工具权限。

重点字段：

- `model`
- `max_steps`
- `max_tool_calls`
- `max_wall_time_seconds`
- `max_consecutive_failures`
- `max_budget_ceiling`
- `safety_lock`
- `allowed_tools`
- `allowed_tool_superset`

要点：

- skill 只能在这些边界内优化行为
- skill 不能突破这些边界

### `[github]`

控制 PR 工作流的发布行为：

- provider 与 repo 上下文
- token 环境变量
- fix branch 前缀
- review mode
- artifact 保留时间

### `[pr_workflow]`

控制 PR 发布策略：

- inline comment 数量上限
- 是否允许 companion PR
- 是否只允许 safe fix
- issue comment rerun trigger

## 7. 本地 CLI 工作流

所有公开本地入口都来自 `main.py`。

### `run-once`

```bash
python main.py run-once --config config/default.toml --repo .
```

适用场景：

- 你要跑完整维护闭环
- 你要一次性拿到 static、dynamic、maintenance、validation 的完整结果

主要效果：

- 创建 run 记录
- 创建 report-local 环境
- 跑完整工作流图
- 输出报告目录

### `scan`

```bash
python main.py scan --config config/default.toml --repo .
```

适用场景：

- 你只想看扫描结果
- 你想确认 include / exclude 是否符合预期

主要输出：

- tracked file 数
- changed file 数
- added / removed file 数

### `review`

```bash
python main.py review --config config/default.toml --repo .
```

适用场景：

- 只想跑静态审查
- 你在调静态规则或 static skill

主要输出：

- 一份 static-review-only 报告
- findings 和 skill 元数据

### `debug`

```bash
python main.py debug --config config/default.toml --repo .
```

适用场景：

- 只想跑动态调试
- 你在调 repro 命令或 dynamic skill

主要输出：

- 一份 dynamic-debug-only 报告

### `report`

```bash
python main.py report --config config/default.toml --repo .
python main.py report --config config/default.toml --repo . --show
```

适用场景：

- 想看最近一次 report 目录
- 想直接把最近一次 `summary.md` 打到终端

### `skill-status`

```bash
python main.py skill-status --config config/default.toml --repo .
python main.py skill-status --config config/default.toml --repo . --agent static_review
```

适用场景：

- 想看每个 agent 当前 active skill 是什么
- 想知道当前是否有 candidate
- 想看 binding 是否被冻结

主要输出包含：

- active version
- source
- candidate version
- shadow run 数
- candidate status
- frozen 状态

### `skill-history`

```bash
python main.py skill-history --config config/default.toml --repo . --agent static_review
```

适用场景：

- 想查看某个 agent 最近的 skill evaluation 历史

主要输出包含：

- run id
- active version
- candidate version
- active score
- candidate score
- 是否晋升
- reasons

### `skill-freeze`

```bash
python main.py skill-freeze --config config/default.toml --repo . --agent maintenance
python main.py skill-freeze --config config/default.toml --repo . --agent maintenance --unfreeze
```

适用场景：

- 想暂停某个 agent 的自动晋升
- 后续再恢复自动晋升

### `skill-promote`

```bash
python main.py skill-promote --config config/default.toml --repo . --agent dynamic_debug
```

适用场景：

- 想手动晋升当前 open candidate

效果：

- 如果存在 open candidate，就更新 DB binding 指针
- 不会改 repo 里的 skill pack 文件

## 8. GitHub PR 工作流

Close-Devs 当前采用两阶段 PR 工作流。

### 第一阶段：`pr-review`

```bash
python main.py pr-review \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --repo "$GITHUB_WORKSPACE"
```

适用场景：

- GitHub Actions 需要分析一个 PR
- 你先只想拿 report 和 publish context，不立即发布评论

主要效果：

- 读取 PR 上下文
- 扫描 PR 仓库状态
- 跑 static 和 dynamic agent
- 跑 maintenance 和 validation
- 写出 report 产物
- 写出 `artifacts/publish_context.json`

### 第二阶段：`pr-publish`

```bash
python main.py pr-publish \
  --config config/default.toml \
  --event-path "$GITHUB_EVENT_PATH" \
  --publish-context "reports/<run_id>/artifacts/publish_context.json" \
  --repo "$GITHUB_WORKSPACE"
```

适用场景：

- report artifact 已经上传
- 需要发布 summary comment、artifact 链接、inline comments、companion PR

主要效果：

- 解析 artifact URL
- 更新一条稳定的 Close-Devs summary comment
- 在满足条件时发布保守型 inline comments
- 在允许且可发布时创建或更新 companion PR

### PR 发布模式

当前 publish mode 有：

- `companion_pr`
- `comment_only`
- `artifact_only`

Close-Devs 会在以下场景自动降级：

- token 缺失
- 权限不足
- PR 来自 fork
- validation 导致 patch 不可发布

## 9. Skill System 与 Shadow Self-Upgrade

repo skill system 是当前运行时的重要控制面。

基线 skill pack 放在：

- `config/skills/static/`
- `config/skills/dynamic/`
- `config/skills/maintenance/`

每套 pack 当前包含：

- `manifest.toml`
- `policy.toml`
- `skill.md`
- `examples.json`

### Skill pack 控制什么

skill pack 会影响：

- prompt guidance
- planning heuristics
- 工具优先级
- 严重级别和优先级偏好
- handoff 风格
- completion checklist
- reflection 与 upgrade hint 风格

### Active skill 与 candidate skill

- repo baseline skill：git 中版本化保存的基线模板
- active skill：数据库中当前实际绑定的版本
- candidate skill：数据库中待评估/待晋升的候选版本

也就是说，运行时真正生效的是 DB binding，不一定等于 repo 里的基线文件。

### 什么是 shadow evaluation

shadow evaluation 的意思是：

- 生产运行仍然使用 active skill
- candidate skill 会被一起评估
- 是否晋升由确定性指标主导，不是随意覆盖

当前晋升控制主要来自 `[skills]`：

- `min_shadow_runs`
- `promotion_margin`

### 安全边界

skill 不能：

- 给 `StaticReviewAgent` 写权限
- 给 `DynamicDebugAgent` 写权限
- 给 `MaintenanceAgent` 默认 push/commit 权限
- 突破 `[agents.*]` 里定义的硬上限

skill 可以：

- 调整工具调用顺序
- 调整优先级策略
- 调整 handoff 风格
- 在硬上限之内建议更保守的预算

### 如何运维 skill system

用：

- `skill-status` 看当前状态
- `skill-history` 看评估历史
- `skill-freeze` 暂停自动晋升
- `skill-promote` 手动激活 open candidate

## 10. 报告、产物与如何解读

每次 run 会输出：

- `reports/<run_id>/summary.md`
- `reports/<run_id>/report.json`
- `reports/<run_id>/findings.json`
- `reports/<run_id>/patch.diff`
- `reports/<run_id>/artifacts/`

### 主要报告产物

#### `summary.md`

人类可读的总览报告，当前会包含：

- workflow 元信息
- execution environment 状态
- agent skills 摘要
- static review 结果
- dynamic debug 结果
- maintenance 结果
- validation 结果

#### `report.json`

机器可读的完整 workflow report，包含 metadata 和 agent result artifacts。

#### `findings.json`

当前 run 的扁平 findings 列表。

#### `patch.diff`

当 maintenance 生成了 patch 时，这里会保存统一 diff。

#### `artifacts/environment.json`

隔离环境的机器可读摘要，包含：

- runtime 路径
- 依赖来源
- install commands
- install failures
- 是否 degraded

#### `artifacts/install.log`

venv / bootstrap / 依赖安装过程的原始日志。

#### PR 专属产物

PR 工作流还可能生成：

- `artifacts/publish_context.json`
- `artifacts/review_payload.json`

### 正确理解 workflow status

很重要的一点：

- `Status: succeeded` 只表示 workflow 执行成功
- 不等于 repo 一定已经被修好

你还必须继续看：

- static findings
- dynamic findings
- maintenance patch
- validation 结果

### 怎么判断问题其实没解决

以下情况通常说明主问题仍未解决：

- `dynamic_validation` 还在报原始 blocker
- `maintenance` 只生成了低风险 cosmetic patch
- `validation` 仍然包含高严重级 finding
- 报告里出现 `regressed` 或 unresolved 的比较结果

### 怎么判断是否真的调用了 AI

看这几个信号：

- 启动日志里没有 `Falling back to mock`
- 日志里出现 `client=OpenAICompatibleLLMClient`
- summary 和 handoff 里有明显模型驱动的语义结论，而不只是 deterministic 工具输出

如果有 fallback warning，就说明这次实际跑的是 `mock`。

## 11. 日志与 Agent Activity

当：

- `[app].log_agent_activity = true`

时，Close-Devs 会输出较细的 agent activity 日志。

常见日志事件包括：

- `Task dispatched`
- `Session started`
- `Step decided`
- `Tool started`
- `Tool finished`
- `Session finalizing`
- `Session finished`
- `Task finished`

这些日志能帮助你快速回答：

- 当前哪个 agent 在跑
- 当前用的是哪个 LLM client
- 调了哪些工具
- 跑了几步
- 是正常结束、预算耗尽还是报错结束

### 如何解读这些日志

- `Task dispatched`：supervisor 把任务发给某个 agent
- `Session started`：自治 session 启动，带上预算和工具集
- `Step decided`：agent 决定下一步要做什么
- `Tool started`：工具调用开始
- `Tool finished`：工具调用返回，可能成功也可能失败
- `Session finished`：agent 已完成、耗尽预算或错误结束

## 12. 数据库与迁移

Close-Devs 当前是 PostgreSQL 优先，SQLite 兼容。

### PostgreSQL 路径

适用场景：

- 默认本地运行
- CI
- PR workflow
- 长期保留较完整的历史

初始化方式：

```bash
docker compose up -d postgres
aerich upgrade
```

### SQLite 路径

适用场景：

- 快速本地 smoke test
- 不想启动 Docker
- 临时单机调试

示例：

```bash
export DATABASE_URL=sqlite:////tmp/close_devs_local.db
python main.py run-once --config config/default.toml --repo .
```

### 迁移规则

如果数据库存在，但表还没建好，执行：

```bash
aerich upgrade
```

## 13. 常见问题与排障

### PostgreSQL 连接失败

现象：

- 启动时连不上 `127.0.0.1:5432`

修复：

```bash
docker compose up -d postgres
aerich upgrade
```

同时检查 `DATABASE_URL` 是否覆盖了默认配置。

### 没有执行 `aerich upgrade`

现象：

- 报错里提到缺少 `runs`、`aerich` 等表

修复：

```bash
aerich upgrade
```

### 缺少 `OPENAI_API_KEY`

现象：

- 日志里出现回退到 `mock` 的 warning

修复：

```bash
export OPENAI_API_KEY=your_key
```

然后重新运行命令。

### Report-local 环境进入 `degraded`

现象：

- `Execution Environment` 显示 `degraded`
- `install_failures` 非 0
- 测试或工具因为缺依赖而失败

应该先看：

- `artifacts/environment.json`
- `artifacts/install.log`

常见原因：

- 依赖文件错误
- 私有依赖拉取失败
- 当前项目布局不在 v1 自动探测支持范围内

### `pr-publish` 找不到 artifact 或无法更新评论

现象：

- publish 阶段拿不到 artifact URL
- PR summary comment 没被更新

检查：

- GitHub Actions 是否上传了本次 run artifact
- `GITHUB_TOKEN` 是否存在
- token 是否具备 comment / publish 权限
- `publish_context.json` 路径是否正确

### SQLite 提示 `database is locked`

现象：

- 本地 SQLite 跑着跑着出现 `database is locked`

常见原因：

- 多个进程同时复用同一个 SQLite 文件

修复方式：

- 每次 smoke run 使用独立 SQLite 文件
- 或停止并发本地运行
- 或直接切回 PostgreSQL

## 14. 推荐使用实践

对 operator：

- 默认使用 PostgreSQL
- 保持 `log_agent_activity = true`
- 不要只看 `Status: succeeded`
- 接受 patch 前一定看 validation

对 contributor：

- 快速本地调试时可以用 SQLite
- 调 agent 行为时优先看 `skill-status` 和 `skill-history`
- 如果在追一个 skill 回归，先 freeze 再调
- 把 repo skill pack 当作版本化基线，而不是运行时临时状态

对仓库维护本身：

- 保持依赖文件准确，让隔离环境能自动 bootstrap
- 在 validation 结果持续稳定前，优先保守使用 safe autofix
- 把 agent summary 当证据入口，但最终判断还是以 report artifacts 和 validation 结果为准
