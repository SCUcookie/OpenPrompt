# AI4S Harness 设计与实现说明

日期：2026-06-11

目标：在不替换 OpenRSD/MMEngine 训练后端的前提下，构建一个可复现、可审计、轻量的本地实验 harness，自动化 GPU 发现/分配、启动、监控、故障检测、指标抽取、实验记录与诊断。

前提假设与约束
- OpenRSD/MMEngine 为训练与评估的事实来源。
- 首版本不依赖外部服务（例如 MLflow、W&B、Prefect）。
- 大型 artefacts（checkpoint、完整日志、原始预测、数据集）不纳入 Git。

总体框架与流程概览
1. 规范定义（Spec）
2. 验证与解析（Validate & Resolve）
3. GPU 选择（Select GPU）
4. 启动（Launch）
5. 运行监控（Monitor）
6. 指标与日志解析（Parse）
7. 报告生成（Report）
8. 失败诊断（Diagnostics）
9. 注册与归档（Registry & Archive）

每个阶段的职责、方法、策略与技术细节

**1. 规范定义（`specs.py`）**
- 职责：定义运行规格类型化结构，校验必需字段，解析相对路径，渲染命令模板。
- 输入：`configs/harness/*.yaml` 或 `*.json`。
- 输出：Validated Spec 对象，命令列表/argv，normalized provenance 字段。
- 方法：
  - 使用 Pydantic 或 dataclasses + 类型检查实现 schema（推荐 Pydantic 便于校验/默认值）。
  - 支持字段：`run_id, stage, dataset, config, workdir, command.argv, gpu_policy, startup.progress_regex, completion.require_metric_keys, failure_signatures, provenance`。
- 策略与原因：
  - 明确定义及类型化可避免 ad-hoc shell launch 导致的不可复现性。
  - 命令模板以 `argv` 形式存储，避免 shell 转义问题，方便 dry-run 与审核。
- 技术细节：
  - 对路径调用 `Path.resolve()` 并验证存在性（可通过 `--allow-missing` 标记绕过）。
  - 命令渲染：若存在模板参数，使用安全替换（`str.format_map` 或 jinja2）；优先原始 argv 保留精确可重播命令。
  - Spec digest：对规范文本计算 SHA256 作为不可变索引字段。

示例（简化 YAML）：

```yaml
run_id: example_run
stage: S2
backend: opensrd_mmengine
dataset:
  name: DOTA
  root: /data/..../DOTA
command:
  argv:
    - /path/to/python
    - tools/train.py
    - /path/to/config.py
gpu_policy:
  allowed: [0,1,2]
  idle_memory_mib: 1000
  idle_util_percent: 10
  stable_polls: 3
startup:
  progress_regex: "Epoch\\(train\\) \[1\].*"
failure_signatures:
  - Traceback
  - CUDA out of memory
```

**2. 注册表（`registry.py`）**
- 职责：维护追加式运行注册表 `records/harness/runs.jsonl`，并存储事件流（`records/harness/events/<run>.jsonl`）。
- 数据模型：每个事件一行 JSON，例如：
  - `run_id, event_type, timestamp, spec_path, spec_digest, backend, command, workdir, log_path, screen, pid, gpu, commit, config_path, metric_summary, report_path`
- 策略与原因：使用 JSONL 便于追加写入、grep、恢复与流式处理；append-only 减少并发冲突风险。
- 方法：
  - 使用文件锁（flock）或原子写入/重命名策略确保并发安全。
  - 提供查询接口：按 run_id 获取最新事件、按状态筛选运行、按时间范围导出事件流。
- 技术细节：
  - 写入：将事件写为一行 JSON，fsync 可选（可通过配置控制）。
  - 事件合成：监控模块写出状态转换事件，reporter 写出 report 生成事件。

**3. GPU 查询与选择（`gpu.py`）**
- 职责：封装 `nvidia-smi` 查询，生成 GPU 快照，判断空闲 GPU，提供 polling 快照。
- 输入：allowed GPU 列表、idle 内存阈值、idle 利用率阈值、stable_polls、排除进程名。
- 输出：候选 GPU 或选择结果、周期性快照用于事件记录。
- 策略与原因：单纯用内存阈值会误判（驻留非训练进程），所以需结合进程名单与 owner 信息并排除已知长期服务进程（如 VLLM::EngineCore）。
- 方法/技术细节：
  - 调用：`nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits` 与 `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`。
  - 解析策略：解析 CSV 输出为 dict 列表；计算 `free_memory = total - memory.used`；计算是否低于阈值并且 GPU 中运行的进程名不在 `exclude_process_names`。
  - 稳定判断：连续 `stable_polls` 次轮询都满足空闲条件才认为可选。
  - 进程归属：尝试 `ps -o user= -p <pid>` 以判断是否为当前用户运行的占用，尽量避免抢占别人的训练。
  - 可注入的模拟输出：为测试允许注入 `nvidia-smi` 捕获样本字符串。

**4. 启动器（`launcher.py`）**
- 职责：基于 resolved spec 启动作业（`screen` 或子进程），并写入 launch 事件。
- 输入：resolved spec、selected GPU、env overrides、workdir、log path、dry-run flag。
- 输出：launch 事件（timestamp, command, screen name, PID, GPU, log path）。
- 策略与原因：保留当前 server-friendly 的 detached `screen` 流程以便后台长期训练，同时支持 foreground 以便调试。
- 方法/技术细节：
  - 环境控制：在启动前设置 `CUDA_VISIBLE_DEVICES=<chosen>`、`PYTHONNOUSERSITE=1`、`PYTHONPATH`、`MPLCONFIGDIR`（避免 matplotlib 在多用户服务器写配置）等。
  - 命令执行：
    - dry-run：打印完整 argv（JSON 或 shell-escaped 字符串），不执行。
    - screen：创建唯一 session 名称（例如 `harness_<run_id>_<ts>`），并用 `screen -dmS <name> bash -c 'exec > >(tee -a <log>) 2>&1; exec env ... <cmd> '` 启动。记录 `screen -ls` 输出解析到 PID（若可得）。
  - 日志处理：确保 stdout/stderr 都重定向到日志文件并保证日志立即刷盘（`stdbuf -oL` 或 Python logging flush）。
  - 权限：如需 sudo 或特殊用户，明确禁止在 harness 默认行为中自动提权，需人工批准。

**5. 监控（`monitor.py`）**
- 职责：轮询进程、日志与 GPU，检测启动接受、运行、完成或失败状态，并写事件到 registry。
- 输入：run id、launch event、log path、progress_regex、checkpoint 期待、failure_signatures、GPU polling policy。
- 输出：状态转换事件、failure context、事件快照文件。
- 策略与原因：把人工巡检流程形式化，避免遗漏早期失败信号并能及时触发诊断。
- 方法/技术细节：
  - 启动接受（accepted）：要求 `screen`/PID 存在、日志文件大小增长，并且在日志中匹配 `progress_regex`。
  - 运行心跳：定期检查 PID 是否存在、日志是否增长、GPU 是否显示训练进程驻留（根据 `nvidia-smi` 的进程 PID 列表）。若任一条件在超时内未满足，则标记异常或等待重试。
  - 完成判定：进程正常退出（exit code 0）或检测到 final checkpoint + final metric 可解析。
  - 失败判定：在日志尾部出现任一 failure signature 或进程非正常退出。针对 `nan`/`inf`：解析指标文件优先，如果只能从日志中判断，应做上下文过滤避免匹配到 metainfo 中的字符串（例如使用正则 `\bnan\b` 并结合数字上下文）。
  - 事件采样：定期写入 `records/harness/events/<run>.jsonl` 包含 timestamp、log_cursor、gpu_snapshot、pid_state、last_tail_lines。

**6. 解析器（`parsers.py`）**
- 职责：从 MMEngine 日志、`vis_data/scalars.json`、`metrics.json`、checkpoint文件中提取训练进度和指标曲线。
- 方法/技术细节：
  - MMEngine scalars：读取 `vis_data/scalars.json`（JSON），提取 key-path（例如 `dota/mAP`），返回时间序列（step, value, walltime）。
  - 日志进度：基于 configured `progress_regex` 提取当前 epoch/iter 进度；对长日志做增量解析（维持文件 cursor）。
  - Checkpoint：通过检查 `last_checkpoint` 指向或 `epoch_*.pth` 文件的存在与修改时间推断完成阶段。
  - 崩溃摘要：对日志尾部做 sliding-window 提取（例如最后 200 行），然后基于 failure_signatures 和语义规则抽取上下文片段。

**7. 报告生成（`reporter.py`）**
- 职责：根据注册表记录、解析器输出与规格，生成 `docs/experiments/YYYYMMDD_<run>.md` 的人类可读实验记录。
- 输出字段：文档必须包含 `docs/experiments/README.md` 中列出的所有字段；若缺失则写 `not captured`。
- 方法/技术细节：
  - 使用模板化 Markdown（Jinja2）或程序化字符串插值构建报告，包含：运行头（run_id, date, commit, machine, GPUs）、数据集、config path、命令行、验证命令、checkpoint、log、指标摘要、class-wise 指标、失败笔记、下一步建议。
  - 可选生成 metric JSON 压缩版（用于快速渲染曲线）。

**8. 诊断（`diagnostics.py`）**
- 职责：当检测到失败时，汇总失败上下文为 AI 可读的诊断包，便于后续自动或人工分析。
- 内容：失败命令、config diff（如与 parent run 不同）、GPU 快照、日志尾部、指标曲线、小型 checkpoint 摘要、怀疑的失败类别、建议下一步。输出为 JSON 与可选 Markdown。
- 技术细节：
  - Config diff：对运行时使用的 config 与 parent config 做行级 diff（或结构化 diff），保留上下文 10 行。
  - 失败分类：基于 signature 和简单规则树判定（比如若日志包含 `CUDA out of memory` -> OOM，若包含 `libpng` -> I/O）。对于 nan/inf，检查最近 N 个数值点，若连续为 NaN 则标注训练数值不稳定。
  - 输出 JSON schema：{run_id, failure_class, top_errors:[], log_tail:[], gpu_snapshot, suggested_actions:[]}

**9. CLI（`scripts/harness.py`）**
- 子命令：`validate-spec`, `launch`, `status`, `monitor`, `collect`, `report`, `diagnose`。
- 实现建议：使用 `argparse` 或 `click`（推荐 `click` 提高可组合性）。
- dry-run：`validate-spec --dry-run` 输出完全解析后的命令 plan 与 environment summary，不做任何启动或写 registry 操作（除非 `--record-dry`）。

**文件布局建议**
- `src/openprompt_rs/harness/`:
  - `__init__.py`
  - `specs.py`
  - `registry.py`
  - `gpu.py`
  - `launcher.py`
  - `monitor.py`
  - `parsers.py`
  - `reporter.py`
  - `diagnostics.py`
  - `cli.py`（或 `scripts/harness.py`）
- `configs/harness/`（规范）
- `records/harness/runs.jsonl`
- `records/harness/events/`（per-run）
- `docs/experiments/`（自动生成 report）
- `artifacts/harness/`（诊断包、metric 表等）

**Acceptance Gates（启动 / 运行 / 完成 / 失败）**
- 启动接收（accepted）：进程/`screen` 存在，日志存在且在短超时内增长，匹配 `progress_regex`。
- 运行健康（runtime health）：PID 存活，screen 列表存在（若 detach），GPU 有进程驻留（如果训练需要 GPU），无失败签名，日志 cursor 持续推进。
- 完成接收（completion）：进程退出正常或找到 final checkpoint + 可解析 final metric。
- 失败检测（failure）：匹配 failure_signatures、检测到 `nan/inf`、进程早期退出或长期无进度。

**小技巧与避免误判**
- 避免 `nan`/`inf` 假阳：优先查询结构化指标文件（`scalars.json`），仅在日志中断言时，使用数字语境（例如 `loss=nan` 或 `loss: nan`）且避免匹配诸如 `metainfo: 'nan'` 的静态文本。
- GPU 占用判断：不仅看 memory，还看 process_name 与 PID，优先避开非当前用户持有的大型长期服务。
- 日志增长判断：用 byte-size 增长而非行数（行缓冲可能延迟写入）。

**示例：启动命令（screen）**
```bash
screen -dmS harness_example_20260611 bash -c "export CUDA_VISIBLE_DEVICES=2; exec > >(tee -a /path/to/log) 2>&1; exec /path/to/python tools/train.py /path/to/config.py --work-dir /path/to/workdir"
```

**JSONL 事件示例**
```json
{"run_id":"example_run","event_type":"launched","timestamp":"2026-06-11T12:00:00Z","spec_path":"configs/harness/example.yaml","spec_digest":"...","command":["/path/to/python","tools/train.py","/path/to/config.py"],"workdir":"/path/to/workdir","log_path":"/path/to/log","screen":"harness_example_20260611","pid":12345,"gpu":2}
```

**测试计划（优先级）**
1. Spec 验证单元测试（字段、路径解析、digest）。
2. Parsers 单元测试：`scalars.json`、小型 log fixture、checkpoint 检测。3. GPU 模拟测试：注入 `nvidia-smi` 捕获文本测试排除进程名与 stable polls。4. Launcher dry-run 测试与 screen 模拟接口测试（可用 mock）。5. Monitor 算法测试（startup acceptance、failure signatures、无误报示例）。6. Reporter snapshot 测试（对比 `docs/experiments/README.md` 字段完整性）。

**扩展与下一步实现建议**
- 第一步：实现 `specs.py` 与 `scripts/harness.py validate-spec --dry-run`，并写相应单元测试。
- 第二步：实现 `registry.py` 与基础 `parsers.py`（scalars.json 解析）。
- 第三步：实现 `gpu.py` 与 `launcher.py`（dry-run + screen 启动）。
- 第四步：实现 `monitor.py` 与 `diagnostics.py`，并接入 reporter。

---

该文档旨在成为可执行的蓝图；如果你希望我直接生成 `src/openprompt_rs/harness/` 的代码骨架和若干测试文件，我可以按实现顺序继续生成具体模块实现。