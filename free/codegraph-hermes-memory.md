# CodeGraph + Hermes 记忆机制 讨论总结

## 1. CodeGraph 概述

### 是什么
Pre-built 代码知识图谱，Rust kernel + tree-sitter 解析 → SQLite 存储 → MCP server 暴露给 AI agent。

- **支持 35+ 语言**（含 CUDA、C++、Python 等）
- **100% 本地**：不上传代码，无 API key
- **自动同步**：原生 OS 文件监听（inotify/FSEvents），2 秒防抖，保存即更新
- **支持 Agent**：Claude Code、Cursor、Codex、OpenCode、**Hermes Agent**、Gemini、Antigravity、Kiro

### Benchmark 数据（7 个真实仓库，Claude Opus 4.8）
| 指标 | 无 CodeGraph | 有 CodeGraph | 改善 |
|------|:----------:|:----------:|------|
| 工具调用 | 最多 57 次 | 1-3 次 | 89%↓ |
| Token | 最多 4.3M | 156k-386k | 69%↓ |
| 成本 | — | — | 60%↓ |
| 文件读取 | 1-24 个 | 0（七个仓库全为 0） | 100%↓ |

### 核心 MCP 工具
`codegraph_explore(query, projectPath?)`：一个调用返回符号源码 + 调用路径 + 影响半径。

---

## 2. 知识图谱存储位置

`codegraph init` 后在项目根目录生成：

```
your-project/
├── .codegraph/
│   ├── codegraph.db          ← SQLite 数据库（含 FTS5 全文索引）
│   └── ...                   ← 锁文件、元数据
```

一个 SQLite 文件搞定全部。WAL 模式，读不阻塞写。

---

## 3. 宿主机 vs 容器安装

**结论：图内容本身一样，但 MCP 服务器必须 Agent 能启动到。**

| 场景 | 图一样？ | 能用吗？ | 原因 |
|------|:---:|:---:|------|
| 宿主机装 + 宿主机 init | ✅ | ✅ | Agent 在宿主机，直接启 codegraph |
| 容器装 + 容器 init | ✅ | ❌ | Agent 在宿主机，找不到容器内二进制 |
| 容器装 + docker exec 桥接 | ✅ | ⚠️ | 理论可行，管道/生命周期可能踩坑 |

**推荐：装宿主机**。项目路径放 WSL 原生文件系统（`~/` 下），不要放 `/mnt/c/`（SQLite WAL 和 socket 在跨文件系统时不靠谱）。

---

## 4. 文件权限问题

### 根因
Docker 默认以 root（uid=0）运行，bind mount 上容器创建的文件 owner 是 root，宿主机普通用户写不进去。

### 对 CodeGraph 的影响
**不影响**——CodeGraph 只需要**读**源码（read 对 others 开放即可），`.codegraph/` 是宿主机 init 的，owner 是自己。

权限问题只影响 Agent **改代码**（write_file/patch）的时候。

### 解决方案

**方案一：chown（一次性）**
```bash
sudo chown -R $USER:$USER /home/armin/project/
```

**方案二：Docker --user（根治，推荐）**
```bash
docker run --user $(id -u):$(id -g) -v /home/armin/project:/app ...
```

docker-compose：
```yaml
services:
  myapp:
    user: "${UID}:${GID}"
```

启动：`UID=$(id -u) GID=$(id -g) docker-compose up`

---

## 5. Hermes 记忆体系（三层）

```
第一层：Memory（持久记忆，~2KB）
  ↓  每个 session 自动注入。存稳定事实：项目路径、偏好、工具坑点。
  ↓  不存：代码细节、任务进度、一次性信息。

第二层：Session Search（会话回溯）
  ↓  按需搜索历史会话 SQLite 数据库。需要关键词明确。
  ↓  不会自动注入新 session，需要主动搜索或用户提示。

第三层：CodeGraph（代码知识图谱，外部工具）
  ↓  项目级记忆。一次 init，跨 session 直接用。
  ↓  不受 Memory 容量限制。
```

### 跨 session 限制

新 session 启动时，我能用的只有 Memory（~2KB）。session1 的对话全文、读过的代码内容**不会自动注入**。

| 场景 | Token 浪费程度 |
|------|:---:|
| Memory 里有结论 | ✅ 不浪费 |
| 关键词明确，session_search 搜到 | ⚠️ 中等 |
| `hermes sessions -c` 回到旧会话 | ✅ 完整上下文 |
| 什么都没有，纯靠猜 | ❌ 从头啃 |
| 有 CodeGraph | ✅ 零文件读取，直接 explore |

---

## 6. 最佳实践

```
项目初始化（一次性）：
  codegraph init          ← 建图，之后自动同步

日常开发：
  Agent 自动读图回答问题   ← 不浪费 token
  关键结论存 Memory       ← 跨 session 持久化
  长期细节靠 Session Search ← 按需回溯
```

---

## 7. 使用命令速查

### 上车

```bash
codegraph install              # 接入 Agent（Hermes / Claude Code 等）
codegraph init                 # 项目初始化 + 建图（一步完成）
```

### 日常

| 命令 | 作用 |
|------|------|
| `codegraph status` | 索引统计、待同步文件 |
| `codegraph sync` | 手动增量同步（一般不需要） |
| `codegraph explore "<问题>"` | CLI 版 explore，同 MCP 效果 |
| `codegraph query <关键词>` | 搜索符号 |
| `codegraph callers <函数名>` | 谁调了这个函数 |
| `codegraph callees <函数名>` | 这个函数调了谁 |
| `codegraph impact <符号>` | 影响半径分析 |
| `codegraph node <文件或符号>` | 看源码 + caller |
| `codegraph affected src/a.py` | 改了这些文件会影响哪些测试 |
| `codegraph files` | 查看项目文件结构 |

### 维护

```bash
codegraph upgrade                # 升级到最新版
codegraph upgrade --check        # 只看有没有新版本
codegraph uninit                 # 从项目移除（删 .codegraph/）
codegraph uninstall              # 从所有 agent 卸载 + 移除 CLI
codegraph uninstall --keep-cli   # 只移除 agent 配置，保留 CLI
```

### 环境变量

```bash
CODEGRAPH_NO_DAEMON=1              # 禁用后台共享服务器（WSL /mnt/c 用）
CODEGRAPH_DIR=.codegraph-win       # Windows/WSL 双开时不同目录名
CODEGRAPH_WATCH_DEBOUNCE_MS=5000   # 文件监听防抖 ms（默认 2000）
```

---

## 8. 容器开发场景：CodeGraph 权限实践

### 场景

宿主机 WSL，项目通过 Docker bind mount 共享：

```
宿主机: /home/armin/repos/sipe-nocs/mixedai
容器内: /root/workspace/MixedAI_Datacenter/sipe-nocs/mixedai
```

### 问题

Docker 默认以 root 运行，bind mount 上的文件和目录 owner 都是 root。宿主机普通用户（armin）无法写入，`codegraph init` 创建 `.codegraph/` 时会报权限错误。

### 解决

```bash
# 一次性改权限
sudo chown -R armin:armin /home/armin/repos/sipe-nocs/mixedai

# 然后 init
cd /home/armin/repos/sipe-nocs/mixedai
codegraph init
```

### 为什么改成一次就行

| 操作 | 权限需求 | 满足？ |
|------|:---:|:---:|
| CodeGraph 解析源码 | 只读（others 可读） | ✅ 容器默认 644 |
| CodeGraph 写索引到 `.codegraph/` | 写入 | ✅ owner 是 armin |
| 容器内新建文件 | — | ⚠️ 还是 root，但 others 可读 |
| inotify 监听 | — | ✅ bind mount 跨容器正常工作 |

结论：只要容器里不设 `chmod 600` 关闭 others 读权限，CodeGraph 全链路正常。

### 预防复发

日常进容器干活带 `-u`，让新建文件的 owner 和宿主机一致：

```bash
docker exec -it -u $(id -u):$(id -g) <容器名> bash
```

如果不想每次都敲，容器内建同名用户（uid/gid 对齐）或配置 docker-compose 的 `user` 字段。

### 容器内代码（非 bind mount）的处理

如果代码只在容器内（如 `/root/mixedai/`），宿主机 codegraph 看不到：

**思路一：docker commit + 重建（推荐）**

```bash
docker commit <容器名> mixedai-image
docker stop <容器名>
docker run --user $(id -u):$(id -g) \
  -v /home/armin/repos/sipe-nocs/mixedai:/root/mixedai \
  ...其他参数... \
  mixedai-image
```

容器内环境全部保留，代码路径变成 bind mount，宿主机直接 codegraph init。

**思路二：容器内装 codegraph + docker exec 桥接**

MCP 配置中把 `command: codegraph` 改为 `command: docker exec -i <容器名> codegraph`，CodeGraph 在容器内运行，但返回的路径是容器路径，Agent 无法直接用宿主机路径操作文件，不推荐。
