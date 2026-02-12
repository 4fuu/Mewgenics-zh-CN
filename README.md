# Mewgenics 简体中文汉化补丁

为 Edmund McMillen 的猫咪养成 roguelike 游戏 **Mewgenics** 制作的社区中文本地化补丁。

> ⚠️ 游戏处于 Early Access 阶段，翻译内容可能随版本更新而变化。

## 安装补丁（玩家）

1. 从 [Releases](../../releases) 下载 `resources.gpak`
2. 备份游戏目录下的原始 `resources.gpak`（重命名为 `resources.gpak.bak` 即可）
3. 将下载的 `resources.gpak` 放入游戏目录（`Steam/steamapps/common/Mewgenics/`）
4. 启动游戏，在设置中将语言切换为中文

如需还原，将备份文件改回 `resources.gpak` 即可。

## 参与翻译（贡献者）

翻译存储在 `translation_progress.json` 中，格式为：

```json
{
  "文件名::KEY": "中文译文",
  "misc.csv::AREA_NAME_TUTORIAL": "小径",
  ...
}
```

### 如何贡献

1. Fork 本仓库
2. 编辑 `translation_progress.json`，修正你认为不准确的翻译
3. 提交 Pull Request，说明修改了哪些条目及原因

**或者直接提交issue进行讨论。**

### 翻译规范

- 参考 `glossary.json` 中的术语表，保持译名一致
- 保留所有标记标签，不翻译标签内容：

| 标记 | 含义 |
|------|------|
| `[m:happy]` | 角色表情 |
| `[s:1.5]` | 文字缩放 |
| `[b]...[/b]` | 粗体 |
| `[i]...[/i]` | 斜体 |
| `[w:500]` | 等待（毫秒） |
| `{catname}` | 动态变量（猫名） |
| `{his}` `{he}` | 代词变量 |
| `&nbsp;` | 不换行空格 |

- 译文要自然流畅，符合中文游戏玩家的阅读习惯
- 如果原文只是标点或无需翻译，保持原样

## 开发者指南

将本仓库克隆到游戏目录（`Steam/steamapps/common/Mewgenics/`），安装 [uv](https://docs.astral.sh/uv/)，然后按以下流程操作。

### 完整工作流

```bash
uv sync  # 初始化项目和依赖
uv run main.py extract        # 1. 解包资源文件
uv run main.py add-zh-column  # 2. 为 CSV 添加 zh 列
uv run main.py translate      # 3. AI 自动翻译（需要 API Key）
uv run main.py pack           # 4. 重新打包
uv run main.py apply          # 5. 应用补丁
```

### 命令详解

#### `extract` — 解包资源文件

从 `resources.gpak` 解包全部 18524 个文件到 `extracted/` 目录（约 4.7 GB）。同时生成两个元数据文件供 `pack` 使用：

- `extracted/__gpak_header.bin` — GPAK 文件头（4 字节）
- `extracted/__gpak_index.txt` — 文件名列表（保持原始顺序）

```bash
uv run main.py extract
```

#### `add-zh-column` — 添加中文列

为 `extracted/data/text/` 下所有 CSV 文件的每一行末尾追加空的 `zh` 列。如果 `zh` 列已存在则跳过。对 `additions.csv` 还会自动设置语言元数据（`CURRENT_LANGUAGE_NAME` → `中文`）。

```bash
uv run main.py add-zh-column
```

#### `translate` — AI 翻译

调用通义千问（qwen3-max）将英文文本翻译为中文，写入 CSV 的 `zh` 列。翻译进度保存在 `translation_progress.json`，中断后可续译。20 线程并发调用 API，翻译时参考 `glossary.json` 术语表保证译名一致。

需要先设置 API Key：

```bash
export DASHSCOPE_API_KEY="your-api-key"        # Linux/macOS
set DASHSCOPE_API_KEY=your-api-key              # Windows CMD
$env:DASHSCOPE_API_KEY="your-api-key"           # PowerShell
```

```bash
uv run main.py translate                           # 翻译全部
uv run main.py translate --dry                     # 仅查看进度，不翻译
uv run main.py translate --files events.csv        # 只翻译指定文件
uv run main.py translate --batch-size 100          # 每批 100 条
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dry` | — | 仅显示翻译进度统计，不调用 API |
| `--files FILE [FILE ...]` | 全部 19 个 CSV | 只翻译指定的 CSV 文件 |
| `--batch-size N` | 50 | 每次 API 调用包含的文本条数 |

#### `pack` — 重新打包

将 `extracted/` 目录按原始索引顺序重新打包为 `resources_patched.gpak`。未解包的文件会从原始 `resources.gpak` 中读取。

```bash
uv run main.py pack
```

#### `apply` — 应用补丁

将 `resources_patched.gpak` 替换为 `resources.gpak`。首次执行时自动备份原文件为 `resources.gpak.bak`。

```bash
uv run main.py apply
```

#### `info` — 查看 GPAK 信息

显示 `resources.gpak` 的文件数量、大小、文件类型分布、CSV 文件列表等。

```bash
uv run main.py info
```

### 替换中文字体

游戏自带 `unicodefont.swf` 作为 CJK fallback 字体。如果觉得默认字体不好看，可以替换：

```bash
uv run replace_unicode_font.py                    # 用默认 TTF 替换
uv run replace_unicode_font.py --font other.ttf   # 用指定 TTF 替换
uv run replace_unicode_font.py --restore           # 恢复原始字体
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--font FILE` | `MaoKenZhuYuanTi-MaokenZhuyuanTi-2.ttf` | 要嵌入的 TTF 字体文件 |
| `--dry` | — | 仅显示转换信息，不写入文件 |
| `--restore` | — | 从备份恢复原始字体 |

替换后需要重新 `pack` 和 `apply` 才能在游戏中生效。

## 汉化目标文件

所有游戏文本存放在 `data/text/` 下的 19 个 CSV 文件中（共约 6.4 MB），编码为 UTF-8 BOM：

| 文件 | 大小 | 内容 |
|------|------|------|
| `events.csv` | 1.7 MB | 随机事件文本（战斗、探索、剧情事件） |
| `npc_dialog.csv` | 1.4 MB | NPC 对话 |
| `abilities.csv` | 1.0 MB | 主动技能名称和效果描述 |
| `items.csv` | 680 KB | 物品名称和描述 |
| `passives.csv` | 596 KB | 被动技能名称和效果描述 |
| `units.csv` | 271 KB | 单位/角色名称和描述 |
| `keyword_tooltips.csv` | 146 KB | 游戏关键词的工具提示说明 |
| `cutscene_text.csv` | 134 KB | 过场动画文本 |
| `furniture.csv` | 110 KB | 家具名称和描述 |
| `misc.csv` | 106 KB | 杂项文本（UI、地名、系统提示等） |
| `mutations.csv` | 91 KB | 猫咪变异名称和效果描述 |
| `progression.csv` | 54 KB | 游戏进度相关（解锁提示、成就等） |
| `enemy_abilities.csv` | 46 KB | 敌人技能名称和描述 |
| `additions.csv` | 41 KB | 追加文本（含语言元数据） |
| `weather.csv` | 39 KB | 天气名称和描述 |
| `teamnames.csv` | 34 KB | 队伍名称 |
| `additions2.csv` | 20 KB | 追加文本 2 |
| `pronouns.csv` | 3 KB | 代词系统（动态替换角色性别代词） |
| `additions3.csv` | 0.6 KB | 追加文本 3 |

运行 `uv run main.py translate --dry` 查看各文件翻译完成度。

## 项目文件

| 文件 | 说明 |
|------|------|
| `main.py` | CLI 入口（解包/打包/应用补丁） |
| `translate.py` | AI 翻译逻辑（批量翻译、术语提取、进度管理） |
| `ai.py` | AI API 封装（通义千问 qwen3-max） |
| `glossary.json` | 术语表（英文 → 中文映射，翻译时强制使用） |
| `translation_progress.json` | 翻译数据（`文件名::KEY` → 中文译文，社区协作核心文件） |
| `replace_unicode_font.py` | 替换 `unicodefont.swf` 中的 CJK fallback 字体 |

## 依赖

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- `openai`（AI 翻译时需要）
- `fonttools`（替换字体时需要）

## 许可

本项目为社区汉化工具，不包含任何游戏资源文件。
