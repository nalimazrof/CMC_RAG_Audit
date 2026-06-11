# CMC RAG Audit

基于检索增强生成（RAG）技术的 CMC 文件自动化合规审计系统。

## 项目简介

本项目旨在利用 RAG 技术，对 CMC（Chemistry, Manufacturing and Controls）相关文件进行自动化审计与合规性检查。系统通过向量检索与大语言模型相结合，实现对文件的智能分析、合规性验证与审计报告生成。

## 功能特性

- 📄 CMC 文件自动解析与向量化
- 🔍 基于 RAG 的智能检索与问答
- ✅ 合规性规则自动比对
- 📊 审计报告自动生成
- 🗄️ 本地向量数据库存储（不上传敏感文件）

## 技术栈

- **Python 3.12+**
- **uv**（依赖与虚拟环境管理）
- **LangChain / LlamaIndex**（RAG 框架，待定）
- **ChromaDB / FAISS**（向量数据库，待定）
- **OpenAI / Local LLM**（大语言模型，待定）

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd CMC_RAG_Audit
```

### 2. 创建并激活虚拟环境

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填写 API 密钥等配置
```

### 5. 运行

```bash
python main.py
```

## 目录结构

```
CMC_RAG_Audit/
├── .venv/              # 虚拟环境（不提交）
├── data/               # 本地测试数据（不提交：*.pdf, *.db）
├── src/                # 核心源代码
│   ├── ingest/         # 文件解析与向量化
│   ├── retrieval/      # RAG 检索模块
│   ├── audit/          # 合规审计逻辑
│   └── report/         # 报告生成
├── tests/              # 单元测试
├── .env                # 环境变量（不提交）
├── .gitignore
├── README.md
└── requirements.txt    # 依赖清单（待添加）
```

## 注意事项

> ⚠️ **安全提示**：本地测试用的 PDF 文件（`*.pdf`）、数据库文件（`*.db`）及 `.env` 配置文件均已加入 `.gitignore`，**请勿手动将敏感数据推送至远程仓库**。

## 贡献指南

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "feat: 添加新功能"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## License

MIT License
