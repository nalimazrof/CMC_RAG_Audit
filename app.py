"""
CMC RAG Audit — Streamlit 前端  v2.0
=======================================
双文档参照审计界面：
  - 左侧侧边栏：上传 Protocol PDF（质量标准）+ Report PDF（稳定性报告）
  - 主界面：双文档比对 → Gemma4 推理 → OOS/OOT 高亮警示

启动方式：
    .venv/Scripts/streamlit.exe run app.py
"""

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── 后端导入 ──────────────────────────────────────────────────
from rag_audit import (
    AUDIT_MODEL,
    EMBED_MODEL,
    COLLECTION_PROTOCOL,
    COLLECTION_REPORT,
    build_audit_prompt,
    build_dual_audit_prompt,
    get_chroma_collection,
    load_pdf,
    ollama_generate_async,
    retrieve_relevant_chunks,
    split_into_chunks,
    upsert_chunks_to_chroma,
)

# ─────────────────────────────────────────────────────────────
# 页面全局配置
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CMC RAG Audit System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS —— 修复侧边栏对比度 + 主题风格
# config.toml 已设置全局 dark theme；这里用精细 CSS 覆盖剩余盲区
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ━━━ 全局字体 ━━━ */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* ━━━ 侧边栏：强制高对比度白色文字 ━━━ */
section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
}
section[data-testid="stSidebar"] * {
    color: #e8edf7 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #e8edf7 !important;
}
/* 侧边栏 info 框（模型配置）*/
section[data-testid="stSidebar"] .stAlert,
section[data-testid="stSidebar"] .stAlert * {
    background-color: #162032 !important;
    color: #a8d4ff !important;
    border-color: #2d5a9e !important;
}
/* 侧边栏 file uploader 区域 */
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background-color: #161c2d !important;
    border-color: #2d5a9e !important;
    color: #e8edf7 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {
    color: #c8d6f0 !important;
}
/* 侧边栏分隔线 */
section[data-testid="stSidebar"] hr {
    border-color: #2d3a54 !important;
}
/* 侧边栏 caption */
section[data-testid="stSidebar"] .stCaption p {
    color: #8fa8d4 !important;
}

/* ━━━ 顶部 Hero 横幅 ━━━ */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2e5c 60%, #0d3b6e 100%);
    border-radius: 14px;
    padding: 28px 36px 22px;
    margin-bottom: 24px;
    border: 1px solid #1e3a6e;
}
.hero-banner h1 { color: #e8f4ff; font-size: 1.85rem; margin: 0 0 6px; }
.hero-banner p  { color: #7eb8f7; margin: 0; font-size: 0.95rem; }

/* ━━━ 模式徽章 ━━━ */
.mode-badge-dual {
    display: inline-block;
    background: #0a3d62;
    color: #5dade2;
    border: 1px solid #1a5276;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}
.mode-badge-single {
    display: inline-block;
    background: #1a3a1a;
    color: #58d68d;
    border: 1px solid #1e8449;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}

/* ━━━ 信息卡片 ━━━ */
.info-card {
    background: #111827;
    border: 1px solid #1e3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card h3 { color: #7eb8f7; margin: 0 0 10px; font-size: 1.0rem; }
.info-card p  { color: #c8d6f0; margin: 0; line-height: 1.65; font-size: 0.93rem; }

/* ━━━ 文件状态卡片 ━━━ */
.file-card {
    background: #0d1a2e;
    border: 1px solid #1e4080;
    border-left: 4px solid #3b7dd8;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 10px;
    color: #a8c8f0;
    font-size: 0.9rem;
}
.file-card strong { color: #e8f4ff; }

/* ━━━ OOS 警告横幅 ━━━ */
.alert-oos {
    background: #2d0a0a;
    border-left: 5px solid #c62828;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #ff8a80;
    font-weight: 600;
    font-size: 0.95rem;
}
.alert-oos .alert-title { font-size: 1.1rem; margin-bottom: 4px; }

/* ━━━ OOT 警告横幅 ━━━ */
.alert-oot {
    background: #2d1a00;
    border-left: 5px solid #e65100;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #ffcc80;
    font-weight: 600;
    font-size: 0.95rem;
}
.alert-oot .alert-title { font-size: 1.1rem; margin-bottom: 4px; }

/* ━━━ 检索来源标签 ━━━ */
.source-tag {
    display: inline-block;
    border-radius: 20px;
    font-size: 0.77rem;
    padding: 3px 11px;
    margin: 2px 3px;
    font-weight: 500;
}
.source-tag-proto  { background: #0a3d62; color: #5dade2; border: 1px solid #1a5276; }
.source-tag-report { background: #1a3a1a; color: #58d68d; border: 1px solid #1e8449; }
.source-tag-page   { background: #1a1a3a; color: #9b59b6; border: 1px solid #6c3483; }

/* ━━━ 审计结论正文 ━━━ */
.conclusion-body {
    background: #0a0f1a;
    border: 1px solid #1e3a5c;
    border-radius: 10px;
    padding: 24px 28px;
    color: #d0dff8;
    line-height: 1.85;
    white-space: pre-wrap;
    font-size: 0.93rem;
}

/* ━━━ 步骤徽章 ━━━ */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #0d3b6e;
    color: #7eb8f7;
    border-radius: 50%;
    width: 26px; height: 26px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 8px;
}

/* ━━━ 进度条蓝色 ━━━ */
.stProgress > div > div > div { background-color: #3b7dd8 !important; }

/* ━━━ 主区域标题颜色 ━━━ */
h2, h3 { color: #a8c8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

OOS_KEYWORDS = [
    "OOS", "超标", "不合格", "out of specification",
    "failed", "failure", "不符合规定", "超出限度", "超出 Protocol",
]
OOT_KEYWORDS = [
    "OOT", "不良趋势", "out of trend", "降解趋势",
    "上升趋势", "下降趋势", "异常趋势", "持续下降", "持续上升",
    "潜在风险", "警戒", "CAPA",
]


def detect_alerts(text: str) -> tuple[bool, bool]:
    t = text.lower()
    return (
        any(kw.lower() in t for kw in OOS_KEYWORDS),
        any(kw.lower() in t for kw in OOT_KEYWORDS),
    )


def highlight_text(text: str) -> str:
    """高亮 OOS（红色）和 OOT（橙色）关键词。"""
    import html as _html
    safe = _html.escape(text)
    for kw in OOS_KEYWORDS:
        safe = re.compile(re.escape(kw), re.IGNORECASE).sub(
            lambda m: (
                f'<mark style="background:#c62828;color:#fff;'
                f'border-radius:3px;padding:1px 4px;font-weight:600;">'
                f'{m.group()}</mark>'
            ),
            safe,
        )
    for kw in OOT_KEYWORDS:
        safe = re.compile(re.escape(kw), re.IGNORECASE).sub(
            lambda m: (
                f'<mark style="background:#e65100;color:#fff;'
                f'border-radius:3px;padding:1px 4px;font-weight:600;">'
                f'{m.group()}</mark>'
            ),
            safe,
        )
    return safe


def run_async(coro):
    """Streamlit 线程内安全运行 async 协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def save_tmp_pdf(uploaded) -> str:
    """将 UploadedFile 写入临时文件，返回路径。"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.getbuffer())
        return f.name


# ─────────────────────────────────────────────────────────────
# session_state 初始化（持久化入库状态，跨 Streamlit 刷新保持）
# ─────────────────────────────────────────────────────────────
if "proto_indexed_count" not in st.session_state:
    st.session_state["proto_indexed_count"]  = 0   # Protocol 集合已入库条数
if "report_indexed_count" not in st.session_state:
    st.session_state["report_indexed_count"] = 0   # Report 集合已入库条数
if "last_proto_name" not in st.session_state:
    st.session_state["last_proto_name"]  = ""      # 上次入库的 Protocol 文件名
if "last_report_name" not in st.session_state:
    st.session_state["last_report_name"] = ""      # 上次入库的 Report 文件名


# ─────────────────────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CMC RAG Audit")
    st.markdown("*v2.0 · 双文档参照审计*")
    st.markdown("---")

    # ── 文件上传区 ───────────────────────────────────────────
    st.markdown("### 📋 文件 ① — Protocol（质量标准）")
    st.markdown(
        '<p style="color:#8fa8d4;font-size:0.82rem;margin:-8px 0 6px;">'
        '包含 Acceptance Criteria / 验收标准的稳定性研究方案</p>',
        unsafe_allow_html=True,
    )
    protocol_file = st.file_uploader(
        label="上传 Protocol PDF",
        type=["pdf"],
        key="protocol_uploader",
        help="例：ICH Q1A 稳定性研究方案、成品质量标准文件",
    )

    st.markdown("---")

    st.markdown("### 📄 文件 ② — Report（稳定性报告）")
    st.markdown(
        '<p style="color:#8fa8d4;font-size:0.82rem;margin:-8px 0 6px;">'
        '包含实际检测数据的稳定性研究报告</p>',
        unsafe_allow_html=True,
    )
    report_file = st.file_uploader(
        label="上传稳定性报告 PDF",
        type=["pdf"],
        key="report_uploader",
        help="例：长期 / 加速稳定性数据报告",
    )

    st.markdown("---")

    # ── 审计问题 ─────────────────────────────────────────────
    st.markdown("### ❓ 审计问题")
    default_query = (
        "请将稳定性报告中的实际检测数据与 Protocol 规定的验收标准逐项比对，"
        "明确指出哪些数据存在 OOS（超标）或 OOT（不良趋势）风险，"
        "并给出 CAPA 改进建议。"
    )
    audit_query = st.text_area(
        "输入审计问题（支持中英文）",
        value=default_query,
        height=130,
    )

    st.markdown("---")

    # ── 模型信息 ─────────────────────────────────────────────
    st.markdown("### ⚙️ 模型配置")
    st.info(
        f"**Embedding**  `{EMBED_MODEL}`\n\n"
        f"**推理模型**  `{AUDIT_MODEL}`\n\n"
        f"**后端**  Ollama @ localhost:11434"
    )

    st.markdown("---")
    run_btn = st.button(
        "🚀 开始自动审计",
        use_container_width=True,
        type="primary",
    )
    st.markdown("---")
    st.caption("CMC RAG Audit System v2.0  ·  Powered by Ollama + ChromaDB")

    # ── ChromaDB 持久化状态（跨刷新保持）─────────────────────
    proto_cnt  = st.session_state.get("proto_indexed_count",  0)
    report_cnt = st.session_state.get("report_indexed_count", 0)
    if proto_cnt > 0 or report_cnt > 0:
        st.markdown("---")
        st.markdown("### 🗄️ ChromaDB 入库状态")
        st.markdown(
            f'<div style="background:#0d1a2e;border:1px solid #1e4080;border-radius:8px;'
            f'padding:10px 14px;font-size:0.82rem;color:#a8c8f0;">'
            f'📋 Protocol：<strong style="color:#5dade2;">{proto_cnt} 条</strong><br>'
            f'📄 Report：<strong style="color:#58d68d;">{report_cnt} 条</strong><br>'
            f'<span style="color:#6a7fa8;font-size:0.76rem;">持久化至 ./chroma_db</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# 主界面 — Hero
# ─────────────────────────────────────────────────────────────
dual_mode = protocol_file is not None and report_file is not None
mode_badge = (
    '<span class="mode-badge-dual">⚡ 双文档对比模式</span>'
    if dual_mode else
    '<span class="mode-badge-single">📄 单文档 / 演示模式</span>'
)
st.markdown(
    f"""<div class="hero-banner">
  <h1>🔬 CMC 稳定性报告智能审计系统 {mode_badge}</h1>
  <p>Protocol 质量标准 ⟷ 稳定性报告实际数据 · 本地 Gemma4 逐项比对 · OOS / OOT 自动识别 · 完全离线运行</p>
</div>""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 主界面 — 等待状态引导
# ─────────────────────────────────────────────────────────────
if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📋", "Step 1 · 上传 Protocol",
         "在左侧上传含有 Acceptance Criteria 的稳定性研究方案 PDF。"),
        ("📄", "Step 2 · 上传 Report",
         "上传包含实际检测数据的稳定性报告 PDF（不上传则用演示数据）。"),
        ("🔍", "Step 3 · 语义检索",
         "系统对两份文档分别向量化，并分别检索与审计问题最相关的段落。"),
        ("🤖", "Step 4 · 比对审计",
         "Gemma4 逐项对比实际数据与 Protocol 标准，标注 OOS / OOT 风险。"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f'<div class="info-card"><h3>{icon} {title}</h3>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    # 文件状态提示
    if protocol_file:
        st.markdown(
            f'<div class="file-card">✅ <strong>Protocol 已上传：</strong>'
            f'{protocol_file.name} &nbsp;({protocol_file.size/1024:.1f} KB)</div>',
            unsafe_allow_html=True,
        )
    if report_file:
        st.markdown(
            f'<div class="file-card">✅ <strong>稳定性报告已上传：</strong>'
            f'{report_file.name} &nbsp;({report_file.size/1024:.1f} KB)</div>',
            unsafe_allow_html=True,
        )
    if not protocol_file and not report_file:
        st.info("👈 在左侧上传两份 PDF 开启双文档对比审计，或直接点击【开始自动审计】使用内置演示数据。")

# ─────────────────────────────────────────────────────────────
# 主界面 — 执行审计管道
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not audit_query.strip():
        st.warning("⚠️ 请先输入审计问题。")
        st.stop()

    st.markdown("---")
    st.markdown("### 🔄 审计进行中…")
    prog  = st.progress(0, text="初始化…")
    stat  = st.empty()

    # ── 演示数据（双文档）────────────────────────────────────
    DEMO_PROTOCOL = [
        {"chunk_id": "proto_001", "source": "demo_protocol.pdf", "page": 3,
         "text": (
             "【验收标准 — 含量测定】\n"
             "成品含量（HPLC 法）：标示量的 98.0%–102.0%。\n"
             "加速稳定性（40°C/75%RH，6 个月）：不低于标示量 97.0%。\n"
             "长期稳定性（25°C/60%RH，24 个月）：不低于标示量 97.0%。"
         )},
        {"chunk_id": "proto_002", "source": "demo_protocol.pdf", "page": 4,
         "text": (
             "【验收标准 — 降解产物】\n"
             "降解产物 A（已知杂质）：≤ 0.08%（24 个月长期）。\n"
             "降解产物 B（已知杂质）：≤ 0.05%（各时间点）。\n"
             "单个未知杂质：≤ 0.10%；总杂质：≤ 0.50%。"
         )},
        {"chunk_id": "proto_003", "source": "demo_protocol.pdf", "page": 5,
         "text": (
             "【验收标准 — 溶出度】\n"
             "Q ≥ 85%（45 min，pH 6.8 磷酸盐缓冲液，桨法 50 rpm）。\n"
             "长期稳定性各时间点均须满足上述标准。"
         )},
    ]
    DEMO_REPORT = [
        {"chunk_id": "report_001", "source": "demo_stability_report.pdf", "page": 7,
         "text": (
             "【含量测定结果】\n"
             "0M:101.2% | 6M:100.5% | 12M:99.1% | 18M:98.3% | 24M:96.8%\n"
             "注：24 个月结果 96.8% 低于 Protocol 规定下限 97.0%。"
         )},
        {"chunk_id": "report_002", "source": "demo_stability_report.pdf", "page": 8,
         "text": (
             "【降解产物检测结果】\n"
             "降解产物 A：0M:0.02% | 6M:0.03% | 12M:0.05% | 18M:0.07% | 24M:0.11%\n"
             "注：24 个月 0.11% 超出 Protocol 限度 0.08%（OOS）。\n"
             "降解产物 B：全程 ≤ 0.04%，符合规定。"
         )},
        {"chunk_id": "report_003", "source": "demo_stability_report.pdf", "page": 9,
         "text": (
             "【溶出度结果】\n"
             "0M:94.2% | 6M:92.1% | 12M:89.5% | 18M:88.0% | 24M:86.3%\n"
             "各时间点均满足 Q≥85%，但呈持续下降趋势（OOT 风险）。"
         )},
    ]

    # ─── Step 1: 解析 PDF ────────────────────────────────────
    stat.markdown('<span class="step-badge">1</span> 解析 PDF 文件…', unsafe_allow_html=True)
    prog.progress(8, text="PDF 解析中…")

    proto_tmp = report_tmp = None
    try:
        if protocol_file:
            proto_tmp  = save_tmp_pdf(protocol_file)
            proto_pages  = load_pdf(proto_tmp)
            proto_chunks = split_into_chunks(proto_pages, prefix="proto_")
            proto_label  = protocol_file.name
            is_demo_proto = False
        else:
            proto_chunks  = DEMO_PROTOCOL
            proto_label   = "demo_protocol.pdf（演示数据）"
            is_demo_proto = True

        if report_file:
            report_tmp    = save_tmp_pdf(report_file)
            report_pages  = load_pdf(report_tmp)
            report_chunks = split_into_chunks(report_pages, prefix="report_")
            report_label  = report_file.name
            is_demo_report = False
        else:
            report_chunks  = DEMO_REPORT
            report_label   = "demo_stability_report.pdf（演示数据）"
            is_demo_report = True

    finally:
        for p in [proto_tmp, report_tmp]:
            if p:
                Path(p).unlink(missing_ok=True)

    prog.progress(22, text="解析完成，开始向量化…")

    # ─── Step 2: 写入 ChromaDB ───────────────────────────────
    stat.markdown('<span class="step-badge">2</span> 向量化写入 ChromaDB…', unsafe_allow_html=True)

    proto_col  = get_chroma_collection(COLLECTION_PROTOCOL)
    report_col = get_chroma_collection(COLLECTION_REPORT)

    proto_existing  = set(proto_col.get()["ids"])  if proto_col.count()  > 0 else set()
    report_existing = set(report_col.get()["ids"]) if report_col.count() > 0 else set()

    new_proto  = [c for c in proto_chunks  if c["chunk_id"] not in proto_existing]
    new_report = [c for c in report_chunks if c["chunk_id"] not in report_existing]

    if new_proto:
        run_async(upsert_chunks_to_chroma(new_proto, proto_col))
    if new_report:
        run_async(upsert_chunks_to_chroma(new_report, report_col))

    prog.progress(50, text="向量化完成，开始语义检索…")

    # ─── Step 3: 分别检索 ────────────────────────────────────
    stat.markdown('<span class="step-badge">3</span> 双文档语义检索…', unsafe_allow_html=True)

    proto_retrieved  = retrieve_relevant_chunks(audit_query, proto_col)
    report_retrieved = retrieve_relevant_chunks(audit_query, report_col)
    prog.progress(68, text="检索完成，调用 Gemma4 比对审计…")

    # ── 入库后验证：任一集合仍为空则终止，给出明确提示 ──────────
    proto_count  = proto_col.count()
    report_count = report_col.count()

    # 更新 session_state（供调试与 UI 状态显示）
    st.session_state["proto_indexed_count"]  = proto_count
    st.session_state["report_indexed_count"] = report_count
    st.session_state["last_proto_name"]      = proto_label
    st.session_state["last_report_name"]     = report_label

    if proto_count == 0 or report_count == 0:
        prog.empty()
        stat.empty()
        empty_names = []
        if proto_count == 0:
            empty_names.append(f"Protocol（{proto_label}）")
        if report_count == 0:
            empty_names.append(f"Report（{report_label}）")
        st.error(
            f"❌ **ChromaDB 入库失败**：{' 和 '.join(empty_names)} 集合为空，"
            f"文档未成功向量化入库。\n\n"
            f"**可能原因**：\n"
            f"- Ollama 服务未启动（请确认 `ollama serve` 正在运行）\n"
            f"- `nomic-embed-text` 模型未拉取（运行 `ollama pull nomic-embed-text`）\n"
            f"- PDF 文件内容无法解析（文本型 PDF，非扫描件）\n\n"
            f"请修复后重新点击【开始自动审计】。"
        )
        st.stop()

    # ── 检索结果为空时降级提示（不阻断，Gemma 可用文本说明信息不足）──
    if not proto_retrieved:
        st.warning(f"⚠️ Protocol 集合（{proto_count} 条）中未检索到与审计问题相关的段落，审计结论可能不完整。")
    if not report_retrieved:
        st.warning(f"⚠️ Report 集合（{report_count} 条）中未检索到与审计问题相关的段落，审计结论可能不完整。")

    # ─── Step 4: Gemma4 生成结论 ─────────────────────────────
    stat.markdown(
        '<span class="step-badge">4</span> Gemma4 正在比对 Protocol 与报告数据（约 30–90 秒）…',
        unsafe_allow_html=True,
    )

    prompt     = build_dual_audit_prompt(audit_query, proto_retrieved, report_retrieved)
    conclusion = run_async(ollama_generate_async(prompt))

    prog.progress(100, text="✅ 审计完成！")
    stat.empty()
    time.sleep(0.3)
    prog.empty()

    # ─────────────────────────────────────────────────────────
    # 结果展示
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 审计结果")

    has_oos, has_oot = detect_alerts(conclusion)

    # ── 指标栏 ───────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Protocol",    proto_label[:22]  + ("…" if len(proto_label)  > 22 else ""))
    m2.metric("📄 Report",      report_label[:22] + ("…" if len(report_label) > 22 else ""))
    m3.metric("🔍 Protocol 匹配块", f"{len(proto_retrieved)} 块")
    m4.metric("🔴 OOS 超标",    "⚠️ 检测到" if has_oos else "✅ 未发现")
    m5.metric("🟡 OOT 趋势",    "⚠️ 检测到" if has_oot else "✅ 未发现")

    # ── 警示横幅 ─────────────────────────────────────────────
    if has_oos:
        st.markdown(
            '<div class="alert-oos">'
            '<div class="alert-title">🚨 OOS 超标警告 — Out of Specification</div>'
            '审计结论中发现数据超出 Protocol 规定的验收标准（Acceptance Criteria）。'
            '请立即启动 OOS 调查程序，追溯根本原因，并评估产品放行影响。'
            '</div>',
            unsafe_allow_html=True,
        )
    if has_oot:
        st.markdown(
            '<div class="alert-oot">'
            '<div class="alert-title">⚠️ OOT 不良趋势警告 — Out of Trend</div>'
            '审计结论中发现数据呈现不良趋势，当前虽未超标但存在风险。'
            '建议纳入持续稳定性监控（ONGOING）计划，评估剩余货架期影响，必要时启动 CAPA。'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 检索来源折叠区 ────────────────────────────────────────
    with st.expander("🔍 检索到的参照内容（Protocol vs Report）", expanded=False):
        col_p, col_r = st.columns(2)

        with col_p:
            st.markdown("**📋 Protocol 检索段落**")
            for i, c in enumerate(proto_retrieved, 1):
                st.markdown(
                    f'<span class="source-tag source-tag-proto">Protocol 片段 {i}</span>'
                    f'<span class="source-tag source-tag-page">第 {c["page"]} 页</span>'
                    f'<span class="source-tag source-tag-proto">相似度 {c["similarity"]:.3f}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(c["text"][:350] + ("…" if len(c["text"]) > 350 else ""))
                st.progress(int(c["similarity"] * 100))
                st.markdown("")

        with col_r:
            st.markdown("**📄 Report 检索段落**")
            for i, c in enumerate(report_retrieved, 1):
                st.markdown(
                    f'<span class="source-tag source-tag-report">Report 片段 {i}</span>'
                    f'<span class="source-tag source-tag-page">第 {c["page"]} 页</span>'
                    f'<span class="source-tag source-tag-report">相似度 {c["similarity"]:.3f}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(c["text"][:350] + ("…" if len(c["text"]) > 350 else ""))
                st.progress(int(c["similarity"] * 100))
                st.markdown("")

    # ── Gemma4 审计结论 ───────────────────────────────────────
    st.markdown("### 🤖 Gemma4 双文档比对审计结论")
    st.markdown(
        f'<div class="conclusion-body">{highlight_text(conclusion)}</div>',
        unsafe_allow_html=True,
    )

    # ── 下载报告 ─────────────────────────────────────────────
    st.markdown("---")
    report_json = json.dumps(
        {
            "query":            audit_query,
            "protocol_source":  proto_label,
            "report_source":    report_label,
            "audit_mode":       "dual" if (not is_demo_proto and not is_demo_report) else "demo",
            "protocol_chunks":  proto_retrieved,
            "report_chunks":    report_retrieved,
            "conclusion":       conclusion,
            "alerts":           {"OOS": has_oos, "OOT": has_oot},
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="⬇️ 下载完整审计报告（JSON）",
        data=report_json.encode("utf-8"),
        file_name="cmc_dual_audit_report.json",
        mime="application/json",
        use_container_width=True,
    )
