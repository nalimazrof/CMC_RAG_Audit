"""
CMC RAG Audit — Streamlit 前端
================================
启动方式：
    streamlit run app.py
或
    .venv/Scripts/streamlit.exe run app.py
"""

import asyncio
import json
import re
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── 导入后端管道 ──────────────────────────────────────────────
from rag_audit import (
    AUDIT_MODEL,
    EMBED_MODEL,
    build_audit_prompt,
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
# 自定义 CSS（现代深色卡片风格 + 警示色标）
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── 全局字体与背景 ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

/* ── 顶部标题栏 ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f36 0%, #0d3b6e 100%);
    border-radius: 14px;
    padding: 28px 36px 22px;
    margin-bottom: 24px;
    border: 1px solid #2a3a5c;
}
.hero-banner h1 { color: #e8f0fe; font-size: 1.9rem; margin: 0 0 6px; }
.hero-banner p  { color: #8fa8d4; margin: 0; font-size: 0.95rem; }

/* ── 信息卡片 ── */
.info-card {
    background: #1e2535;
    border: 1px solid #2d3a54;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card h3 { color: #a8c0f0; margin: 0 0 10px; font-size: 1.05rem; }
.info-card p  { color: #c8d6f0; margin: 0; line-height: 1.65; }

/* ── 检索来源标签 ── */
.source-tag {
    display: inline-block;
    background: #1b3a6b;
    color: #7eb8f7;
    border: 1px solid #2d5a9e;
    border-radius: 20px;
    font-size: 0.78rem;
    padding: 3px 12px;
    margin: 3px 4px;
}

/* ── OOS 警告横幅 ── */
.alert-oos {
    background: #3b0f0f;
    border-left: 5px solid #e53935;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ff8a80;
    font-weight: 600;
}

/* ── OOT 警告横幅 ── */
.alert-oot {
    background: #3b2900;
    border-left: 5px solid #fb8c00;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #ffcc80;
    font-weight: 600;
}

/* ── 普通结论正文 ── */
.conclusion-body {
    background: #161c2d;
    border: 1px solid #2a3650;
    border-radius: 10px;
    padding: 22px 26px;
    color: #d0dff8;
    line-height: 1.8;
    white-space: pre-wrap;
    font-size: 0.95rem;
}

/* ── 步骤进度 ── */
.step-badge {
    display: inline-block;
    background: #0d3b6e;
    color: #7eb8f7;
    border-radius: 50%;
    width: 26px; height: 26px;
    text-align: center;
    line-height: 26px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 8px;
}

/* ── 相似度进度条颜色覆盖 ── */
.stProgress > div > div > div { background-color: #3b7dd8 !important; }

/* ── 侧边栏 ── */
section[data-testid="stSidebar"] { background: #111827; }
section[data-testid="stSidebar"] h2 { color: #8fa8d4; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

# OOS / OOT 关键词（可按需扩充）
OOS_KEYWORDS = [
    "OOS", "超标", "不合格", "out of specification",
    "failed", "failure", "不符合规定", "超出限度",
]
OOT_KEYWORDS = [
    "OOT", "不良趋势", "out of trend", "降解趋势",
    "trend", "上升趋势", "下降趋势", "异常趋势",
    "潜在风险", "警戒", "注意",
]


def detect_alerts(text: str) -> tuple[bool, bool]:
    """返回 (has_oos, has_oot)"""
    t = text.lower()
    has_oos = any(kw.lower() in t for kw in OOS_KEYWORDS)
    has_oot = any(kw.lower() in t for kw in OOT_KEYWORDS)
    return has_oos, has_oot


def highlight_text(text: str) -> str:
    """
    在结论文本中，对 OOS / OOT 关键词进行 HTML 高亮。
    返回带 <mark> 标签的 HTML 字符串。
    """
    import html as html_lib

    safe = html_lib.escape(text)

    for kw in OOS_KEYWORDS:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        safe = pattern.sub(
            lambda m: f'<mark style="background:#e53935;color:#fff;'
                      f'border-radius:3px;padding:0 3px;">{m.group()}</mark>',
            safe,
        )
    for kw in OOT_KEYWORDS:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        safe = pattern.sub(
            lambda m: f'<mark style="background:#fb8c00;color:#fff;'
                      f'border-radius:3px;padding:0 3px;">{m.group()}</mark>',
            safe,
        )
    return safe


def run_async(coro):
    """在 Streamlit 线程中安全运行异步协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CMC RAG Audit")
    st.markdown("---")

    st.markdown("### 📂 上传审计文件")
    uploaded_file = st.file_uploader(
        label="拖拽或点击上传 CMC 稳定性报告 PDF",
        type=["pdf"],
        help="支持 ICH Q1A/Q1B/Q3A/Q3B 等 CMC 稳定性报告格式",
    )

    st.markdown("---")
    st.markdown("### ❓ 审计问题")
    default_query = (
        "请评估该稳定性报告中的降解产物控制策略、"
        "杂质限度设定和溶出度趋势是否存在 OOS 或 OOT 风险，"
        "并给出合规性结论。"
    )
    audit_query = st.text_area(
        "输入审计问题（支持中英文）",
        value=default_query,
        height=140,
        help="可自由输入任何针对 CMC 文件的合规性审计问题",
    )

    st.markdown("---")
    st.markdown("### ⚙️ 模型配置")
    st.info(
        f"**Embedding 模型**\n`{EMBED_MODEL}`\n\n"
        f"**推理模型**\n`{AUDIT_MODEL}`\n\n"
        f"**后端**\nOllama @ localhost:11434",
    )

    st.markdown("---")
    run_btn = st.button(
        "🚀 开始自动审计",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")
    st.caption("CMC RAG Audit System v0.1  |  Powered by Ollama + ChromaDB")

# ─────────────────────────────────────────────────────────────
# 主界面 — 顶部 Hero
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <h1>🔬 CMC 稳定性报告智能审计系统</h1>
  <p>基于本地 RAG 管道 · nomic-embed-text 语义检索 · Gemma4 推理分析 · 完全离线运行</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# 主界面 — 等待上传状态
# ─────────────────────────────────────────────────────────────
if not uploaded_file and not run_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """<div class="info-card">
            <h3>📄 Step 1 · 上传文件</h3>
            <p>在左侧侧边栏上传待审计的 CMC 稳定性报告 PDF 文件。</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="info-card">
            <h3>🔍 Step 2 · 语义检索</h3>
            <p>系统自动完成 PDF 解析 → 文本切块 → 向量化 → ChromaDB 语义检索。</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="info-card">
            <h3>🤖 Step 3 · 生成结论</h3>
            <p>本地 Gemma4 模型根据检索内容生成结构化合规审计结论，自动标记 OOS / OOT 风险。</p>
            </div>""",
            unsafe_allow_html=True,
        )
    st.info("👈 请先在左侧上传 PDF 文件，或直接点击【开始自动审计】使用内置演示数据。")

# ─────────────────────────────────────────────────────────────
# 主界面 — 已上传文件预览
# ─────────────────────────────────────────────────────────────
if uploaded_file:
    st.markdown(
        f"""<div class="info-card">
        <h3>✅ 文件已上传</h3>
        <p>文件名：<strong>{uploaded_file.name}</strong> &nbsp;·&nbsp;
           大小：<strong>{uploaded_file.size / 1024:.1f} KB</strong></p>
        </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# 主界面 — 执行审计管道
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not audit_query.strip():
        st.warning("⚠️ 请先输入审计问题。")
        st.stop()

    st.markdown("---")
    st.markdown("### 🔄 审计进行中…")

    # 进度占位
    progress_bar  = st.progress(0, text="初始化中…")
    status_holder = st.empty()

    # ── Step 1: PDF 处理 ──────────────────────────────────────
    status_holder.markdown(
        '<span class="step-badge">1</span> 解析 PDF 文件…',
        unsafe_allow_html=True,
    )
    progress_bar.progress(10, text="PDF 解析中…")

    if uploaded_file:
        # 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            pages  = load_pdf(tmp_path)
            chunks = split_into_chunks(pages)
            source_label = uploaded_file.name
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        # 演示模式：使用内置 CMC 文本
        status_holder.markdown(
            '<span class="step-badge">1</span> 未检测到上传文件，使用内置演示数据…',
            unsafe_allow_html=True,
        )
        demo_texts = [
            {
                "chunk_id": "demo_001",
                "text": (
                    "本品为口服固体制剂，活性成分为盐酸二甲双胍，规格 500mg/片。"
                    "原料药质量标准参照 USP 47 及 EP 11.0 执行，杂质限度符合 ICH Q3A 要求。"
                    "重金属检测：铅 ≤ 5 ppm，镉 ≤ 1 ppm，砷 ≤ 3 ppm，汞 ≤ 3 ppm。"
                ),
                "source": "demo_cmc.txt",
                "page":    1,
            },
            {
                "chunk_id": "demo_002",
                "text": (
                    "生产工艺为湿法制粒压片工艺。关键工艺参数（CPP）已通过工艺验证（3 批次）确认。"
                    "制粒终点判断采用水分在线检测，目标含水量 2.0%±0.5%。"
                    "压片机主压力控制范围：8–12 kN，转速 ≤ 35 rpm。"
                ),
                "source": "demo_cmc.txt",
                "page":    2,
            },
            {
                "chunk_id": "demo_003",
                "text": (
                    "成品质量标准：含量测定 98.0%–102.0%（HPLC 法），溶出度 Q≥85%（45 min，pH 6.8磷酸盐缓冲液）。"
                    "微生物限度：需氧菌总数 ≤ 10³ CFU/g，霉菌和酵母菌 ≤ 10² CFU/g，不得检出大肠埃希菌。"
                    "包装：双铝泡罩包装，防潮防光。稳定性：加速 6 个月及长期 24 个月数据均符合质量标准。"
                    "第 18 个月检测发现降解产物 A 出现上升趋势（OOT），当前值 0.09%，限度 0.10%，建议关注。"
                ),
                "source": "demo_cmc.txt",
                "page":    3,
            },
            {
                "chunk_id": "demo_004",
                "text": (
                    "变更控制：2023 年 Q3 对辅料羟丙基甲基纤维素（HPMC）供应商进行了变更。"
                    "已按 ICH Q10 要求完成变更评估，执行了对比溶出度研究（f2≥50），"
                    "并向监管机构提交了 CBE-30 补充申请，获批后方可实施。"
                    "第 24 月含量检测结果 97.2%，低于内控下限 97.5%，判定为 OOS，已启动偏差调查。"
                ),
                "source": "demo_cmc.txt",
                "page":    4,
            },
        ]
        chunks       = demo_texts   # 兼容后续逻辑
        source_label = "demo_cmc.txt（内置演示数据）"

    progress_bar.progress(25, text="文件解析完成，开始向量化…")

    # ── Step 2: 写入 ChromaDB ─────────────────────────────────
    status_holder.markdown(
        '<span class="step-badge">2</span> 向量化并写入 ChromaDB…',
        unsafe_allow_html=True,
    )

    collection   = get_chroma_collection()
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    if uploaded_file:
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    else:
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if new_chunks:
        run_async(upsert_chunks_to_chroma(new_chunks, collection))

    progress_bar.progress(55, text="向量化完成，开始语义检索…")

    # ── Step 3: 检索 ──────────────────────────────────────────
    status_holder.markdown(
        '<span class="step-badge">3</span> 语义检索相关段落…',
        unsafe_allow_html=True,
    )
    retrieved = retrieve_relevant_chunks(audit_query, collection)
    progress_bar.progress(70, text="检索完成，调用 Gemma4 生成结论…")

    # ── Step 4: 生成审计结论 ──────────────────────────────────
    status_holder.markdown(
        '<span class="step-badge">4</span> Gemma4 正在生成审计结论（约 30–90 秒）…',
        unsafe_allow_html=True,
    )
    prompt     = build_audit_prompt(audit_query, retrieved)
    conclusion = run_async(ollama_generate_async(prompt))
    progress_bar.progress(100, text="✅ 审计完成！")
    status_holder.empty()
    time.sleep(0.3)
    progress_bar.empty()

    # ─────────────────────────────────────────────────────────
    # 结果展示区
    # ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 审计结果")

    # ── 顶部指标 ────────────────────────────────────────────
    has_oos, has_oot = detect_alerts(conclusion)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 来源文件",     source_label[:30] + ("…" if len(source_label) > 30 else ""))
    m2.metric("📦 检索到文本块", f"{len(retrieved)} 块")
    m3.metric("🔴 OOS 风险",     "⚠️ 检测到" if has_oos else "✅ 未发现")
    m4.metric("🟡 OOT 趋势",     "⚠️ 检测到" if has_oot else "✅ 未发现")

    # ── 警示横幅 ────────────────────────────────────────────
    if has_oos:
        st.markdown(
            '<div class="alert-oos">🚨 OOS 警告 — 审计结论中检测到超标（Out of Specification）相关内容，'
            '请立即核查并启动偏差调查程序。</div>',
            unsafe_allow_html=True,
        )
    if has_oot:
        st.markdown(
            '<div class="alert-oot">⚠️ OOT 警告 — 审计结论中检测到不良趋势（Out of Trend）相关内容，'
            '建议纳入持续稳定性监控计划并评估影响。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 检索来源卡片 ─────────────────────────────────────────
    with st.expander("🔍 检索到的相关文档段落", expanded=False):
        for i, chunk in enumerate(retrieved, 1):
            sim_pct = int(chunk["similarity"] * 100)
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<span class="source-tag">来源 {i}</span>'
                    f'<span class="source-tag">📄 {chunk["source"]}</span>'
                    f'<span class="source-tag">第 {chunk["page"]} 页</span>',
                    unsafe_allow_html=True,
                )
                st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))
            with col_b:
                st.markdown(f"**相似度**")
                st.progress(sim_pct, text=f"{chunk['similarity']:.3f}")
            st.markdown("---")

    # ── Gemma4 审计结论（高亮关键词）────────────────────────
    st.markdown("### 🤖 Gemma4 审计结论")
    highlighted_html = highlight_text(conclusion)
    st.markdown(
        f'<div class="conclusion-body">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )

    # ── 下载按钮 ─────────────────────────────────────────────
    st.markdown("---")
    report_data = json.dumps(
        {
            "query":      audit_query,
            "pdf_source": source_label,
            "retrieved":  retrieved,
            "conclusion": conclusion,
            "alerts":     {"OOS": has_oos, "OOT": has_oot},
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="⬇️ 下载审计报告 JSON",
        data=report_data.encode("utf-8"),
        file_name="cmc_audit_report.json",
        mime="application/json",
        use_container_width=True,
    )
