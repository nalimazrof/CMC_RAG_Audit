"""
CMC RAG Audit Pipeline  v2.0
==============================
双文档参照审计闭环：
  Protocol PDF（质量标准/验收标准）→ 向量化为参照基准集合
  Report   PDF（稳定性报告/实际数据）→ 向量化为结果集合
  Gemma4 Prompt → 逐项比对，精准指出 OOS / OOT 风险

全程调用本地 Ollama (http://localhost:11434)，无需任何云端 API Key。
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import requests

# ─────────────────────────────────────────────
# 配置区（按需修改）
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"   # 专用 embedding 模型
AUDIT_MODEL     = "gemma4:latest"      # 推理模型

CHROMA_PATH            = "./chroma_db"
COLLECTION_PROTOCOL    = "cmc_protocol"    # Protocol（质量标准）集合
COLLECTION_REPORT      = "cmc_report"      # 稳定性报告（实际数据）集合

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80
TOP_K         = 4


# ─────────────────────────────────────────────
# 1. PDF 解析与结构化切块
# ─────────────────────────────────────────────

def load_pdf(pdf_path: str) -> list[dict]:
    """
    用 PyMuPDF 逐页解析 PDF，返回带页码元数据的段落列表。
    每个元素：{"text": "...", "page": 3, "source": "xxx.pdf"}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw     = page.get_text("text")
        cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()
        if cleaned:
            pages.append({
                "text":   cleaned,
                "page":   page_num,
                "source": path.name,
            })
    doc.close()
    print(f"[PDF] 解析完成：{path.name}，共 {len(pages)} 页有效内容")
    return pages


def split_into_chunks(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
    prefix: str     = "",          # 用于生成唯一 chunk_id，区分 protocol / report
) -> list[dict]:
    """
    滑动窗口切块，保留 source / page 元数据。
    prefix 用于区分两类文档（避免 chunk_id 冲突）。
    """
    chunks = []
    for page_info in pages:
        text   = page_info["text"]
        source = page_info["source"]
        page   = page_info["page"]

        start = 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                cid = f"{prefix}{source}_p{page}_c{len(chunks)}"
                chunks.append({
                    "text":     chunk,
                    "source":   source,
                    "page":     page,
                    "chunk_id": cid,
                })
            start += chunk_size - overlap

    print(f"[切块] [{prefix or 'default'}] 共 {len(chunks)} 块 (size={chunk_size}, overlap={overlap})")
    return chunks


# ─────────────────────────────────────────────
# 2. 本地 Ollama 接口
# ─────────────────────────────────────────────

def ollama_embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """调用 Ollama /api/embeddings，返回向量。"""
    url  = f"{OLLAMA_BASE_URL}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if "embedding" not in data:
        raise ValueError(f"Ollama 未返回 embedding 字段：{data}")
    return data["embedding"]


def ollama_generate(prompt: str, model: str = AUDIT_MODEL, stream: bool = False) -> str:
    """调用 Ollama /api/generate，返回完整文本（非流式）。"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": stream},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


async def ollama_embed_async(text: str, model: str = EMBED_MODEL) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ollama_embed, text, model)


async def ollama_generate_async(prompt: str, model: str = AUDIT_MODEL) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ollama_generate, prompt, model)


# ─────────────────────────────────────────────
# 3. ChromaDB 向量存储（支持命名集合）
# ─────────────────────────────────────────────

def get_chroma_collection(
    collection_name: str,
    persist_path: str = CHROMA_PATH,
):
    """初始化 ChromaDB 持久化集合（cosine 距离）。"""
    client = chromadb.PersistentClient(path=persist_path)
    col    = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[ChromaDB] 集合 '{collection_name}' 已有 {col.count()} 条记录")
    return col


async def upsert_chunks_to_chroma(
    chunks: list[dict],
    collection,
    batch_size: int = 8,
) -> None:
    """异步批量向量化并 upsert 到 ChromaDB。"""
    total = len(chunks)
    print(f"[Embedding] 向量化 {total} 块，模型：{EMBED_MODEL}")

    for batch_start in range(0, total, batch_size):
        batch      = chunks[batch_start: batch_start + batch_size]
        embeddings = await asyncio.gather(*[ollama_embed_async(c["text"]) for c in batch])

        collection.upsert(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = embeddings,
            documents  = [c["text"]     for c in batch],
            metadatas  = [{"source": c["source"], "page": c["page"]} for c in batch],
        )
        print(f"  → [{min(batch_start + batch_size, total)}/{total}] 已写入")

    print(f"[ChromaDB] 写入完成，集合现有 {collection.count()} 条")


# ─────────────────────────────────────────────
# 4. 语义检索
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    collection,
    top_k: int = TOP_K,
) -> list[dict]:
    """余弦相似度检索，返回 top_k 块。"""
    query_vec = ollama_embed(query)
    results   = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({
            "text":       doc,
            "source":     meta.get("source", ""),
            "page":       meta.get("page", "?"),
            "similarity": round(1 - dist, 4),
        })
    return out


# ─────────────────────────────────────────────
# 5. 双文档审计 Prompt
# ─────────────────────────────────────────────

DUAL_AUDIT_PROMPT_TEMPLATE = """\
你是一位资深 CMC（Chemistry, Manufacturing and Controls）法规合规审计专家，熟悉 ICH Q1A、Q1B、Q3A、Q3B、Q10 等指南。

以下提供了两类文档内容，请严格按照任务要求进行逐项比对审计。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【文档 A — 稳定性研究 Protocol（质量标准 / 验收标准）】
以下是从 Protocol 中检索到的相关验收标准段落：

{protocol_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【文档 B — 稳定性报告（实际检测数据）】
以下是从稳定性报告中检索到的实际测试结果段落：

{report_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【审计任务】
{query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【审计要求（请严格执行以下每一项）】

1. **逐项数据比对**
   - 提取文档 B（稳定性报告）中所有可量化的检测数据（含量、溶出度、杂质、微生物限度等）。
   - 将每项数据与文档 A（Protocol）中对应的 Acceptance Criteria（验收标准/限度）逐一对比。
   - 以表格或列表形式输出：检测项目 | Protocol 规定限度 | 实际检测值 | 比对结论。

2. **OOS 判定（Out of Specification，超标）**
   - 明确指出哪些数据点超出 Protocol 规定的限度范围。
   - 对每个 OOS 项目标注：OOS — [检测项目]，实测值 [X]，Protocol 限度 [Y]，超标幅度 [Z]。

3. **OOT 判定（Out of Trend，不良趋势）**
   - 识别随时间变化呈现单向劣化趋势（持续上升/下降）的数据，即使尚未超标也需标注。
   - 对每个 OOT 项目标注：OOT — [检测项目]，趋势描述，当前值与限度的余量。

4. **合规性总结**
   - 给出整体合规性结论：合规 / 存在缺陷 / 严重不合规。
   - 列出主要发现（Findings）及监管风险等级（高 / 中 / 低）。

5. **改进建议**
   - 针对每个 OOS / OOT 项目，给出具体的纠正和预防措施（CAPA）建议。

6. **证据溯源**
   - 每项结论须注明证据来源（文件名 + 页码）。

请用**中文**回复，格式清晰，使用标题和列表，语言专业简洁。

【审计结论】
"""


def build_dual_audit_prompt(
    query: str,
    protocol_chunks: list[dict],
    report_chunks: list[dict],
) -> str:
    """构造双文档对比审计 Prompt。"""

    def _format_chunks(chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[片段 {i}] 来源：{c['source']} | 第 {c['page']} 页 | 相似度：{c['similarity']}\n"
                f"{c['text']}"
            )
        return "\n\n---\n\n".join(parts) if parts else "（未检索到相关内容）"

    return DUAL_AUDIT_PROMPT_TEMPLATE.format(
        query            = query,
        protocol_context = _format_chunks(protocol_chunks),
        report_context   = _format_chunks(report_chunks),
    )


# ─────────────────────────────────────────────
# 6. 单文档兼容 Prompt（无 Protocol 时降级使用）
# ─────────────────────────────────────────────

SINGLE_AUDIT_PROMPT_TEMPLATE = """\
你是一位资深 CMC（Chemistry, Manufacturing and Controls）法规合规审计专家。
请根据以下从 CMC 文件中检索到的相关内容，对审计问题进行分析并给出结论。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【审计问题】
{query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【检索到的相关文档内容】
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【审计要求】
1. 判断文档内容是否充分支持审计问题的回答。
2. 给出明确合规性结论（合规 / 存在缺陷 / 信息不足）。
3. 识别并标注任何 OOS（超标）或 OOT（不良趋势）风险。
4. 每项结论注明证据来源（文件名 + 页码）。
5. 给出改进建议和 CAPA 措施。

请用中文回复，语言简洁专业。

【审计结论】
"""


def build_audit_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """单文档审计 Prompt（向后兼容原有接口）。"""
    parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        parts.append(
            f"[来源 {i}] 文件：{c['source']} | 第 {c['page']} 页 | 相似度：{c['similarity']}\n"
            f"{c['text']}"
        )
    context = "\n\n---\n\n".join(parts)
    return SINGLE_AUDIT_PROMPT_TEMPLATE.format(query=query, context=context)


# ─────────────────────────────────────────────
# 7. 双文档审计管道（新核心入口）
# ─────────────────────────────────────────────

async def run_dual_audit_pipeline(
    protocol_pdf: str,
    report_pdf: str,
    audit_query: str,
) -> dict:
    """
    双文档 RAG 审计管道：
      Protocol PDF → 向量化 → cmc_protocol 集合（参照基准）
      Report   PDF → 向量化 → cmc_report   集合（实际数据）
      → 分别检索 → Gemma4 逐项比对 → 输出 OOS/OOT 结论

    Returns:
        {query, protocol_source, report_source, protocol_chunks,
         report_chunks, prompt, conclusion}
    """
    print("\n" + "═" * 60)
    print("  CMC RAG Dual-Document Audit Pipeline 启动")
    print("═" * 60)

    # ── Protocol ────────────────────────────────────────────
    print("\n📋 [Protocol] 解析文件…")
    proto_pages  = load_pdf(protocol_pdf)
    proto_chunks = split_into_chunks(proto_pages, prefix="proto_")
    proto_col    = get_chroma_collection(COLLECTION_PROTOCOL)

    proto_existing = set(proto_col.get()["ids"]) if proto_col.count() > 0 else set()
    proto_new      = [c for c in proto_chunks if c["chunk_id"] not in proto_existing]
    if proto_new:
        print(f"📋 [Protocol] 向量化写入 {len(proto_new)} 块…")
        await upsert_chunks_to_chroma(proto_new, proto_col)

    # ── Report ──────────────────────────────────────────────
    print("\n📄 [Report] 解析文件…")
    report_pages  = load_pdf(report_pdf)
    report_chunks = split_into_chunks(report_pages, prefix="report_")
    report_col    = get_chroma_collection(COLLECTION_REPORT)

    report_existing = set(report_col.get()["ids"]) if report_col.count() > 0 else set()
    report_new      = [c for c in report_chunks if c["chunk_id"] not in report_existing]
    if report_new:
        print(f"📄 [Report] 向量化写入 {len(report_new)} 块…")
        await upsert_chunks_to_chroma(report_new, report_col)

    # ── 分别检索 ────────────────────────────────────────────
    print(f"\n🔍 检索 Protocol 相关标准段落…")
    proto_retrieved  = retrieve_relevant_chunks(audit_query, proto_col,  top_k=TOP_K)

    print(f"🔍 检索 Report 实际数据段落…")
    report_retrieved = retrieve_relevant_chunks(audit_query, report_col, top_k=TOP_K)

    # ── 构造 Prompt 并生成结论 ───────────────────────────────
    print(f"\n🤖 调用 {AUDIT_MODEL} 进行双文档比对审计…")
    prompt     = build_dual_audit_prompt(audit_query, proto_retrieved, report_retrieved)
    conclusion = await ollama_generate_async(prompt)

    print("\n" + "═" * 60)
    print("  审计结论")
    print("═" * 60)
    print(conclusion)
    print("═" * 60 + "\n")

    return {
        "query":            audit_query,
        "protocol_source":  Path(protocol_pdf).name,
        "report_source":    Path(report_pdf).name,
        "protocol_chunks":  proto_retrieved,
        "report_chunks":    report_retrieved,
        "conclusion":       conclusion,
    }


# ─────────────────────────────────────────────
# 8. 单文档兼容管道（无 Protocol 时）
# ─────────────────────────────────────────────

async def run_audit_pipeline(pdf_path: str, audit_query: str) -> dict:
    """
    单文档 RAG 审计管道（向后兼容）。
    Protocol 未提供时自动降级到此模式。
    """
    print("\n" + "═" * 60)
    print("  CMC RAG Audit Pipeline（单文档模式）启动")
    print("═" * 60)

    pages  = load_pdf(pdf_path)
    chunks = split_into_chunks(pages, prefix="report_")
    col    = get_chroma_collection(COLLECTION_REPORT)

    existing = set(col.get()["ids"]) if col.count() > 0 else set()
    new      = [c for c in chunks if c["chunk_id"] not in existing]
    if new:
        await upsert_chunks_to_chroma(new, col)

    retrieved  = retrieve_relevant_chunks(audit_query, col, top_k=TOP_K)
    prompt     = build_audit_prompt(audit_query, retrieved)
    conclusion = await ollama_generate_async(prompt)

    print("\n" + "═" * 60)
    print(conclusion)
    print("═" * 60 + "\n")

    return {
        "query":      audit_query,
        "pdf_source": Path(pdf_path).name,
        "retrieved":  retrieved,
        "conclusion": conclusion,
    }


def save_audit_report(result: dict, output_path: str = "audit_report.json") -> None:
    """将审计结果保存为 JSON。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[报告] 已保存：{output_path}")


# ─────────────────────────────────────────────
# 9. 演示模式（无真实 PDF）
# ─────────────────────────────────────────────

async def demo_dual_audit() -> None:
    """
    双文档演示模式：
    内置示例 Protocol（标准）和 Report（实际数据，含 OOS/OOT）直接写库。
    """
    print("\n[演示模式 - 双文档] 使用内置 CMC 示例数据\n")

    # ── Protocol 示例数据 ──────────────────────────────────
    protocol_samples = [
        {
            "chunk_id": "proto_001",
            "text": (
                "【验收标准 — 含量测定】\n"
                "成品含量（HPLC 法）：标示量的 98.0%–102.0%。\n"
                "加速稳定性（40°C/75%RH，6 个月）：不低于标示量 97.0%。\n"
                "长期稳定性（25°C/60%RH，24 个月）：不低于标示量 97.0%。"
            ),
            "source": "demo_protocol.pdf",
            "page":    3,
        },
        {
            "chunk_id": "proto_002",
            "text": (
                "【验收标准 — 有关物质（降解产物）】\n"
                "单个未知杂质：≤ 0.10%。\n"
                "总杂质：≤ 0.50%。\n"
                "降解产物 A（已知杂质）：≤ 0.08%（24 个月长期）。\n"
                "降解产物 B（已知杂质）：≤ 0.05%（各时间点）。"
            ),
            "source": "demo_protocol.pdf",
            "page":    4,
        },
        {
            "chunk_id": "proto_003",
            "text": (
                "【验收标准 — 溶出度】\n"
                "Q ≥ 85%（45 min，pH 6.8 磷酸盐缓冲液，桨法 50 rpm）。\n"
                "长期稳定性各时间点均须满足上述标准。"
            ),
            "source": "demo_protocol.pdf",
            "page":    5,
        },
    ]

    # ── Report 示例数据（含 OOS + OOT）──────────────────────
    report_samples = [
        {
            "chunk_id": "report_001",
            "text": (
                "【长期稳定性检测结果 — 含量测定】\n"
                "0 个月：101.2%\n"
                "6 个月：100.5%\n"
                "12 个月：99.1%\n"
                "18 个月：98.3%\n"
                "24 个月：96.8%  ← 低于 Protocol 规定下限 97.0%"
            ),
            "source": "demo_stability_report.pdf",
            "page":    7,
        },
        {
            "chunk_id": "report_002",
            "text": (
                "【长期稳定性检测结果 — 降解产物】\n"
                "降解产物 A：\n"
                "  0 个月：0.02%\n"
                "  6 个月：0.03%\n"
                "  12 个月：0.05%\n"
                "  18 个月：0.07%\n"
                "  24 个月：0.11%  ← 超出 Protocol 限度 0.08%（OOS）\n"
                "降解产物 B：\n"
                "  0–24 个月均 ≤ 0.04%，符合规定。"
            ),
            "source": "demo_stability_report.pdf",
            "page":    8,
        },
        {
            "chunk_id": "report_003",
            "text": (
                "【长期稳定性检测结果 — 溶出度】\n"
                "0 个月：94.2%\n"
                "6 个月：92.1%\n"
                "12 个月：89.5%\n"
                "18 个月：88.0%\n"
                "24 个月：86.3%  符合 Q≥85% 规定，但呈持续下降趋势（OOT 风险）。"
            ),
            "source": "demo_stability_report.pdf",
            "page":    9,
        },
    ]

    # ── 写入向量库 ────────────────────────────────────────
    proto_col  = get_chroma_collection(COLLECTION_PROTOCOL)
    report_col = get_chroma_collection(COLLECTION_REPORT)

    proto_existing  = set(proto_col.get()["ids"])  if proto_col.count()  > 0 else set()
    report_existing = set(report_col.get()["ids"]) if report_col.count() > 0 else set()

    new_proto  = [s for s in protocol_samples if s["chunk_id"] not in proto_existing]
    new_report = [s for s in report_samples   if s["chunk_id"] not in report_existing]

    async def _bulk_upsert(items, col, label):
        if not items:
            print(f"[演示] {label} 已存在，跳过写入")
            return
        embs = await asyncio.gather(*[ollama_embed_async(s["text"]) for s in items])
        col.upsert(
            ids        = [s["chunk_id"] for s in items],
            embeddings = embs,
            documents  = [s["text"]     for s in items],
            metadatas  = [{"source": s["source"], "page": s["page"]} for s in items],
        )
        print(f"[演示] {label} 写入 {len(items)} 条")

    await _bulk_upsert(new_proto,  proto_col,  "Protocol")
    await _bulk_upsert(new_report, report_col, "Report")

    # ── 审计 ──────────────────────────────────────────────
    query = (
        "请将稳定性报告中的实际检测数据与 Protocol 规定的验收标准逐项比对，"
        "明确指出哪些数据存在 OOS（超标）或 OOT（不良趋势）风险，并给出 CAPA 建议。"
    )

    proto_retrieved  = retrieve_relevant_chunks(query, proto_col,  top_k=TOP_K)
    report_retrieved = retrieve_relevant_chunks(query, report_col, top_k=TOP_K)

    prompt     = build_dual_audit_prompt(query, proto_retrieved, report_retrieved)
    conclusion = await ollama_generate_async(prompt)

    print("\n" + "═" * 60)
    print("  双文档审计结论（演示模式）")
    print("═" * 60)
    print(conclusion)
    print("═" * 60)

    save_audit_report({
        "query":           query,
        "protocol_source": "demo_protocol.pdf",
        "report_source":   "demo_stability_report.pdf",
        "protocol_chunks": proto_retrieved,
        "report_chunks":   report_retrieved,
        "conclusion":      conclusion,
    }, output_path="audit_report_demo.json")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # 用法：python rag_audit.py <protocol_pdf> <report_pdf> "<审计问题>"
        asyncio.run(run_dual_audit_pipeline(sys.argv[1], sys.argv[2], sys.argv[3]))

    elif len(sys.argv) == 3:
        # 单文档降级模式
        asyncio.run(run_audit_pipeline(sys.argv[1], sys.argv[2]))

    elif len(sys.argv) == 1:
        # 无参数 → 双文档演示
        asyncio.run(demo_dual_audit())

    else:
        print("用法：")
        print("  演示模式（无 PDF）：   python rag_audit.py")
        print("  双文档审计：          python rag_audit.py <protocol.pdf> <report.pdf> \"审计问题\"")
        print("  单文档（降级）：       python rag_audit.py <report.pdf> \"审计问题\"")
        sys.exit(1)
