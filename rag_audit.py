"""
CMC RAG Audit Pipeline
======================
闭环流程：PDF 解析 → 结构化切块 → 本地 Ollama Embedding → ChromaDB 向量存储
         → 语义检索 → Gemma 生成审计结论

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
OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"    # 专用 embedding 模型（274MB，支持 /api/embeddings）
AUDIT_MODEL       = "gemma4:latest"       # 用于生成审计结论的本地模型
CHROMA_PATH       = "./chroma_db"         # ChromaDB 本地持久化路径（已在 .gitignore 中）
COLLECTION_NAME   = "cmc_standards"       # 向量集合名称

CHUNK_SIZE        = 600    # 每块目标字符数
CHUNK_OVERLAP     = 80     # 块间重叠字符数，保留上下文连贯性
TOP_K             = 4      # 检索时返回最相关的 Top-K 块


# ─────────────────────────────────────────────
# 1. PDF 解析与结构化切块
# ─────────────────────────────────────────────

def load_pdf(pdf_path: str) -> list[dict]:
    """
    用 PyMuPDF 逐页解析 PDF，返回带页码元数据的段落列表。
    每个元素形如：{"text": "...", "page": 3, "source": "xxx.pdf"}
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        # 清理多余空行与首尾空白
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


def split_into_chunks(pages: list[dict],
                      chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    滑动窗口切块：保留 source / page 元数据，连续块之间保留 overlap 字符。
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
            if len(chunk) > 30:          # 过滤极短碎片
                chunks.append({
                    "text":       chunk,
                    "source":     source,
                    "page":       page,
                    "chunk_id":   f"{source}_p{page}_c{len(chunks)}",
                })
            start += chunk_size - overlap

    print(f"[切块] 共生成 {len(chunks)} 个文本块（chunk_size={chunk_size}, overlap={overlap}）")
    return chunks


# ─────────────────────────────────────────────
# 2. 本地 Ollama 接口（Embedding & Generate）
# ─────────────────────────────────────────────

def ollama_embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """
    调用 Ollama /api/embeddings 获取文本向量。
    返回浮点数列表（维度取决于模型）。
    """
    url  = f"{OLLAMA_BASE_URL}/api/embeddings"
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if "embedding" not in data:
        raise ValueError(f"Ollama 返回数据中没有 embedding 字段：{data}")
    return data["embedding"]


def ollama_generate(prompt: str,
                    model: str = AUDIT_MODEL,
                    stream: bool = False) -> str:
    """
    调用 Ollama /api/generate 生成文本（非流式，直接返回完整回复）。
    """
    url  = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": stream,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


async def ollama_embed_async(text: str, model: str = EMBED_MODEL) -> list[float]:
    """异步版本的 Embedding，用于批量并发向量化。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ollama_embed, text, model)


async def ollama_generate_async(prompt: str, model: str = AUDIT_MODEL) -> str:
    """异步版本的文本生成。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ollama_generate, prompt, model)


# ─────────────────────────────────────────────
# 3. ChromaDB 向量存储
# ─────────────────────────────────────────────

def get_chroma_collection(persist_path: str = CHROMA_PATH,
                          collection_name: str = COLLECTION_NAME):
    """
    初始化 ChromaDB 持久化客户端，返回目标集合（不存在则自动创建）。
    使用自定义 Embedding 函数（Ollama），不依赖 ChromaDB 内置模型。
    """
    client = chromadb.PersistentClient(path=persist_path)
    # 使用 cosine 距离（语义检索更准确）
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[ChromaDB] 集合 '{collection_name}' 当前已有 {collection.count()} 条记录")
    return collection


async def upsert_chunks_to_chroma(chunks: list[dict],
                                  collection,
                                  batch_size: int = 8) -> None:
    """
    异步批量向量化并写入 ChromaDB。
    - 已存在的 chunk_id 会被更新（upsert 语义）
    - 分批处理避免一次性请求过多导致超时
    """
    total = len(chunks)
    print(f"[Embedding] 开始向量化 {total} 个文本块，使用模型：{EMBED_MODEL}")

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]

        # 并发向量化本批次
        tasks = [ollama_embed_async(c["text"]) for c in batch]
        embeddings = await asyncio.gather(*tasks)

        ids        = [c["chunk_id"] for c in batch]
        documents  = [c["text"]     for c in batch]
        metadatas  = [{"source": c["source"], "page": c["page"]} for c in batch]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        done = min(batch_start + batch_size, total)
        print(f"  → [{done}/{total}] 已写入 ChromaDB")

    print(f"[ChromaDB] 全部写入完成，集合现有 {collection.count()} 条记录")


# ─────────────────────────────────────────────
# 4. 语义检索
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(query: str,
                              collection,
                              top_k: int = TOP_K) -> list[dict]:
    """
    将查询语句向量化后，在 ChromaDB 中做余弦相似度检索，
    返回 top_k 个最相关文本块（含来源和相似度分数）。
    """
    query_vec = ollama_embed(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks_out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks_out.append({
            "text":       doc,
            "source":     meta.get("source", ""),
            "page":       meta.get("page", "?"),
            "similarity": round(1 - dist, 4),   # cosine distance → similarity
        })

    return chunks_out


# ─────────────────────────────────────────────
# 5. 审计提示词（Prompt）构造
# ─────────────────────────────────────────────

AUDIT_PROMPT_TEMPLATE = """你是一位专业的 CMC（Chemistry, Manufacturing and Controls）法规合规审计专家。
请根据以下从 CMC 文件中检索到的相关内容，对用户提出的审计问题进行分析并给出结论。

═══════════════════════════════════════
【审计问题】
{query}

═══════════════════════════════════════
【检索到的相关文档内容】
{context}

═══════════════════════════════════════
【审计要求】
1. 首先判断上述文档内容是否能够充分回答审计问题。
2. 若能，请给出明确的合规性结论（合规 / 存在缺陷 / 信息不足）。
3. 指出具体的证据来源（文件名 + 页码）。
4. 若发现潜在合规风险，请列出改进建议。
5. 请用中文回复，语言简洁专业。

【审计结论】
"""


def build_audit_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """将检索到的文本块拼接为上下文，填入审计提示词模板。"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[来源 {i}] 文件：{chunk['source']} | 第 {chunk['page']} 页 | 相似度：{chunk['similarity']}\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return AUDIT_PROMPT_TEMPLATE.format(query=query, context=context)


# ─────────────────────────────────────────────
# 6. 完整闭环管道
# ─────────────────────────────────────────────

async def run_audit_pipeline(pdf_path: str, audit_query: str) -> dict:
    """
    完整 RAG 审计管道：
    PDF → 切块 → Embedding → ChromaDB → 检索 → Gemma 审计结论

    Args:
        pdf_path:    本地 PDF 文件路径
        audit_query: 审计问题（自然语言）

    Returns:
        包含检索结果和审计结论的字典
    """
    print("\n" + "═" * 60)
    print("  CMC RAG Audit Pipeline 启动")
    print("═" * 60)

    # Step 1: 解析 PDF
    print("\n📄 Step 1: 解析 PDF 文件...")
    pages = load_pdf(pdf_path)

    # Step 2: 结构化切块
    print("\n✂️  Step 2: 结构化切块...")
    chunks = split_into_chunks(pages)

    # Step 3: 初始化向量库
    print("\n🗄️  Step 3: 初始化 ChromaDB...")
    collection = get_chroma_collection()

    # Step 4: 向量化并写入（若已存在则跳过重复写入）
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if new_chunks:
        print(f"\n🔢 Step 4: 向量化写入 {len(new_chunks)} 个新文本块...")
        await upsert_chunks_to_chroma(new_chunks, collection)
    else:
        print(f"\n✅ Step 4: 所有文本块已在向量库中，跳过写入（共 {collection.count()} 条）")

    # Step 5: 语义检索
    print(f"\n🔍 Step 5: 语义检索（Top-{TOP_K}）...")
    print(f"  审计问题：{audit_query}")
    retrieved = retrieve_relevant_chunks(audit_query, collection, top_k=TOP_K)
    print(f"  检索到 {len(retrieved)} 个相关文本块：")
    for i, r in enumerate(retrieved, 1):
        print(f"    [{i}] {r['source']} P{r['page']} | 相似度 {r['similarity']}")

    # Step 6: 构造 Prompt 并调用 Gemma 生成审计结论
    print(f"\n🤖 Step 6: 调用本地 {AUDIT_MODEL} 生成审计结论...")
    prompt   = build_audit_prompt(audit_query, retrieved)
    conclusion = await ollama_generate_async(prompt)

    print("\n" + "═" * 60)
    print("  审计结论")
    print("═" * 60)
    print(conclusion)
    print("═" * 60 + "\n")

    return {
        "query":      audit_query,
        "pdf_source": str(Path(pdf_path).name),
        "retrieved":  retrieved,
        "conclusion": conclusion,
    }


def save_audit_report(result: dict, output_path: str = "audit_report.json") -> None:
    """将审计结果保存为 JSON 报告文件。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[报告] 审计结果已保存：{output_path}")


# ─────────────────────────────────────────────
# 7. 快速测试入口（无真实 PDF 时用虚拟内容演示）
# ─────────────────────────────────────────────

async def demo_without_pdf() -> None:
    """
    无需真实 PDF 的演示模式：
    直接将示例 CMC 文本段落写入向量库，然后执行审计查询。
    """
    print("\n[演示模式] 使用内置 CMC 示例文本（无需真实 PDF）\n")

    sample_texts = [
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
            ),
            "source": "demo_cmc.txt",
            "page":    4,
        },
    ]

    print("[演示] 初始化向量库并写入示例数据...")
    collection = get_chroma_collection()

    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    new_items    = [s for s in sample_texts if s["chunk_id"] not in existing_ids]

    if new_items:
        tasks      = [ollama_embed_async(s["text"]) for s in new_items]
        embeddings = await asyncio.gather(*tasks)

        collection.upsert(
            ids        = [s["chunk_id"]  for s in new_items],
            embeddings = embeddings,
            documents  = [s["text"]      for s in new_items],
            metadatas  = [{"source": s["source"], "page": s["page"]} for s in new_items],
        )
        print(f"[演示] 已写入 {len(new_items)} 条示例文本到向量库")
    else:
        print(f"[演示] 示例数据已在向量库中（共 {collection.count()} 条），跳过写入")

    # 执行审计查询
    audit_query = "该产品的杂质控制策略是否符合 ICH Q3A 要求？溶出度标准是否合理？"
    print(f"\n🔍 执行审计查询：{audit_query}")

    retrieved  = retrieve_relevant_chunks(audit_query, collection)
    prompt     = build_audit_prompt(audit_query, retrieved)
    conclusion = await ollama_generate_async(prompt)

    print("\n" + "═" * 60)
    print("  审计结论（演示模式）")
    print("═" * 60)
    print(conclusion)
    print("═" * 60)

    save_audit_report({
        "query":      audit_query,
        "pdf_source": "demo",
        "retrieved":  retrieved,
        "conclusion": conclusion,
    }, output_path="audit_report_demo.json")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # 用法：python rag_audit.py <pdf路径> "<审计问题>"
        pdf_file    = sys.argv[1]
        query_text  = sys.argv[2]

        async def main():
            result = await run_audit_pipeline(pdf_file, query_text)
            save_audit_report(result)

        asyncio.run(main())

    elif len(sys.argv) == 1:
        # 无参数 → 演示模式（无需真实 PDF）
        asyncio.run(demo_without_pdf())

    else:
        print("用法：")
        print("  演示模式（无需 PDF）：  python rag_audit.py")
        print("  真实 PDF 审计：        python rag_audit.py <pdf路径> \"审计问题\"")
        sys.exit(1)
