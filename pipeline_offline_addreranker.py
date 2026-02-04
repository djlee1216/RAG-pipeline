# import os
# import re
# import json
# import traceback
# import logging
# from datetime import datetime
# from typing import List, Dict, Any, Optional

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from query import Query
# from generate import check_question  # 기존 유지
# from LLM import submit_prompt_flex


# ############################################
# # Models
# ############################################
# GEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# RERANK_MODEL = "BAAI/bge-reranker-large"

# # retrieval / rerank params
# K_RETRIEVAL = 20
# TOP_K_RERANK = 5
# BATCH_SIZE = 8
# MAX_LENGTH = 512


# ############################################
# # Utils
# ############################################
# def _preview(text: str, n: int = 320) -> str:
#     t = re.sub(r"\s+", " ", (text or "").strip())
#     return t[:n] + ("..." if len(t) > n else "")


# def _get_hf_token() -> Optional[str]:
#     return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


# def build_blocks_from_retrievals(retrievals: List[Dict[str, Any]]) -> str:
#     """
#     question.retrievals(구조화 메타, web+3gpp 혼합) -> reranker 파서가 읽을 수 있는 블록 문자열
#     """
#     blocks = ["The retrieved context provided to the LLM is:"]
#     for r in retrievals:
#         rank = int(r.get("rank", 0))
#         stype = r.get("source_type", "unknown")
#         score = float(r.get("score", 0.0))
#         meta = r.get("meta", {}) or {}
#         rid = meta.get("id", rank)

#         # text가 있으면 그걸 우선, 없으면 preview
#         text = r.get("text") or r.get("preview") or ""
#         blocks.append(
#             f"[Retrieval {rank}] (score={score:.4f}, type={stype}, id={rid})\n{text}\n"
#         )
#     return "\n".join(blocks).strip()


# ############################################
# # Reranker (BAAI/bge-reranker-large)
# ############################################
# def reranker(
#     query: str,
#     retrievals: List[Dict[str, Any]],
#     *,
#     reranker_model: str = RERANK_MODEL,
#     top_k: int = TOP_K_RERANK,
#     max_length: int = MAX_LENGTH,
#     batch_size: int = BATCH_SIZE,
#     save_json: bool = True,
#     json_path: str = "rerank_report.json",
#     print_moves: bool = True,
#     print_topk: bool = True,
#     filter_type: Optional[str] = None,  # "3gpp" or "web" or None
# ) -> Dict[str, Any]:

#     if not retrievals:
#         raise ValueError("retrievals is empty. Ensure question.get_*_context filled question.retrievals.")

#     # optional filter by type (없으면 섞어서 rerank)
#     if filter_type is not None:
#         ft = filter_type.lower()
#         retrievals = [r for r in retrievals if (r.get("source_type", "").lower() == ft)]
#         if not retrievals:
#             raise ValueError(f"No retrievals left after filter_type='{filter_type}'.")

#     # block string (for parsing/디버그용; 실제 점수는 retrievals 기반으로 바로 계산)
#     context_str = build_blocks_from_retrievals(retrievals)

#     # 1) Parse blocks into items
#     BLOCK_RE = re.compile(
#         r"\[Retrieval\s+(?P<idx>\d+)\]\s*"
#         r"\(score=(?P<score>[-+]?\d*\.?\d+),\s*type=(?P<type>[^,]+),\s*id=(?P<id>\d+)\)\s*\n"
#         r"(?P<text>.*?)(?=\n\s*\[Retrieval\s+\d+\]\s*\(|\Z)",
#         flags=re.DOTALL
#     )

#     items: List[Dict[str, Any]] = []
#     for m in BLOCK_RE.finditer(context_str):
#         items.append({
#             "old_rank": int(m.group("idx")),
#             "retrieval_score": float(m.group("score")),
#             "type": m.group("type").strip(),
#             "id": int(m.group("id")),
#             "text": (m.group("text") or "").strip(),
#         })

#     if not items:
#         raise ValueError("No retrieval blocks parsed. build_blocks_from_retrievals() output format check needed.")

#     # old_rank 순으로 정렬
#     items.sort(key=lambda x: x["old_rank"])

#     # 2) Load BAAI reranker
#     hf_token = _get_hf_token()

#     tokenizer = AutoTokenizer.from_pretrained(
#         reranker_model,
#         use_fast=True,
#         token=hf_token,
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         reranker_model,
#         device_map="auto",
#         dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # ✅ torch_dtype 경고 회피
#         token=hf_token,
#     )
#     model.eval()
#     device = next(model.parameters()).device

#     # pad_token 안전장치
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
#     model.config.pad_token_id = tokenizer.pad_token_id

#     # 3) Score pairs (query, passage)
#     @torch.no_grad()
#     def score_pairs(q: str, passages: List[str]) -> List[float]:
#         scores: List[float] = []
#         for i in range(0, len(passages), batch_size):
#             batch = passages[i:i + batch_size]
#             inputs = tokenizer(
#                 [q] * len(batch),
#                 batch,
#                 padding=True,
#                 truncation=True,
#                 max_length=max_length,
#                 return_tensors="pt",
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             out = model(**inputs)

#             # bge-reranker-large: logits shape (batch, 1) → (batch,)
#             logits = out.logits.squeeze(-1)
#             scores.extend(logits.detach().float().cpu().tolist())
#         return [float(x) for x in scores]

#     passages = [x["text"] for x in items]
#     rerank_scores = score_pairs(query, passages)

#     for x, rr in zip(items, rerank_scores):
#         x["rerank_score"] = float(rr)

#     # 4) Rerank + compute new ranks
#     reranked = sorted(items, key=lambda x: x["rerank_score"], reverse=True)
#     for new_rank, x in enumerate(reranked, start=1):
#         x["new_rank"] = new_rank
#         x["delta"] = x["old_rank"] - new_rank  # +면 올라감

#     # 5) Print rank movements
#     if print_moves:
#         print("\n=== Rank changes (old -> new) ===")
#         for x in sorted(reranked, key=lambda t: t["old_rank"]):
#             print(
#                 f"{x['old_rank']} -> {x['new_rank']}  "
#                 f"(type={x['type']}, id={x['id']}, "
#                 f"rerank={x['rerank_score']:.4f}, retrieval={x['retrieval_score']:.4f})"
#             )

#     # 6) Top-k 출력 + 반환용 포맷 구성
#     topk_items = reranked[:top_k]

#     if print_topk:
#         print(f"\n=== Top-{top_k} reranked chunks ===")
#         for x in topk_items:
#             print("-" * 100)
#             print(
#                 f"[Top {x['new_rank']}] (rerank_score={x['rerank_score']:.4f}, "
#                 f"orig_rank={x['old_rank']}, orig_score={x['retrieval_score']:.4f}, "
#                 f"type={x['type']}, id={x['id']})"
#             )
#             print(x["text"])
#             print("-" * 100)

#     # 7) JSON 저장 (전체 랭크변화 + top-k 전문 포함)
#     payload = {
#         "query": query,
#         "reranker_model": reranker_model,
#         "created_at": datetime.now().isoformat(timespec="seconds"),
#         "count": len(reranked),
#         "filter_type": filter_type,
#         "max_length": max_length,
#         "batch_size": batch_size,
#         "results": [
#             {
#                 "old_rank": x["old_rank"],
#                 "new_rank": x["new_rank"],
#                 "delta": x["delta"],
#                 "type": x["type"],
#                 "id": x["id"],
#                 "retrieval_score": x["retrieval_score"],
#                 "rerank_score": x["rerank_score"],
#                 "snippet": _preview(x["text"], 800),
#             }
#             for x in reranked
#         ],
#         "top_k": [
#             {
#                 "old_rank": x["old_rank"],
#                 "new_rank": x["new_rank"],
#                 "type": x["type"],
#                 "id": x["id"],
#                 "retrieval_score": x["retrieval_score"],
#                 "rerank_score": x["rerank_score"],
#                 "text": x["text"],
#             }
#             for x in topk_items
#         ],
#     }

#     saved_path = None
#     if save_json:
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(payload, f, ensure_ascii=False, indent=2)
#         saved_path = json_path
#         print(f"\n✅ Saved JSON report to: {json_path}")

#     # 8) generate()로 넘길 topk_list(list[str]) 구성
#     topk_list: List[str] = []
#     for i, x in enumerate(topk_items, start=1):
#         topk_list.append(
#             f"[Retrieval {i}] (score={x['rerank_score']:.4f}, type={x['type']}, id={x['id']})\n"
#             f"{x['text']}\n"
#         )

#     # 9) print_topk_and_answer에 넣을 구조화 retrievals
#     retrievals_struct: List[Dict[str, Any]] = []
#     for i, x in enumerate(topk_items, start=1):
#         retrievals_struct.append({
#             "rank": i,
#             "source_type": x["type"],
#             "score": float(x["rerank_score"]),
#             "preview": _preview(x["text"], 320),
#             "meta": {
#                 "old_rank": x["old_rank"],
#                 "new_rank": i,
#                 "delta": x["delta"],
#                 "id": x["id"],
#             }
#         })

#     return {
#         "topk_list": topk_list,
#         "retrievals": retrievals_struct,
#         "json_path": saved_path,
#     }


# ############################################
# # Generate 
# ############################################
# def generate_for_rerank(question, context_rerank: List[str], model_name: str):
#     """
#     rerank된 top-k context(list[str])를 받아 답변 생성
#     """
#     try:
#         content = "\n".join(context_rerank)

#         prompt = f"""
# Here is some rules that you must to remind when you make a response.
#     Rule 1. The answer must not exceed 1,000 characters.
#     Rule 2. Identify exactly what the question is asking and provide a focused response.
#     Rule 3. Even if information is present in the context, do not include it in your answer if it is irrelevant to the question.

# Please answer the following question:
# {question.query}

# Considering the following context:
# {content}

# Please answer the following question, add between paranthesis the retrieval(e.g. Retrieval 3) that you used for each eleement of your reasoning:
# {question.question}
# The answer must not exceed 1,000 characters.
#         """.strip()

#         predicted_answers_str = submit_prompt_flex(prompt, model=model_name)

#         context_str = f"The retrieved context provided to the LLM is:\n{content}"
#         return predicted_answers_str, context_str, question.question

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         return None, None, None


# ############################################
# # Print (네 스타일 유지)
# ############################################
# def print_topk_and_answer(question, response, k=5):
#     print("\n\n\n")
#     marker = "[Terms & Definitions]:"
#     idx = question.question.find(marker)
#     if idx != -1:
#         print(question.question[idx:])

#     print("-" * 60)
#     print("[Top-k Retrieved Contexts]\n")

#     retrievals = getattr(question, "retrievals", None)
#     if not retrievals:
#         ctx = getattr(question, "context", None)
#         if isinstance(ctx, list):
#             for i, c in enumerate(ctx[:k], 1):
#                 print(f"({i}) {c[:1500]}{' ...' if len(c) > 500 else ''}\n")
#         else:
#             print("(no structured retrievals)\n")
#     else:
#         for r in retrievals[:k]:
#             meta = r.get("meta", {}) or {}
#             stype = r.get("source_type", "unknown")
#             score = r.get("score", 0.0)
#             preview = r.get("preview", "")

#             old_rank = meta.get("old_rank", None)
#             move_info = f" (old_rank={old_rank} -> new_rank={r.get('rank')})" if old_rank is not None else ""

#             if stype == "web":
#                 rel_path = meta.get("rel_path", "")
#                 site = rel_path.split("/")[0] if rel_path else "web"
#                 print(f"({r['rank']}) [WEB] {site}  (score={score:.4f}){move_info}")
#                 if rel_path:
#                     print(f"    path: {rel_path}")
#                 print(f"    {preview}\n")

#             elif stype == "3gpp":
#                 doc = meta.get("doc_title") or meta.get("source_file", "3gpp")
#                 section = f"{meta.get('section_id','')} {meta.get('section_title','')}".strip()
#                 print(f"({r['rank']}) [3GPP] {doc}  (score={score:.4f}){move_info}")
#                 if section:
#                     print(f"    section: {section}")
#                 print(f"    {preview}\n")

#             else:
#                 print(f"({r['rank']}) [UNKNOWN] (score={score:.4f}){move_info}")
#                 print(f"    {preview}\n")

#     print("-" * 60)
#     print("Answer:\n")
#     print((response or "").strip())
#     print("-" * 60 + "\n")


# ############################################
# # Main pipeline
# ############################################
# def TelcoRAG(query, answer=None, options=None, model_name=GEN_MODEL):
#     try:
#         question = Query(query, [])
#         question.query = question.question
#         question.def_TA_question()
#         question.question = question.enhanced_query

#         question.get_3GPP_context(k=K_RETRIEVAL, model_name=model_name, validate_flag=False)

#         if answer is not None:
#             response, context, _ = check_question(question, answer, options, model_name=model_name)
#             return response, question.context


#         if not getattr(question, "retrievals", None):
#             raise ValueError("question.retrievals is empty. Need structured retrievals (web+3gpp) for mixed rerank.")

#         out = reranker(
#             query=question.question,
#             retrievals=question.retrievals,   
#             top_k=TOP_K_RERANK,
#             filter_type=None,               
#             save_json=True,
#             json_path="rerank_report.json",
#             print_moves=True,
#             print_topk=True,
#         )

#         # print가 rerank 결과를 찍도록
#         question.retrievals = out["retrievals"]

#         # generate는 top-k reranked context만 사용
#         context_rerank = out["topk_list"]
#         response, context_used, _ = generate_for_rerank(question, context_rerank, model_name)

#         print_topk_and_answer(question, response, k=TOP_K_RERANK)
#         return response, context_used

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print(traceback.format_exc())
#         return None, None


# ############################################
# # CLI
# ############################################
# if __name__ == "__main__":
#     while True:
#         user_q = input("Enter your question (type 'exit' to quit): ").strip()
#         if not user_q:
#             continue
#         if user_q.lower() in ("exit", "quit", "q"):
#             print("Bye.")
#             break

#         TelcoRAG(user_q, model_name=GEN_MODEL)
import os
import re
import json
import traceback
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from query import Query
from generate import check_question
from LLM import submit_prompt_flex

############################################
# Models & Config
############################################
GEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RERANK_MODEL = "BAAI/bge-reranker-large"

K_RETRIEVAL = 20    # 1차 FAISS 검색 개수 (후보군)
TOP_K_RERANK = 5    # 2차 리랭크 최종 선택 개수 (LLM 전달용)
BATCH_SIZE = 8
MAX_LENGTH = 512

############################################
# Utils
############################################
def _preview(text: str, n: int = 320) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:n] + ("..." if len(t) > n else "")

def _get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

############################################
# Reranker (BAAI/bge-reranker-large)
############################################
def reranker(
    query: str,
    retrievals: List[Dict[str, Any]],
    *,
    reranker_model: str = RERANK_MODEL,
    top_k: int = TOP_K_RERANK,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    save_json: bool = True,
    json_path: str = "rerank_report.json",
    print_moves: bool = True,
) -> Dict[str, Any]:

    if not retrievals:
        raise ValueError("retrievals is empty. Ensure 1st retrieval step is working.")

    # 1) 기존 리트리벌 결과 파싱
    items = []
    for r in retrievals:
        items.append({
            "old_rank": r.get("rank"),
            "retrieval_score": r.get("score"),
            "type": r.get("source_type"),
            "id": r.get("meta", {}).get("id", 0),
            "text": r.get("text") or r.get("preview") or "",
            "meta": r.get("meta", {})
        })

    # 2) Reranker 로드 (BGE-Reranker)
    hf_token = _get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(reranker_model, use_fast=True, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        reranker_model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )
    model.eval()
    device = next(model.parameters()).device

    # 3) Scoring pairs (query, passage)
    passages = [x["text"] for x in items]
    rerank_scores = []
    with torch.no_grad():
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            inputs = tokenizer([query] * len(batch), batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**inputs)
            rerank_scores.extend(out.logits.squeeze(-1).float().cpu().tolist())

    for x, rr in zip(items, rerank_scores):
        x["rerank_score"] = float(rr)

    # 4) 정렬 및 상위 k개 추출
    reranked = sorted(items, key=lambda x: x["rerank_score"], reverse=True)
    topk_items = reranked[:top_k]

    # 5) 결과 구조화
    retrievals_struct = []
    topk_list_for_gen = []
    for i, x in enumerate(topk_items, start=1):
        topk_list_for_gen.append(f"[Retrieval {i}] (score={x['rerank_score']:.4f}, type={x['type']})\n{x['text']}\n")
        retrievals_struct.append({
            "rank": i,
            "source_type": x["type"],
            "score": x["rerank_score"],
            "preview": _preview(x["text"], 500),
            "meta": {**x["meta"], "old_rank": x["old_rank"], "new_rank": i}
        })

    if save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"query": query, "top_k": retrievals_struct}, f, ensure_ascii=False, indent=2)

    return {"topk_list": topk_list_for_gen, "retrievals": retrievals_struct}

############################################
# Generate (Reranked Context만 사용)
############################################
def generate_for_rerank(question, context_rerank: List[str], model_name: str):
    try:
        content = "\n".join(context_rerank)
        # 룰을 명확히 전달하는 프롬프트
        prompt = f"""
Rule 1. Identify exactly what the question is asking and provide a focused response.
Rule 2. Even if information is present in the context, do not include it if it is irrelevant.

[Context]
{content}

[Question]
{question.query}

Please answer the question based on the context above. 
Add between paranthesis the retrieval (e.g. Retrieval 3) used for each reasoning.
""".strip()

        predicted_answers_str = submit_prompt_flex(prompt, model=model_name)
        return predicted_answers_str, content

    except Exception as e:
        logging.error(f"Error in generate: {e}")
        return None, None

############################################
# Main Pipeline
############################################
def TelcoRAG(query, answer=None, options=None, model_name=GEN_MODEL):
    try:
        # 1. 초기화 (Query 객체 생성)
        question = Query(query, [])
        
        # 2. 용어 정제 (Enhanced Query 반영)
        # 이 메소드 호출로 Abbreviation.txt, glossary.txt, 3GPP_vocabulary.docx 등이 반영됩니다.
        print(f"\n🔍 [1/4] Refining query with terms & definitions...")
        question.def_TA_question() 
        
        # 정제된 쿼리를 검색 엔진에 전달할 쿼리로 설정
        question.question = question.enhanced_query 
        print(f"✨ Enhanced Query:\n{question.question[:500]}...")

        # 3. 1차 검색 (FAISS에서 20개 추출)
        print(f"📡 [2/4] Retrieving top-{K_RETRIEVAL} chunks from FAISS...")
        question.get_3GPP_context(k=K_RETRIEVAL)

        if not getattr(question, "retrievals", None) or len(question.retrievals) == 0:
            raise ValueError("No retrievals found during the first search step.")

        # 4. 리랭크 (20개 -> 5개)
        print(f"🔄 [3/4] Reranking candidates with {RERANK_MODEL}...")
        out = reranker(query=question.question, retrievals=question.retrievals, top_k=TOP_K_RERANK)

        # ★ 핵심: question 객체의 context를 리랭크된 5개로 교체 ★
        question.retrievals = out["retrievals"]
        question.context = out["topk_list"]

        # 5. 최종 답변 생성 (정제된 쿼리 + 리랭크된 컨텍스트)
        print(f"✍️ [4/4] Generating final answer with {model_name}...")
        response, context_used = generate_for_rerank(question, question.context, model_name)

        # 6. 결과 출력 (커스텀 스타일)
        print("\n" + "="*80)
        print(f"FINAL ANSWER (Top-{TOP_K_RERANK} Reranked)")
        print("-" * 80)
        print(response)
        print("="*80 + "\n")

        return response, context_used

    except Exception as e:
        print(f"❌ Error occurred in TelcoRAG: {e}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    while True:
        user_input = input("Enter your question (q to quit): ").strip()
        if user_input.lower() in ('q', 'quit', 'exit'):
            break
        TelcoRAG(user_input)
