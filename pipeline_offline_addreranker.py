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
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

K_RETRIEVAL = 20    # 1차 FAISS 검색 개수 (후보군)
TOP_K_RERANK = 5    # 2차 리랭크 최종 선택 개수 (LLM 전달용)
BATCH_SIZE = 8
MAX_LENGTH = 1024

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
        out = reranker(query=query, retrievals=question.retrievals, top_k=TOP_K_RERANK)

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
