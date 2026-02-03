import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.query import Query
from src.generate import generate

# =========================
# Config
# =========================
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

K_RETRIEVAL = 20   # FAISS에서 뽑는 후보 chunk 수
K_CONTEXT   = 5    # LLM 프롬프트에 실제로 넣는 chunk 수

INPUT_JSONL  = "3gpp_rag_eval_qa_100.jsonl"
OUTPUT_JSONL = "3gpp_rag_eval_qa_100_open_answers.jsonl"


# =========================
# Core pipeline (open-ended)
# =========================
def build_query_obj(user_query: str) -> Query:
    """
    Query 객체 생성 -> Terms/Definitions 확장 -> 3GPP retrieval 수행.
    """
    q = Query(user_query, [])

    # def_TA_question()은 self.query를 쓰므로 먼저 세팅
    q.query = q.question

    # Terms/Definitions, Abbrev 확장
    q.def_TA_question()

    # 원래 질문 텍스트도 확장된 질문으로 overwrite (네 pipeline 방식)
    q.question = q.enhanced_query

    # 3GPP context retrieval (validator off)
    q.get_3GPP_context(k=K_RETRIEVAL, model_name=MODEL_NAME, validate_flag=False)

    return q


def slice_context_for_llm(q: Query) -> List[str]:
    """
    retrieval 결과(context)가 list이면 상위 K_CONTEXT만 LLM에 넣도록 슬라이스.
    """
    ctx = getattr(q, "context", None)

    if isinstance(ctx, list):
        q.context = ctx[:K_CONTEXT]
        return q.context

    if isinstance(ctx, str) and ctx.strip():
        q.context = [ctx]
        return q.context

    q.context = []
    return q.context


def answer_one_open(question_text: str) -> Dict[str, Any]:
    """
    1개 질문에 대해 open-ended 답 생성.
    """
    start = time.time()

    qobj = build_query_obj(question_text)
    context_used = slice_context_for_llm(qobj)  # ✅ LLM에 들어갈 컨텍스트 5개만 유지

    # generate()는 question.context를 기반으로 prompt를 만들기 때문에,
    # 위에서 qobj.context를 5개로 줄였으면 LLM 입력도 5개만 들어감.
    answer_str, context_str, _ = generate(qobj, MODEL_NAME)

    return {
        "question": question_text,
        "model": MODEL_NAME,
        "k_retrieval": K_RETRIEVAL,
        "k_context": K_CONTEXT,
        "answer": answer_str,              # ✅ 생성된 주관식 답변
        "context_used": context_used,      # ✅ 실제 LLM에 넣은 chunk 5개(리스트)
        "elapsed_sec": round(time.time() - start, 4),
    }


# =========================
# JSONL runner
# =========================
def run_jsonl(
    input_path: str = INPUT_JSONL,
    output_path: str = OUTPUT_JSONL,
    resume_append: bool = True,
) -> None:
    """
    INPUT_JSONL에서 question 읽어 open-ended 답 생성 후 OUTPUT_JSONL로 저장.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {in_path.resolve()}")

    out_mode = "a" if resume_append else "w"

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(out_mode, encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            # 1) parse json
            try:
                item = json.loads(line)
            except Exception:
                fout.write(json.dumps({
                    "line": idx,
                    "error": "json_parse_failed",
                    "raw": line[:500]
                }, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # 2) read question
            q = item.get("question")
            if not isinstance(q, str) or not q.strip():
                fout.write(json.dumps({
                    "line": idx,
                    "error": "missing_question",
                    "keys": list(item.keys())
                }, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            question_text = q.strip()

            # 3) generate answer
            try:
                result = answer_one_open(question_text)

                # 원본 메타를 같이 남기고 싶으면 keep (정답/옵션은 "참고용"으로만 보존)
                for keep_key in ("category", "difficulty", "source", "answer", "options", "explanation"):
                    if keep_key in item:
                        result[keep_key] = item[keep_key]

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"[{idx}] DONE | {question_text[:90]}")

            except Exception as e:
                fout.write(json.dumps({
                    "line": idx,
                    "question": question_text[:300],
                    "error": str(e),
                    "traceback": traceback.format_exc()[-4000:],
                }, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"[{idx}] ERROR: {e}")

    print(f"\n✅ Saved results to: {out_path.resolve()}")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    run_jsonl()
