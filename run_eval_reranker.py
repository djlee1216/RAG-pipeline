import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any

from pipeline_offline_addreranker import TelcoRAG, GEN_MODEL


# ===============================
# VESSL 경로 설정 (중요)
# ===============================
BASE_DIR = Path(__file__).resolve().parent        # /djlee/repo
INPUT_JSONL  = BASE_DIR / "3gpp_rag_eval_qa_100.jsonl"
OUTPUT_DIR   = BASE_DIR.parent / "outputs"        # /djlee/outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL = OUTPUT_DIR / "3gpp_rag_eval_qa_100_answers_reranker.jsonl"


def answer_one_open(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    addreranker 기반 주관식 답안 생성
    """
    question = item["question"].strip()
    start = time.time()

    # 🔴 핵심: answer / options 절대 넘기지 않는다
    answer, context_used = TelcoRAG(
        query=question,
        answer=None,
        options=None,
        model_name=GEN_MODEL,
    )

    elapsed = round(time.time() - start, 4)

    result = {
        "question": question,
        "answer": answer,
        "context_used": context_used,
        "latency_sec": elapsed,
    }

    # 원본 메타 보존 (있으면)
    for k in ["category", "difficulty", "source"]:
        if k in item:
            result[k] = item[k]

    return result


def main():
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_JSONL}")

    print(f"[INFO] Input : {INPUT_JSONL}")
    print(f"[INFO] Output: {OUTPUT_JSONL}")

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
         OUTPUT_JSONL.open("a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                fout.write(json.dumps({
                    "line": idx,
                    "error": "json_parse_failed",
                    "raw": line[:300],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            if "question" not in item or not isinstance(item["question"], str):
                fout.write(json.dumps({
                    "line": idx,
                    "error": "missing_question",
                    "keys": list(item.keys()),
                }, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            try:
                result = answer_one_open(item)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{idx}] DONE | {item['question'][:80]}")

            except Exception as e:
                fout.write(json.dumps({
                    "line": idx,
                    "question": item.get("question", "")[:300],
                    "error": str(e),
                    "traceback": traceback.format_exc()[-4000:],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{idx}] ERROR: {e}")

    print(f"\n✅ Saved results to: {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
