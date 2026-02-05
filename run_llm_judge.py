import json
import os
import requests
import numpy as np

# 절대 경로 설정
INPUT_PATH = "/djlee/outputs/3gpp_rag_eval_qa_100_answers_baseline.jsonl"
OUTPUT_PATH = "/djlee/outputs/3gpp_rag_eval_baseline_score.jsonl"
API_URL = "http://localhost:8000/v1/chat/completions"

def get_critique_score(question, answer):
    """4개 항목(각 25점) 기술 평가"""
    prompt = f"""당신은 3GPP 통신 표준 기술 전문 심사위원입니다.
아래 [질문]에 대한 [시스템 답변]의 기술적 완성도를 엄격히 채점하세요.

[질문]: {question}
[시스템 답변]: {answer}

다음 4가지 기준(각 25점)에 따라 점수를 매기세요:
1. 기술적 정확성: 답변이 3GPP 표준 규격과 논리적으로 일치하는가?
2. 구체성: 구체적인 절차, 파라미터, 프로토콜 명칭을 사용했는가?
3. 완결성: 질문의 의도에 대해 누락 없이 설명했는가?
4. 가독성: 전문 용어를 정확하게 사용하며 문장이 명확한가?

응답은 반드시 아래 JSON 형식으로만 출력하세요:
{{
    "accuracy": 점수,
    "specificity": 점수,
    "completeness": 점수,
    "clarity": 점수,
    "total_score": 합계점수,
    "analysis": "기술적 비평 요약"
}}"""

    payload = {
        "model": "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        return json.loads(response.json()['choices'][0]['message']['content'])
    except:
        return None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    scored_results = []
    scores = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"[INFO] Processing {len(lines)} samples...")

    for i, line in enumerate(lines):
        data = json.loads(line)
        q = data.get('question', '')
        a = data.get('answer', '')
        
        res = get_critique_score(q, a)
        
        if res:
            # score, question, answer 순서로 저장하기 위한 순서 고정 딕셔너리
            ordered_entry = {
                "final_total_score": res.get('total_score'),
                "question": q,
                "answer": a,
                "details": {
                    "accuracy": res.get('accuracy'),
                    "specificity": res.get('specificity'),
                    "completeness": res.get('completeness'),
                    "clarity": res.get('clarity')
                },
                "analysis": res.get('analysis')
            }
            scored_results.append(ordered_entry)
            scores.append(res.get('total_score'))
            print(f"[{i+1}/{len(lines)}] Score: {res.get('total_score')}")

    # 최종 결과 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in scored_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 결과 요약 출력
    if scores:
        print("\n" + "="*40)
        print(f"📊 Evaluation Summary")
        print(f"- Average Score: {np.mean(scores):.2f} / 100")
        print(f"- Output: {OUTPUT_PATH}")
        print("="*40)

if __name__ == "__main__":
    main()
