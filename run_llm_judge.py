import json
import os
import requests
import numpy as np

# 경로 설정
INPUT_PATH = "/djlee/outputs/3gpp_rag_eval_qa_100_answers_baseline.jsonl"
OUTPUT_PATH = "/djlee/outputs/3gpp_rag_eval_final_scored.jsonl"
API_URL = "http://localhost:8000/v1/chat/completions"

def get_critique_score(question, answer):
    prompt = f"""당신은 3GPP 통신 표준 기술 전문 심사위원입니다.
아래의 [질문]에 대해 제공된 [시스템 답변]의 기술적 완성도를 채점하세요.

[질문]: {question}
[시스템 답변]: {answer}

다음 4가지 기준(각 25점)에 따라 점수를 매기세요:
1. 기술적 정확성: 3GPP 표준 규격과 논리적으로 일치하는가?
2. 구체성: 구체적인 절차, 파라미터, 프로토콜 명칭을 사용했는가?
3. 완결성: 질문의 의도에 대해 빠진 부분 없이 설명했는가?
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
        result = json.loads(response.json()['choices'][0]['message']['content'])
        return result
    except:
        return None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        return

    scored_results = []
    scores = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        data = json.loads(line)
        q, a = data.get('question', ''), data.get('answer', '')
        
        res = get_critique_score(q, a)
        
        if res:
            # 요청하신 대로 순서를 강제한 새로운 딕셔너리 생성
            ordered_data = {
                "final_total_score": res.get('total_score'),
                "question": q,
                "answer": a,
                "detail_scores": {
                    "accuracy": res.get('accuracy'),
                    "specificity": res.get('specificity'),
                    "completeness": res.get('completeness'),
                    "clarity": res.get('clarity')
                },
                "analysis": res.get('analysis')
            }
            scored_results.append(ordered_data)
            scores.append(res.get('total_score'))
            print(f"[{i+1}/{len(lines)}] Score: {res.get('total_score')}")

    # 결과 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in scored_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if scores:
        print(f"\n📊 평가 완료 | 평균 점수: {np.mean(scores):.2f} / 100")
        print(f"📂 저장 경로: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
