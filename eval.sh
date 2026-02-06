set -euo pipefail

# 1) 작업 디렉토리 이동
cd /djlee/repo

# 2) Llama-3.1-70B-FP8 서버 실행
nohup /djlee/venv/bin/vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --host 0.0.0.0 --port 8000 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    > /djlee/logs/vllm_judge.log 2>&1 &

# 3) 서버 로딩 대기
echo "[WAIT] Llama-70B Server loading..."
START_TIME=$(date +%s)

while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$(( (CURRENT_TIME - START_TIME) / 60 ))
    echo -n "로딩 중... (${ELAPSED}분 경과) "
    sleep 30
done

echo -e "\n✅ Judge Server Ready! 평가를 시작합니다."

# 4) 평가 파이썬 스크립트 실행
/djlee/venv/bin/python /djlee/repo/run_llm_judge.py
