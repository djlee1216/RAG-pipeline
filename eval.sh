# 1) 작업 디렉토리 이동
cd /djlee/repo

# 2) 이전 프로세스 완전 종료 (포트 8000, 8001 점유 해제)
echo "[CLEANUP] GPU 메모리 및 포트 정리 중..."
pkill -f "vllm serve" || true
sleep 15 

# 3) Llama-3.1-70B-FP8 서버 실행
# 경로를 /djlee로 다시 맞췄습니다.
nohup /djlee/venv/bin/vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
    --host 0.0.0.0 --port 8000 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    > /djlee/logs/vllm_judge.log 2>&1 &

# 4) 서버 로딩 대기
echo "[WAIT] Llama-70B Server loading... (Qwen 30B가 12분 걸렸으니, 이건 더 걸릴 수 있습니다)"
START_TIME=$(date +%s)

while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$(( (CURRENT_TIME - START_TIME) / 60 ))
    echo -n "로딩 중... (${ELAPSED}분 경과) "
    sleep 30
done

echo -e "\n✅ Judge Server Ready! 평가를 시작합니다."

# 5) 평가 파이썬 스크립트 실행
/djlee/venv/bin/python /djlee/repo/run_llm_judge.py

# 2. 결과 파일에서 score 필드만 추출하여 50개 단위로 평균 산출
echo -e "\n"
echo "=========================================================="
echo "📊  [구간별 성능 리포트] 50문항 단위 점수 평균 분석"
echo "=========================================================="

# JSONL 파일 내의 "score": 90 또는 "score": 90.5 형태를 정확히 추출
cat /djlee/outputs/eval_result.jsonl | grep -oP '"score":\s*\K[0-9.]+' | awk '{
    sum += $1; 
    total_sum += $1;
    count++; 
    
    # 50개마다 구간 평균 출력
    if (count % 50 == 0) {
        printf "📍 구간 %3d ~ %3d 평균 점수: %6.2f / 100.00\n", count-49, count, sum/50; 
        sum = 0;
    }
} END {
    # 50개로 딱 나누어 떨어지지 않는 나머지 문항 처리
    if (count > 0) {
        if (count % 50 != 0) {
            remainder = count % 50;
            printf "📍 마지막 구간 (%d ~ %d) 평균 점수: %6.2f / 100.00\n", (int(count/50)*50)+1, count, sum/remainder;
        }
        print "----------------------------------------------------------"
        printf "✅ 전체 문항 (%d개) 최종 평균: %6.2f / 100.00\n", count, total_sum/count;
    } else {
        print "❌ 데이터를 찾을 수 없습니다. 경로(/djlee/outputs/eval_result.jsonl)를 확인하세요.";
    }
    print "=========================================================="
}'
