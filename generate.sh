set -euo pipefail
cd /djlee

mkdir -p repo venv logs outputs

# -------------------------
# 1) OS 패키지 설치
# -------------------------
apt-get update
apt-get install -y python3 python3-venv python3-pip git curl

# -------------------------
# 2) repo clone/pull
# -------------------------
if [ ! -d "repo/.git" ]; then
  git clone https://github.com/djlee1216/RAG-pipeline.git repo
else
  git -C repo pull
fi

# -------------------------
# 3) venv + requirements
# -------------------------
if [ ! -x "venv/bin/python" ]; then
  python3 -m venv venv
fi

venv/bin/pip install -U pip
if [ -f "repo/requirements.txt" ]; then
  venv/bin/pip install -r repo/requirements.txt
else
  echo "ERROR: repo/requirements.txt not found"
  exit 1
fi

# -------------------------
# 4) vLLM 설치
# -------------------------
if [ ! -x "venv/bin/vllm" ]; then
  echo "[SETUP] Installing vLLM..."
  venv/bin/pip install -U vllm
fi

# -------------------------
# 5) vLLM 서버 시작
# -------------------------
mkdir -p logs

# 기존 프로세스 정리
pkill -f "vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507" || true
pkill -f "vllm serve BAAI/bge-m3" || true

# 종료 시 자동 정리 함수
cleanup() {
  echo "[CLEANUP] Stopping vLLM servers..."
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507" || true
  pkill -f "vllm serve BAAI/bge-m3" || true
}
# trap cleanup EXIT

# LLM server 실행
nohup venv/bin/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --dtype auto --max-model-len 16000 \
  --seed 42 --gpu_memory_utilization 0.5 \
  > logs/vllm_llm.log 2>&1 &

# Embedding server 실행
nohup venv/bin/vllm serve BAAI/bge-m3 \
  --host 0.0.0.0 --port 8001 \
  --dtype auto --max-model-len 8192 \
  --seed 42 --gpu_memory_utilization 0.4 \
  > logs/vllm_embed.log 2>&1 &

echo "[INFO] vLLM processes started. Waiting for servers to load models..."

# -------------------------
# 6) 서버 헬스체크 (로딩 대기)
# -------------------------
echo "[WAIT] LLM 서버(8000)가 응답할 때까지 대기합니다..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    echo -n "."
    sleep 10
done
echo -e "\n✅ LLM Server Ready!"

echo "[WAIT] Embedding 서버(8001)가 응답할 때까지 대기합니다..."
while ! curl -s http://localhost:8001/v1/models > /dev/null; do
    echo -n "."
    sleep 5
done
echo -e "\n✅ Embedding Server Ready!"

# -------------------------
# 8) 메인 파이썬 실행
# -------------------------
echo "[INFO] Running main evaluation..."
cd /djlee/repo
../venv/bin/python run_eval_baseline.py | tee ../logs/run_eval_baseline.py

echo "✅ All tasks completed."
