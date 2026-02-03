# import os
# import traceback
# from src.query import Query
# from src.generate import generate, check_question
# from src.LLMs.LLM import submit_prompt_flex
# import traceback
# import git
# import asyncio
# import time


# #folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
# #clone_directory = "./3GPP-Release18"

# ### 내가 사용할 모델###
# model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# # model_embedding = "BAAI/bge-m3"
# #####################
# k= 5
# def print_topk_and_answer(question, response, k=k):

#     print("\n")
#     print("\n")
#     print("\n")
#     marker = "[Terms & Definitions]:"
#     idx = question.question.find(marker)

#     if idx != -1:
#         print(question.question[idx:])

#     print("-" * 60)

#     print("[Top-k Retrieved Contexts]\n")

#     retrievals = getattr(question, "retrievals", None)
#     if not retrievals:
#         # fallback: context 문자열만 있는 경우
#         ctx = getattr(question, "context", None)
#         if isinstance(ctx, list):
#             for i, c in enumerate(ctx[:k], 1):
#                 print(f"({i}) {c[:1500]}){' ...' if len(c) > 500 else ''}\n")
#         else:
#             print("(no structured retrievals)\n")
#     else:
#         for r in retrievals[:k]:
#             meta = r.get("meta", {}) or {}
#             stype = r.get("source_type", "unknown")
#             score = r.get("score", 0.0)
#             preview = r.get("preview", "")

#             if stype == "web":
#                 rel_path = meta.get("rel_path", "")
#                 site = rel_path.split("/")[0] if rel_path else "web"
#                 print(f"({r['rank']}) [WEB] {site}  (score={score:.4f})")
#                 if rel_path:
#                     print(f"    path: {rel_path}")
#                 print(f"    {preview}\n")

#             elif stype == "3gpp":
#                 doc = meta.get("doc_title") or meta.get("source_file", "3gpp")
#                 section = f"{meta.get('section_id','')} {meta.get('section_title','')}".strip()
#                 print(f"({r['rank']}) [3GPP] {doc}  (score={score:.4f})")
#                 if section:
#                     print(f"    section: {section}")
#                 print(f"    {preview}\n")

#             else:
#                 print(f"({r['rank']}) [UNKNOWN] (score={score:.4f})")
#                 print(f"    {preview}\n")

#     print("-" * 60)
#     print("Answer:\n")
#     print(response.strip())
#     print("-" * 60 + "\n")

# def TelcoRAG(query, answer= None, options= None, model_name=model_name):
#     try:
#         question = Query(query, [])
        
#         # 메서드가 self.query를 쓰므로 먼저 세팅
#         question.query = question.question

#         # 메서드 호출
#         question.def_TA_question()

#         # 메서드 결과를 question.question에 반영
#         question.question = question.enhanced_query

#         question.get_3GPP_context(k=20, model_name=model_name, validate_flag=False) # 근거 문맥을 3GPP 문서에서 뽑아오는 단계
                
#         ## 20개중 상위 5개만 사용 ##
                
#         if answer is not None:
#             response, context, _ = check_question(question, answer, options, model_name=model_name)
#             print_topk_and_answer(question, response, k=5)
#             return response, question.context
#         else:
#             response, context, _ = generate(question, model_name)
#             print_topk_and_answer(question, response, k=5)
#             return response, context
        
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print(traceback.format_exc())

# def answer_one(question: str) -> str:
#     # retrieval + prompt + llm 호출
#     user_q = question.strip()
#     TelcoRAG(user_q, model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")
    

# if __name__ == "__main__":
#     while True:
#         user_q = input("Enter your question (type 'exit' to quit): ").strip()

#         if not user_q:
#             continue
#         if user_q.lower() in ("exit", "quit", "q"):
#             print("Bye.")
#             break

#         TelcoRAG(user_q, model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")
import os
import traceback
import time

from src.query import Query
from src.generate import generate, check_question

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

K_RETRIEVAL = 20   # 검색은 20개
K_CONTEXT   = 5    # LLM에는 상위 5개만

def print_topk_and_answer(question, response, k=K_CONTEXT):
    print("\n\n\n")
    marker = "[Terms & Definitions]:"
    idx = question.question.find(marker)
    if idx != -1:
        print(question.question[idx:])

    print("-" * 60)
    print("[Top-k Retrieved Contexts]\n")

    # question.context 기준으로 출력 (LLM에 들어간 것과 동일하게 보이도록)
    ctx = getattr(question, "context", None)
    if isinstance(ctx, list):
        for i, c in enumerate(ctx[:k], 1):
            print(f"({i}) {c[:1500]}{' ...' if len(c) > 1500 else ''}\n")
    elif isinstance(ctx, str):
        print(ctx[:1500])
    else:
        print("(no context)\n")

    print("-" * 60)
    print("Answer:\n")
    print((response or "").strip())
    print("-" * 60 + "\n")


def TelcoRAG(query, answer=None, options=None, model_name=model_name):
    try:
        start = time.time()

        question = Query(query, [])

        # def_TA_question()이 self.query를 쓰므로 먼저 세팅
        question.query = question.question

        # Terms/Abbrev 붙인 enhanced_query 생성
        question.def_TA_question()

        # 질문 본문에 enhanced query 반영
        question.question = question.enhanced_query

        # ✅ 1) retrieval은 20개
        question.get_3GPP_context(k=K_RETRIEVAL, model_name=model_name, validate_flag=False)

        # ✅ 2) LLM에는 상위 5개만
        if hasattr(question, "context") and isinstance(question.context, list):
            question.context = question.context[:K_CONTEXT]

              # 이제 generate()/check_question()은 줄어든 question.context만 보고 프롬프트를 만듦
        if answer is not None:
            ok, pred, used_prompt = check_question(question, answer, options, model_name=model_name)
            # check_question 반환 형식이 (bool, "Option [...] ", prompt) 이라 response는 pred 사용
            response = pred
            print_topk_and_answer(question, response, k=K_CONTEXT)
            print(f"Elapsed: {time.time()-start:.2f}s")
            return response, question.context
        else:
            response, context_str, _ = generate(question, model_name)
            print_topk_and_answer(question, response, k=K_CONTEXT)
            print(f"Elapsed: {time.time()-start:.2f}s")
            return response, context_str

    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    while True:
        user_q = input("Enter your question (type 'exit' to quit): ").strip()
        if not user_q:
            continue
        if user_q.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        TelcoRAG(user_q, model_name=model_name)
