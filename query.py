# import os
# import json
# import numpy as np
# import torch
# import traceback
# from tqdm.auto import tqdm
# from torch.utils.data import DataLoader, TensorDataset
# from retrieval import find_nearest_neighbors_faiss
# from index import get_faiss_batch_index
# from pdf_reader import fetch_snippets_and_search
# from embeddings import get_embeddings
# from get_definitions import define_TA_question
# from input import get_documents
# from chunking import chunk_doc
# from LLM import submit_prompt_flex, a_submit_prompt_flex, embedding
# from validator import validator_RAG
# from NNRouter import NNRouter
# from LLM2 import a_submit_prompt_flex_UI, submit_prompt_flex_UI
# import json
# from pathlib import Path
# from typing import List, Dict

# import numpy as np
# import faiss
# import torch
# from transformers import AutoTokenizer, AutoModel

# class Query:
#     def __init__(self, query, context):
#         self.question = query
#         self.query = query 
#         self.enhanced_query = query
#         self.context = [context] if isinstance(context, str) else context
#         self.context_source = []
#         self.wg = []
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = NNRouter()
#         resources_dir = os.path.join(os.path.dirname(__file__), "resources") #라우팅 모델을 파일에서 읽어옴.
#         router_path = os.path.join(resources_dir, "router_new.pth")
#         self.model.load_state_dict(torch.load(router_path, map_location='cpu'))
#         self.model.to(self.device)
#         self.model.eval()
#         self.original_labels_mapping = np.arange(21, 39)

#     def def_TA_question(self):
#         self.query = define_TA_question(self.query)
#         self.enhanced_query = self.query

#     def candidate_answers(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', UI_flag=True):
#         try:
#             if isinstance(getattr(self, "context", None), list):
#                 context_str = "\n".join(self.context)
#             else:
#                 context_str = str(getattr(self, "context", ""))

#             prompt = f"""
#     You are a telecommunications / 3GPP domain expert.
#     Answer the question using the retrieved context below.
#     If the context is insufficient, say "제공된 자료로는 답변할 수 없습니다."


#     Question:
#     {self.query}

#     Retrieved context:
#     {context_str}

#     Answer:
#     """.strip()
#             print(1)
#             # verbose=False 로 조용히 호출
#             if UI_flag:
#                 generated_output_str = submit_prompt_flex_UI(prompt, model=model_name)  # UI버전도 print 제거 필요
#             else:
#                 generated_output_str = submit_prompt_flex(prompt, model=model_name, verbose=False)

#             self.answer = (generated_output_str or "").strip()

#         except Exception as e:
#             self.answer = "ERROR: answer generation failed."




#     @staticmethod
#     def get_embeddings_list(text_list):
#         response = embedding(text_list)
#         embeddings = [item.embedding for item in response.data]
#         return dict(zip(text_list, embeddings))

#     @staticmethod
#     def inner_product(a, b):
#         return sum(x * y for x, y in zip(a, b))
    
#     @staticmethod
#     def get_col2(embeddings_list):
#         resources_dir = os.path.join(os.path.dirname(__file__), "resources")
#         file_path = os.path.join(resources_dir, "series_description.json")
#         if os.path.isfile(file_path):
#             with open(file_path, 'r') as file:
#                 series_dict = json.load(file)
#         else:
#             topics_with_series = [
#                 ("Requirements (21 series): Focuses on the overarching requirements necessary for UMTS (Universal Mobile Telecommunications System) and later cellular standards, including GSM enhancements, security standards, and the general evolution of 3GPP systems.", "21 series"),
#                 ("Service aspects ('stage 1') (22 series): This series details the initial specifications for services provided by the network, outlining the service requirements before the technical realization is detailed.", "22 series"),
#                 ("Technical realization ('stage 2') (23 series): Focuses on the architectural and functional framework necessary to implement the services described in stage 1, providing a bridge to the detailed protocols and interfaces defined in stage 3.", "23 series"),
#                 ("Signalling protocols ('stage 3') - user equipment to network (24 series): Details the protocols and signaling procedures for communication between user equipment and the network, ensuring interoperability and successful service delivery.", "24 series"),
#                 ("Radio aspects (25 series): Covers the specifications related to radio transmission technologies, including frequency bands, modulation schemes, and antenna specifications, critical for ensuring efficient and effective wireless communication.", "25 series"),
#                 ("CODECs (26 series): Contains specifications for voice, audio, and video codecs used in the network, defining how data is compressed and decompressed to enable efficient transmission over bandwidth-limited wireless networks.", "26 series"),
#                 ("Data (27 series): This series focuses on the data services and capabilities of the network, including specifications for data transmission rates, data service features, and support for various data applications.", "27 series"),
#                 ("Signalling protocols ('stage 3') - (RSS-CN) and OAM&P and Charging (overflow from 32.- range) (28 series): Addresses additional signaling protocols related to operation, administration, maintenance, provisioning, and charging, complementing the core signaling protocols outlined in the 24 series.", "28 series"),
#                 ("Signalling protocols ('stage 3') - intra-fixed-network (29 series): Specifies signaling protocols used within the fixed parts of the network, ensuring that various network elements can communicate effectively to provide seamless service to users.", "29 series"),
#                 ("Programme management (30 series): Relates to the management and coordination of 3GPP projects and work items, including documentation and specification management procedures.", "30 series"),
#                 ("Subscriber Identity Module (SIM / USIM), IC Cards. Test specs. (31 series): Covers specifications for SIM and USIM cards, including physical characteristics, security features, and interaction with mobile devices, as well as testing specifications for these components.", "31 series"),
#                 ("OAM&P and Charging (32 series): Focuses on operation, administration, maintenance, and provisioning aspects of the network, as well as the charging principles and mechanisms for billing and accounting of network services.", "32 series"),
#                 ("Security aspects (33 series): Details the security mechanisms and protocols necessary to protect network operations, user data, and communication privacy, including authentication, encryption, and integrity protection measures.", "33 series"),
#                 ("UE and (U)SIM test specifications (34 series): Contains test specifications for User Equipment (UE) and (U)SIM cards, ensuring that devices and SIM cards meet 3GPP standards and perform correctly in the network.", "34 series"),
#                 ("Security algorithms (35 series): Specifies the cryptographic algorithms used in the network for securing user data and signaling information, including encryption algorithms and key management procedures.", "35 series"),
#                 ("LTE (Evolved UTRA), LTE-Advanced, LTE-Advanced Pro radio technology (36 series): Details the technical specifications for LTE, LTE-Advanced, and LTE-Advanced Pro technologies, including radio access network (RAN) protocols, modulation schemes, and network architecture.", "36 series"),
#                 ("Multiple radio access technology aspects (37 series): Addresses the integration and interoperability of multiple radio access technologies within the network, enabling seamless service across different types of network infrastructure.", "37 series"),
#                 ("Radio technology beyond LTE (38 series): Focuses on the development and specification of radio technologies that extend beyond the capabilities of LTE, aiming to improve speed, efficiency, and functionality for future cellular networks.", "38 series")
#             ]
#             series_dict = {index: {"description": desc, "embeddings": Query.get_embeddings(desc)} for desc, index in topics_with_series}
#             with open(file_path, 'w') as file:
#                 json.dump(series_dict, file, indent=4)
        
#         similarity_column = []
#         for embeddings in embeddings_list:
#             coef = [Query.inner_product(embeddings, series_dict[series_id]['embeddings']) for series_id in series_dict]
#             similarity_column.append(coef)
#         return similarity_column
    
#     @staticmethod
#     def preprocessing_softmax(embeddings_list):
#         embeddings = np.array(embeddings_list)
#         similarity = np.array(Query.get_col2(embeddings))
#         X_train_1_tensor = torch.tensor(embeddings, dtype=torch.float32)
#         X_train_2_tensor = torch.nn.functional.softmax(10 * torch.tensor(similarity, dtype=torch.float32), dim=-1)
#         dataset = TensorDataset(X_train_1_tensor, X_train_2_tensor)
#         return DataLoader(dataset, batch_size=128, shuffle=True)
    
#     @staticmethod
#     def get_embeddings(text):
#         response = embedding(text)
#         return response.data[0].embedding

#     def predict_wg(self):
#         text_embeddings = Query.get_embeddings_list([self.enhanced_query])
#         embeddings = text_embeddings[self.enhanced_query]
#         test_dataloader = Query.preprocessing_softmax([embeddings])
#         label_list = []
#         with torch.no_grad():
#             for X1, X2 in test_dataloader:
#                 X1, X2 = X1.to(self.device), X2.to(self.device)
#                 outputs = self.model(X1, X2)
#                 _, top_indices = outputs.topk(5, dim=1)
#                 predicted_labels = self.original_labels_mapping[top_indices.cpu().numpy()]
#                 label_list = predicted_labels
#         self.wg = label_list[0]
        
#     def get_question_context_faiss(self, batch=None, k=20, use_context=False):
#         import os, json, traceback
#         import faiss

#         try:
#             # ===== 경로 
#             FAISS_INDEX_PATH = "/NAS/inno_aidev/users/djlee/faiss_indexes/web_corpus_chunk.faiss"
#             META_JSONL_PATH  = "/NAS/inno_aidev/users/djlee/faiss_indexes/web_corpus_chunk.meta.jsonl"

#             # ===== FAISS index 로드 =====
#             if not hasattr(self, "faiss_index") or self.faiss_index is None:
#                 if not os.path.exists(FAISS_INDEX_PATH):
#                     raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
#                 self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)

#             # ===== META jsonl 로드 =====
#             if not hasattr(self, "meta_lines") or self.meta_lines is None:
#                 if not os.path.exists(META_JSONL_PATH):
#                     raise FileNotFoundError(f"Meta jsonl not found: {META_JSONL_PATH}")

#                 meta_lines = []
#                 with open(META_JSONL_PATH, "r", encoding="utf-8") as f:
#                     for line in f:
#                         line = line.strip()
#                         if not line:
#                             continue
#                         meta_lines.append(json.loads(line))
#                 self.meta_lines = meta_lines  # list[dict]

#             # ===== bge-m3 로드 =====
#             if not hasattr(self, "bge_tokenizer") or not hasattr(self, "bge_model"):
#                 import torch
#                 from transformers import AutoTokenizer, AutoModel
#                 MODEL_NAME = "BAAI/bge-m3"
#                 self._bge_device = "cuda" if torch.cuda.is_available() else "cpu"
#                 self.bge_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#                 self.bge_model = AutoModel.from_pretrained(MODEL_NAME).to(self._bge_device)
#                 self.bge_model.eval()

#             # ===== 검색 =====
#             ctx_for_query = self.context if (use_context and isinstance(getattr(self, "context", None), list)) else None

#             hits = find_nearest_neighbors_faiss(
#                 query_text=self.query,
#                 faiss_index=self.faiss_index,
#                 k=k,
#                 context=ctx_for_query,
#                 tokenizer=self.bge_tokenizer,
#                 model=self.bge_model,
#                 device=self._bge_device
#             )
#             # hits: [(idx, score), ...]

#             # ===== (5) 결과 정리: self.retrievals + self.context =====
#             self.retrievals = []
#             self.context = []

#             for rank, (idx, score) in enumerate(hits, start=1):
#                 meta = self.meta_lines[idx] if 0 <= idx < len(self.meta_lines) else {}

#                 text_path = meta.get("text_path", "")
#                 chunk_text = ""
#                 if text_path and os.path.exists(text_path):
#                     with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
#                         chunk_text = f.read()

#                 # ---- 500자 preview ----
#                 preview = chunk_text.replace("\n", " ").strip()
#                 if len(preview) > 500:
#                     preview = preview[:1500] + " ..."

#                 # ---- 타입 구분 (web vs 3gpp) ----
#                 if meta.get("source_type") == "3gpp" or "doc_title" in meta:
#                     source_type = "3gpp"
#                 elif meta.get("chunk_tag") == "web_corpus_chunk" or "rel_path" in meta:
#                     source_type = "web"
#                 else:
#                     source_type = "unknown"

#                 # ---- 출력/후처리용 구조화 저장 ----
#                 self.retrievals.append({
#                     "rank": rank,
#                     "index": int(idx),
#                     "score": float(score),
#                     "source_type": source_type,
#                     "meta": meta,
#                     "preview": preview,
#                 })

#                 # ---- LLM 입력용 context (전체 텍스트 넣고 싶으면 chunk_text 사용) ----
#                 self.context.append(
#                     f"[Retrieval {rank}] (score={float(score):.4f}, type={source_type}, id={idx})\n{chunk_text}\n"
#                 )

#         except Exception as e:
#             print(f"[get_question_context_faiss] error: {e}")
#             print(traceback.format_exc())
#             self.retrievals = []
#             self.context = ["ERROR: retrieval failed."]


    
#     def validate_context(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', k=5, UI_flag=True):
#         self.context = validator_RAG(self.question, self.context, model_name=model_name, k=k, UI_flag=UI_flag)
        
#     def get_3GPP_context(self, k=20, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', validate_flag=False, UI_flag=False):
        
#         #document_ds = get_documents("/NAS/inno_aidev/users/djlee/wordfile") #해당 시리즈 문서 로딩
#         #Document_ds = [chunk_doc(doc) for doc in document_ds]
#         #print(Document_ds)
   
#         #series_docs = get_embeddings(Document_ds)
  
#         # embedded_docs = [serie for serie in series_docs.values()]
#         # self.get_question_context_faiss(batch=embedded_docs, k=5, use_context=False)
#         # self.candidate_answers(model_name=model_name, UI_flag=UI_flag)
#         #embedded_docs = list(series_docs.values())
   
#        # old_list = self.wg
#        # self.predict_wg()
#        # new_series = {f'Series{series_number}': [doc for doc in Document_ds if doc[0]['source'][:2].isnumeric() and int(doc[0]['source'][:2]) == series_number] for series_number in self.wg if series_number not in old_list}
#        # new_series = get_embeddings(new_series)
#        # old_series = {'Summaries': series_docs['Summaries'], **{f'Series{series_number}': series_docs[f'Series{series_number}'] for series_number in self.wg if series_number in old_list}}
#        #embedded_docs = [serie for serie in new_series.values()] + [serie for serie in old_series.values()]
#         self.get_question_context_faiss(batch=None, k=20, use_context=True)
#         #self.candidate_answers(model_name=model_name, UI_flag=UI_flag)


#     async def get_online_context(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', validator_flag= True, options=None):
#         if options is None:
#             querytoOSINT = f"""Rephrase the following question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

#         {self.question}"""
#         else:
#             querytoOSINT = f"""Rephrase the following multiple choice question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

#             {self.question}
#     Answer options:
#     {options}"""
#         osintquery = await a_submit_prompt_flex(querytoOSINT, model=model_name)
#         print("_"*100)
#         print(osintquery)
#         try:
#             online_info = await fetch_snippets_and_search(query= osintquery.rstrip('"'), question=self.question, model_name=model_name, validator=validator_flag, UI=False)     
#         except:
#             online_info = await fetch_snippets_and_search(query= self.question, question=self.question, model_name=model_name, validator=validator_flag, UI=False)

#         return online_info
    
#     async def get_online_context_UI(self, model_name='gpt-4o-mini', validator_flag= True, options=None):
#         if options is None:
#             querytoOSINT = f"""Rephrase the fallowing question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

#         {self.question}"""
#         else:
#             querytoOSINT = f"""Rephrase the fallowing multiple choice question so that it can be a concise google search query to find the answer to my original question (O.S.I.N.T. syle)

#             {self.question}
#     Answer options:
#     {options}"""
#         osintquery = await a_submit_prompt_flex_UI(querytoOSINT, model=model_name)
#         print("_FA"*100)
#         print(osintquery)
#         try:
#             online_info = await fetch_snippets_and_search(query= osintquery.rstrip('"'), question=self.question, model_name=model_name, validator=validator_flag, UI=True)     
#         except:
#             online_info = await fetch_snippets_and_search(query= self.question, question=self.question, model_name=model_name, validator=validator_flag, UI=True)


#         return online_info
import os
import json
import numpy as np
import torch
import traceback
from pathlib import Path
from typing import List, Dict

# 프로젝트 내부 모듈 임포트
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from retrieval import find_nearest_neighbors_faiss
from index import get_faiss_batch_index
from pdf_reader import fetch_snippets_and_search
from embeddings import get_embeddings
from get_definitions import define_TA_question
from input import get_documents
from chunking import chunk_doc
from LLM import submit_prompt_flex, a_submit_prompt_flex, embedding
from validator import validator_RAG
from NNRouter import NNRouter
from LLM2 import a_submit_prompt_flex_UI, submit_prompt_flex_UI

import faiss
from transformers import AutoTokenizer, AutoModel

class Query:
    def __init__(self, query, context):
        # 기본 경로 설정: 현재 파일(query.py)의 위치를 기준으로 resources 폴더 지정
        self.base_dir = Path(__file__).resolve().parent
        self.resources_dir = self.base_dir / "resources"
        
        self.question = query
        self.query = query 
        self.enhanced_query = query
        self.context = [context] if isinstance(context, str) else context
        self.context_source = []
        self.wg = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 라우터 모델 로드 (경로 수정)
        self.model = NNRouter()
        router_path = self.resources_dir / "router_new.pth"
        
        if not router_path.exists():
            print(f"⚠️ 경고: 라우터 모델을 찾을 수 없습니다: {router_path}")
        else:
            self.model.load_state_dict(torch.load(router_path, map_location='cpu'))
        
        self.model.to(self.device)
        self.model.eval()
        self.original_labels_mapping = np.arange(21, 39)

    def def_TA_question(self):
        self.query = define_TA_question(self.query)
        self.enhanced_query = self.query

    def candidate_answers(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', UI_flag=True):
        try:
            if isinstance(getattr(self, "context", None), list):
                context_str = "\n".join(self.context)
            else:
                context_str = str(getattr(self, "context", ""))

            prompt = f"""
    You are a telecommunications / 3GPP domain expert.
    Answer the question using the retrieved context below.
    If the context is insufficient, say "제공된 자료로는 답변할 수 없습니다."

    Question:
    {self.query}

    Retrieved context:
    {context_str}

    Answer:
    """.strip()
            
            if UI_flag:
                generated_output_str = submit_prompt_flex_UI(prompt, model=model_name)
            else:
                generated_output_str = submit_prompt_flex(prompt, model=model_name, verbose=False)

            self.answer = (generated_output_str or "").strip()
        except Exception as e:
            self.answer = "ERROR: answer generation failed."

    @staticmethod
    def get_embeddings_list(text_list):
        response = embedding(text_list)
        embeddings = [item.embedding for item in response.data]
        return dict(zip(text_list, embeddings))

    @staticmethod
    def inner_product(a, b):
        return sum(x * y for x, y in zip(a, b))
    
    @staticmethod
    def get_col2(embeddings_list):
        # 이 메소드 내부에서도 resources_dir 사용
        base_dir = Path(__file__).resolve().parent
        file_path = base_dir / "resources" / "series_description.json"
        
        if file_path.exists():
            with open(file_path, 'r') as file:
                series_dict = json.load(file)
        else:
            # 기본값 설정 (생략)
            series_dict = {} 
        
        similarity_column = []
        for embeddings in embeddings_list:
            coef = [Query.inner_product(embeddings, series_dict[series_id]['embeddings']) for series_id in series_dict]
            similarity_column.append(coef)
        return similarity_column
    
    @staticmethod
    def preprocessing_softmax(embeddings_list):
        embeddings = np.array(embeddings_list)
        similarity = np.array(Query.get_col2(embeddings))
        X_train_1_tensor = torch.tensor(embeddings, dtype=torch.float32)
        X_train_2_tensor = torch.nn.functional.softmax(10 * torch.tensor(similarity, dtype=torch.float32), dim=-1)
        dataset = TensorDataset(X_train_1_tensor, X_train_2_tensor)
        return DataLoader(dataset, batch_size=128, shuffle=True)
    
    @staticmethod
    def get_embeddings(text):
        response = embedding(text)
        return response.data[0].embedding

    def predict_wg(self):
        text_embeddings = Query.get_embeddings_list([self.enhanced_query])
        embeddings = text_embeddings[self.enhanced_query]
        test_dataloader = Query.preprocessing_softmax([embeddings])
        label_list = []
        with torch.no_grad():
            for X1, X2 in test_dataloader:
                X1, X2 = X1.to(self.device), X2.to(self.device)
                outputs = self.model(X1, X2)
                _, top_indices = outputs.topk(5, dim=1)
                predicted_labels = self.original_labels_mapping[top_indices.cpu().numpy()]
                label_list = predicted_labels
        self.wg = label_list[0]
        
    def get_question_context_faiss(self, batch=None, k=20, use_context=False):
        try:
            # ===== 핵심 수정: 하드코딩된 NAS 경로 제거 및 resources 활용 =====
            FAISS_INDEX_PATH = self.resources_dir / "web_corpus_chunk.faiss"
            META_JSONL_PATH  = self.resources_dir / "web_corpus_chunk.meta.jsonl"

            # FAISS index 로드
            if not hasattr(self, "faiss_index") or self.faiss_index is None:
                if not FAISS_INDEX_PATH.exists():
                    raise FileNotFoundError(f"❌ FAISS 인덱스 파일을 찾을 수 없습니다: {FAISS_INDEX_PATH}")
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

            # META jsonl 로드
            if not hasattr(self, "meta_lines") or self.meta_lines is None:
                if not META_JSONL_PATH.exists():
                    raise FileNotFoundError(f"❌ 메타 데이터를 찾을 수 없습니다: {META_JSONL_PATH}")

                meta_lines = []
                with open(META_JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            meta_lines.append(json.loads(line))
                self.meta_lines = meta_lines

            # bge-m3 로드
            if not hasattr(self, "bge_tokenizer") or not hasattr(self, "bge_model"):
                MODEL_NAME = "BAAI/bge-m3"
                self._bge_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.bge_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.bge_model = AutoModel.from_pretrained(MODEL_NAME).to(self._bge_device)
                self.bge_model.eval()

            # 검색 실행
            ctx_for_query = self.context if (use_context and isinstance(getattr(self, "context", None), list)) else None
            hits = find_nearest_neighbors_faiss(
                query_text=self.query,
                faiss_index=self.faiss_index,
                k=k,
                context=ctx_for_query,
                tokenizer=self.bge_tokenizer,
                model=self.bge_model,
                device=self._bge_device
            )

            self.retrievals = []
            self.context = []

            for rank, (idx, score) in enumerate(hits, start=1):
                meta = self.meta_lines[idx] if 0 <= idx < len(self.meta_lines) else {}
                
                # 텍스트 추출 (경로가 유효한지 체크)
                text_path = meta.get("text_path", "")
                chunk_text = ""
                if text_path and os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
                        chunk_text = f.read()

                preview = chunk_text.replace("\n", " ").strip()
                if len(preview) > 500:
                    preview = preview[:1500] + " ..."

                source_type = "3gpp" if (meta.get("source_type") == "3gpp" or "doc_title" in meta) else "web"

                self.retrievals.append({
                    "rank": rank,
                    "index": int(idx),
                    "score": float(score),
                    "source_type": source_type,
                    "meta": meta,
                    "preview": preview,
                })

                self.context.append(
                    f"[Retrieval {rank}] (score={float(score):.4f}, type={source_type})\n{chunk_text}\n"
                )

        except Exception as e:
            print(f"[get_question_context_faiss] error: {e}")
            self.retrievals = []
            self.context = ["ERROR: retrieval failed."]

    def validate_context(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', k=5, UI_flag=True):
        self.context = validator_RAG(self.question, self.context, model_name=model_name, k=k, UI_flag=UI_flag)
        
    def get_3GPP_context(self, k=20, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', validate_flag=False, UI_flag=False):
        self.get_question_context_faiss(batch=None, k=k, use_context=True)

    async def get_online_context(self, model_name='Qwen/Qwen3-30B-A3B-Instruct-2507', validator_flag=True, options=None):
        querytoOSINT = f"Rephrase for Google Search: {self.question}"
        osintquery = await a_submit_prompt_flex(querytoOSINT, model=model_name)
        try:
            online_info = await fetch_snippets_and_search(query=osintquery.rstrip('"'), question=self.question, model_name=model_name, validator=validator_flag, UI=False)     
        except:
            online_info = await fetch_snippets_and_search(query=self.question, question=self.question, model_name=model_name, validator=validator_flag, UI=False)
        return online_info
    
    
