import numpy as np
import os
import traceback
from LLM import embedding
import logging

def search_faiss_index(faiss_index, query_embedding, k=5):
    import numpy as np

    if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D numpy array")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    query_embedding_reshaped = query_embedding.reshape(1, -1).astype(np.float32)

    # FAISS: returns (D, I)
    D, I = faiss_index.search(query_embedding_reshaped, k)

    # 기존 코드 호환을 위해 (I, D)로 리턴
    return I, D


def get_query_embedding_OpenAILarge(query_text, context=None, tokenizer=None, model=None, device="cpu"):
    import traceback

    try:
        if tokenizer is None or model is None:
            raise ValueError("bge-m3 tokenizer/model is required (tokenizer/model is None).")

        if context is not None:
            if isinstance(context, list):
                query_text = f"{query_text}\n" + "\n".join(context)
            else:
                query_text = f"{query_text}\n{str(context)}"

        import torch
        with torch.no_grad():
            inputs = tokenizer(
                [query_text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # (1, T, H)

            mask = inputs["attention_mask"].unsqueeze(-1).type_as(last_hidden)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts

            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        return emb[0].detach().cpu().numpy().astype("float32")

    except Exception as e:
        print(f"[get_query_embedding_OpenAILarge] error: {e}")
        traceback.print_exc()
        raise

             
def find_nearest_neighbors_faiss(
    query_text,
    faiss_index,
    k,
    context=None,
    tokenizer=None,
    model=None,
    device=None
):
    import numpy as np
    import traceback
    import torch

    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        query_embedding = get_query_embedding_OpenAILarge(
            query_text=query_text,
            context=context,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        # FAISS 검색
        I, D = search_faiss_index(faiss_index, query_embedding, k)

        out = []
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            out.append((int(idx), float(D[0][rank])))
        return out

    except Exception as e:
        print(f"[find_nearest_neighbors_faiss] error: {str(e)}")
        traceback.print_exc()
        return []

 
