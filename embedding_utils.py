from sentence_transformers import SentenceTransformer

# 选择一个小型 embedding 模型，本地生成向量
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """
    输入列表 of str，返回对应向量
    """
    return model.encode(texts, convert_to_numpy=True)