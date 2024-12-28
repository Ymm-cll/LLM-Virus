import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# 读取数据
file_path = 'dataset/advbench.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 提取所有的prompt文本
prompts = []
for line in tqdm(lines):
    try:
        # 解析JSON字符串为字典
        data = eval(line)
        prompt = data['prompt']
        if prompt:
            prompts.append(prompt)
    except json.JSONDecodeError:
        print(f"Skipping invalid line: {line.strip()}")

# 加载SentenceTransformer模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# 获取prompts的嵌入向量
embeddings = model.encode(prompts)

# 聚类的数量k
k = 13  # 你可以根据需求修改k的值

# 使用KMeans聚类
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(embeddings)

# 获取聚类中心
cluster_centers = kmeans.cluster_centers_

# 找出每个聚类中心最接近的prompt
closest_prompts = []

for i in range(k):
    # 获取当前聚类中心
    cluster_center = cluster_centers[i]

    # 计算所有文本到当前聚类中心的余弦相似度
    similarities = cosine_similarity([cluster_center], embeddings)[0]

    # 找出相似度最高的文本索引
    closest_idx = np.argmax(similarities)

    # 存储最接近中心的文本及其索引
    closest_prompts.append((prompts[closest_idx], closest_idx, similarities[closest_idx]))

# 打印每个聚类中心及其最接近的文本
for i, (prompt, idx, similarity) in enumerate(closest_prompts):
    print(f"Cluster {i + 1}:")
    print(f"  Most similar prompt: {prompt}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Original index: {idx}")
    print("-" * 50)

with open('dataset/advbench_local.jsonl', 'w', encoding='utf-8') as f:
    for item in closest_prompts:
        f.write(str({"oid": item[1].item(), "prompt": item[0]}) + '\n')
