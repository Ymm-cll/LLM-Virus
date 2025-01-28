import json
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


file_path = 'dataset/advbench.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

prompts = []
for line in tqdm(lines):
    try:
        data = eval(line)
        prompt = data['prompt']
        if prompt:
            prompts.append(prompt)
    except json.JSONDecodeError:
        print(f"Skipping invalid line: {line.strip()}")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(prompts)

k = 13


kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(embeddings)


cluster_centers = kmeans.cluster_centers_


closest_prompts = []

for i in range(k):
    cluster_center = cluster_centers[i]

    similarities = cosine_similarity([cluster_center], embeddings)[0]

    closest_idx = np.argmax(similarities)

    closest_prompts.append((prompts[closest_idx], closest_idx, similarities[closest_idx]))


for i, (prompt, idx, similarity) in enumerate(closest_prompts):
    print(f"Cluster {i + 1}:")
    print(f"  Most similar prompt: {prompt}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Original index: {idx}")
    print("-" * 50)

with open('dataset/advbench_local.jsonl', 'w', encoding='utf-8') as f:
    for item in closest_prompts:
        f.write(str({"oid": item[1].item(), "prompt": item[0]}) + '\n')
