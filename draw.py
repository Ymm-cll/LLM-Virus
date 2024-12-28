import numpy as np
import methods
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
# matplotlib.use('TkAgg')


def linechart(eids, targets):
    success_rates = []
    lengths = []

    for target, eid in zip(targets, eids):
        print(target)
        with open(f"log/{target}/{eid}.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            data.append(eval(line))

        success_rate = {"max": [], "average": [], "min": []}
        length = {"max": [], "average": [], "min": []}

        for item in data:
            generation_sr = item["generation"]
            sr = []
            le = []
            for item1 in generation_sr:
                sr.append(item1["success_rate"])
                le.append(item1["length"])
            success_rate["max"].append(np.max(sr).item())
            success_rate["average"].append(np.average(sr).item())
            success_rate["min"].append(np.min(sr).item())
            length["average"].append(-np.mean(le).item())

        success_rates.append(success_rate["average"])
        lengths.append(length["average"])

    num_rounds = len(success_rates[0])

    # Create subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 10))  # 18 for width, 10 for height

    markers = {
        "gpt": "o",
        "claude": "s",
        "llama": "v",
        "gemma": "^"
    }
    plt.rcParams['font.family'] = 'Comic Sans MS'
    colors = plt.get_cmap('tab20', len(targets))  # 获取颜色映射

    for i, target in enumerate(targets):
        if "gpt" in target:
            target_marker = "gpt"
        if "llama" in target:
            target_marker = "llama"
        if "gemma" in target:
            target_marker = "gemma"
        if target == "claude-3-5-haiku-20241022":
            target = "claude-3.5-haiku-20241022"
        if target == "llama3.1-8b":
            target = "llama-3.1-8b"
        if target == "gemma2-9b":
            target = "gemma-2-9b"
        if target == "gemma2-27b":
            target = "gemma-2-27b"

        if "claude" in target:
            target = target[:-9] + "\n" + target[-9:]
            target_marker = "claude"

        # Plot on the first subplot (success rates)
        ax1.plot(range(num_rounds), success_rates[i], label=target, marker=markers[target_marker],
                 markersize=10, linewidth=3, color=colors(i))

        # Plot on the second subplot (average lengths)
        ax2.plot(range(num_rounds), lengths[i], label=target, marker=markers[target_marker],
                 markersize=10, linewidth=3, color=colors(i))

    # Customize the first subplot (Success Rate)
    fig.suptitle("Attack Success Rate / Template Length\nover Evolution Generation", fontsize=20, fontweight="bold")
    ax1.set_xlabel("Generation", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Attack Success Rate", fontsize=16, fontweight="bold")
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_xticks(range(0, 11, 1))
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([i * 0.1 for i in range(11)])
    # ax1.legend(title="Models", fontsize=14, title_fontsize=14, framealpha=0.3)
    ax1.grid(True)

    # Customize the second subplot (Average Length)
    ax2.set_xlabel("Generation", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Average Length", fontsize=16, fontweight="bold")
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_xticks(range(0, 11, 1))
    # ax2.set_ylim(0, np.max(lengths) + 10)  # Adjust y-axis limit based on maximum average length
    # ax2.set_yticks(np.linspace(0, np.max(lengths) + 10, 11))
    ax2.legend(title="Models", fontsize=14, title_fontsize=14, framealpha=0.3)
    ax2.grid(True)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure
    plt.savefig("./figure/evolution_and_length.png", dpi=450, bbox_inches='tight')
    # plt.show()  # Uncomment if you want to display the plot immediately


def scatter(eid, target):
    temp = methods.get_geneartions(target, eid)
    generations = []
    for generation in temp:
        temp_gen = []
        for item in generation:
            temp_gen.append(item["prompt"])
        generations.append(temp_gen)

    # 加载 SentenceTransformer 模型
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 获取每个generation的文本的嵌入向量
    embeddings_list = []
    labels = []
    for i, generation in enumerate(generations):
        embeddings = model.encode(generation)
        embeddings_list.append(embeddings)
        labels.extend([i] * len(generation))  # 为每个文本分配一个 generation 的标签

    # 合并所有 generation 的嵌入向量
    embeddings = np.vstack(embeddings_list)
    print(embeddings.shape)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(embeddings_2d.shape)
    print(embeddings_2d)
    # 可视化结果
    plt.figure(figsize=(8, 6))

    # 使用 colormap 生成不同颜色
    colors = plt.get_cmap('tab20', len(generations))  # 获取颜色映射

    # 为每个 generation 绘制散点图，使用不同的颜色
    for i in range(len(generations)):
        indices = np.array(labels) == i
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Generation {i}', s=50, color=colors(i), marker="x")

    # 设置图表标题和标签
    plt.title('Text Embeddings Visualization (2D)', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(loc="lower right", framealpha=0.5)

    # 在 Jupyter 中通常不需要调用 plt.show()
    plt.show()


if __name__ == "__main__":
    eids = [3, 2, 1, 1, 1, 2, 1, 1, 2, 1]
    targets = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-0613", "claude-3-haiku-20240307", "claude-3-5-haiku-20241022", "llama-3.1-70b", "llama3.1-8b", "gemma2-27b", "gemma2-9b"]
    linechart(eids, targets)
    # scatter(3, "gpt-3.5-turbo")