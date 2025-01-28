import os
from openai import OpenAI
from datasets import load_dataset
import random
from local_llm import LocalLLM


def get_client(type):
    client = None
    if type == "api":
        client = OpenAI(api_key="xxx", base_url="xxx")
    if type == "local":
        client = LocalLLM()
    if type == "llama":
        client = OpenAI(api_key="xxx",
                          base_url="xxx")
    return client


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_dataset(dataset_name, save_path, sample_num):
    random.seed(42)
    ds = None
    if dataset_name == "jailbreak":
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
    if dataset_name == "advbench":
        ds = load_dataset("walledai/AdvBench")
    if dataset_name == "jbb":
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

    if ds:
        if sample_num == None:
            sample_data = ds["harmful"] if dataset_name == "jbb" else ds["train"]
        else:
            # 采样100条数据
            sample_data = random.sample(list(ds['train']), sample_num)

        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 保存为文本文件
        with open(os.path.join(save_path, f'{dataset_name}.jsonl'), 'w', encoding="utf-8") as f:
            for item in sample_data:
                f.write(str(item) + '\n')
        print(f"Sampled data saved to {save_path}")
    else:
        print(f"Dataset {dataset_name} not found.")


def get_dataset(dataset_name):
    ds = []
    if "log" in dataset_name:
        path = dataset_name
    else:
        path = f"./dataset/{dataset_name}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            ds.append(eval(line))
    return ds


def clean_jailbreak_ds():
    ds = get_dataset("jailbreak")
    new_ds = []
    for item in ds:
        if "[INSERT PROMPT HERE]" in item["Prompt"]:
            new_ds.append(item)
    with open('dataset/jailbreak.jsonl', 'w', encoding="utf-8") as f:
        for item in new_ds:
            f.write(str(item) + '\n')


def random_pairs(lst, n):
    pairs = []
    for _ in range(n):
        pair = random.sample(lst, 2)  # 随机选择2个元素
        pairs.append(pair)
    return pairs


def get_geneartions(target, eid, bench_name):
    with open(f"log/{bench_name}/{target}/{eid}.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(eval(line))
    generations = []
    for item in data:
        generations.append(item["generation"])
    return generations


if __name__ == "__main__":
    # 示例调用
    download_dataset("jailbreak", "./dataset", sample_num=None)
    # download_dataset("advbench", "./dataset", sample_num=None)
    # download_dataset("jbb", "./dataset", sample_num=None)
    # clean_jailbreak_ds()
    # ds = get_dataset("advbench")
    # # print(ds)
