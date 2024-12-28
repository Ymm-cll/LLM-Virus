import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.functional import softmax
import math

import methods


def calculate_perplexity(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)

    # Get the model's output (logits) for the tokens
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        log_likelihood = outputs.loss.item()

    # Calculate perplexity (exponentiation of the negative log likelihood)
    perplexity = math.exp(log_likelihood)

    return perplexity


def main():
    target = "llama-3.1-70b"
    eid = 1
    generations = methods.get_geneartions(target, eid)
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained GPT-2 model and tokenizer
    model_path = "/data/models/GPT-2"  # You can also use "gpt2-medium", "gpt2-large", "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    sum = 0
    count = 0
    for generation in generations:
        for item in generation:
            text = item["prompt"]
            # Calculate perplexity
            perplexity = calculate_perplexity(text, model, tokenizer, device)
            print(perplexity)
            sum += perplexity
            count += 1

    print("Average perplexity: {:.2f}".format(sum/count))


def main_2():
    ds = methods.get_dataset("jailbreak")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained GPT-2 model and tokenizer
    model_path = "/data/models/Llama-3-8B-Instruct"  # You can also use "gpt2-medium", "gpt2-large", "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    sum = 0
    count = 0
    for item in ds:
        text = item["Prompt"]
        # Calculate perplexity
        perplexity = calculate_perplexity(text, model, tokenizer, device)
        print(perplexity)
        sum += perplexity
        count += 1

    print("Average perplexity: {:.2f}".format(sum / count))

if __name__ == "__main__":
    # main()
    main_2()