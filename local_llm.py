import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLM:
    def __init__(self):
        self.model_cache = {}
        self.tokenizer_cache = {}
        self.lock = threading.Lock()

    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self.Completions(self)

        class Completions:
            def __init__(self, chat):
                self.chat = chat

            def create(self, model, messages, temperature, max_tokens):
                tokenizer, model_instance = self.chat.client.load_model(model)
                tokenizer.pad_token = tokenizer.eos_token

                prompt_tokens = self.chat.client.get_tokens(messages, tokenizer)

                encoded_inputs = tokenizer.encode_plus(
                    self.chat.client.format_messages(messages),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(prompt_tokens, tokenizer.model_max_length)
                ).to("cuda")

                input_ids = encoded_inputs["input_ids"]
                attention_mask = encoded_inputs["attention_mask"]

                if temperature == 0:
                    temperature = 1e-6

                output = model_instance.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_tokens,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

                response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                completion_tokens = len(output[0]) - len(input_ids[0])

                return Response(choices=[Choice(message=Message(content=response_text))],
                                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))

    @property
    def chat(self):
        return self.Chat(self)

    def get_tokens(self, messages, tokenizer):
        input_text = self.format_messages(messages)
        input_ids = tokenizer.encode(input_text)
        return len(input_ids)

    def load_model(self, model_name):
        with self.lock:
            if model_name in self.model_cache:
                model_instance = self.model_cache[model_name]
            else:
                model_path = "/data/models/" + model_name
                model_instance = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True).to("cuda")
                self.model_cache[model_name] = model_instance
            tokenizer = AutoTokenizer.from_pretrained("/data/models/" + model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model_instance.config.pad_token_id = tokenizer.pad_token_id
            print(tokenizer.pad_token_id)

            return tokenizer, model_instance

    def format_messages(self, messages):
        return "\n".join([msg['content'] for msg in messages])


class Response:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage


class Choice:
    def __init__(self, message):
        self.message = message


class Message:
    def __init__(self, content):
        self.content = content


class Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


if __name__ == "__main__":
    client = LocalLLM()

    def test_request(model, messages):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=128
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Prompt Tokens: {response.usage.prompt_token}")
        print(f"Completion Tokens: {response.usage.completion_tokens}")

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    model = "Llama-3.1-8B-Instruct"

    threads = []
    for _ in range(5):
        thread = threading.Thread(target=test_request, args=(model, messages))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
