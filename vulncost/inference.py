from mlx_lm import load, generate


class MLXInference:
    def __init__(self, model_id: str):
        # load once and reuse for every call
        self.model, self.tokenizer = load(model_id)

    def run(self, prompt: str):
        # wrap the raw prompt in Qwen's chat format
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # count input tokens (what to be sent to model)
        input_tokens = len(self.tokenizer.encode(formatted))

        # generate the model's response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=16, # one-word answer needs very few tokens
            verbose=False,
        )

        # count output tokens (what the model produced)
        output_tokens = len(self.tokenizer.encode(response))

        return response, input_tokens, output_tokens


def parse_prediction(response: str) -> str:
    text = response.upper()                      # normalize casing
    text = text.replace("<|IM_END|>", "")        # strip Qwen's end token
    text = text.strip()                          # strip whitespace/newlines

    has_vuln = "VULNERABLE" in text
    has_safe = "SAFE" in text

    if has_vuln and not has_safe:
        return "VULNERABLE"
    if has_safe and not has_vuln:
        return "SAFE"
    return "UNKNOWN"                             # unclear / both / neither