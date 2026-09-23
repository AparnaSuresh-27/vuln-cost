"""
GPU inference for the RACE box (unsloth + 4-bit Qwen2.5-Coder-7B, optional LoRA).
Mirrors MLXInference.run() but also returns a status, so an over-length or OOM
sample is logged as failed instead of crashing an overnight run.

Kept separate from inference.py because that module imports mlx_lm, which does
not exist on Linux.
"""
import torch

MAX_NEW_TOKENS = 8  # the answer is one word; matches eval_sanity.py


class HFInference:
    def __init__(self, model_path: str, max_seq_length: int = 16384):
        from unsloth import FastLanguageModel
        # model_path = adapter dir (reads its base model from adapter_config.json)
        # or a base model id for the untrained control
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_path, max_seq_length=max_seq_length, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = max_seq_length

    def format(self, prompt: str) -> str:
        # same chat wrapping as training (finetune_vd.py) and the MLX runs
        msgs = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

    def run(self, prompt: str):
        """Returns (response, input_tokens, output_tokens, status).
        input_tokens is always the FULL prompt length, even when skipped, so the
        cost of the rung is recorded whether or not the model could read it."""
        enc = self.tokenizer(self.format(prompt), return_tensors="pt",
                             add_special_tokens=False)
        input_ids = enc["input_ids"]
        n_in = int(input_ids.shape[1])

        if n_in + MAX_NEW_TOKENS > self.max_seq_length:
            return "", n_in, 0, "too_long"

        input_ids = input_ids.to(self.model.device)
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # greedy = deterministic
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "", n_in, 0, "oom"

        gen = out[0][n_in:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text, n_in, int(gen.shape[0]), "ok"


def parse_prediction(response: str) -> str:
    # identical rule to vulncost.inference.parse_prediction (copied, since that
    # module can't be imported on Linux): one class word or UNKNOWN
    text = response.upper().replace("<|IM_END|>", "").strip()
    has_vuln = "VULNERABLE" in text
    has_safe = "SAFE" in text
    if has_vuln and not has_safe:
        return "VULNERABLE"
    if has_safe and not has_vuln:
        return "SAFE"
    return "UNKNOWN"