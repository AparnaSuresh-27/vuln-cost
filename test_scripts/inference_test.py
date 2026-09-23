# from vulncost.inference import MLXInference

# engine = MLXInference("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
# response, in_tok, out_tok = engine.run("Reply with exactly the word: ok")
# print("response:", repr(response))
# print("input tokens:", in_tok, "| output tokens:", out_tok)

from vulncost.inference import parse_prediction   # or wherever you put it
print(parse_prediction('Ok<|im_end|>\n'))         # -> UNKNOWN (model said "Ok", not SAFE/VULN)
print(parse_prediction('VULNERABLE<|im_end|>\n')) # -> VULNERABLE
print(parse_prediction('The function is SAFE.'))  # -> SAFE