from vulncost.data_loader import PairedFunctionLoader
from vulncost.prompt_builder import make_prompt_builder
from vulncost.inference import MLXInference, parse_prediction
from vulncost.logger import RunLogger, CallRecord
from vulncost.enums.approach import Approach
from vulncost.enums.agents import Agent
from vulncost.enums.context_config import ContextConfig
import time

MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" #small model for my gpu
DATA_PATH = "data/primevul_valid_paired.jsonl"
SUBSET = 20 # number of records (kept even to keep pairs stay whole)

def label_to_pred_str(target: int) -> str:
    # ground-truth label as a string to compare against the parsed prediction
    return "VULNERABLE" if target == 1 else "SAFE"

def main():
    loader = PairedFunctionLoader(DATA_PATH)
    engine = MLXInference(MODEL_ID)
    logger = RunLogger("logs", "control_function_only")

    results = [] # collect (sample_id, target, prediction) for P-C

    for i, sample in enumerate(loader.get_samples()):
        if i >= SUBSET:
            break

        prompt = make_prompt_builder(
            Approach.context, ContextConfig.function_only, sample.func
        ).build()

        start = time.time()
        response, in_tok, out_tok = engine.run(prompt)
        duration = time.time() - start

        prediction = parse_prediction(response)

        record = CallRecord(
            model=MODEL_ID,
            approach=Approach.context,
            context_config=ContextConfig.function_only,
            agent= None,                 
            sample_id=sample.idx,
            repeat=0,
            call_number=1,
            prompt=prompt,
            response=response,
            input_tokens=in_tok,
            output_tokens=out_tok,
            prediction=prediction,
            label=sample.target,
            status="ok",
            duration_s=duration,
            temperature=0.0,
            seed=0,
        )
        logger.log(record)
        results.append((sample.idx, sample.target, prediction))
        print(f"[{i}] {sample.idx}: pred={prediction} truth={label_to_pred_str(sample.target)}")

    logger.close()

    p_c_count = 0
    for i, (idx, target, prediction) in enumerate(results):
        if i % 2 == 0 and i + 1 < len(results):
            (_, pair_target, pair_prediction) = results[i + 1]
            if prediction == label_to_pred_str(target):
                if pair_prediction == label_to_pred_str(pair_target):
                    p_c_count += 1

    num_pairs = len(results) // 2
    p_c = p_c_count / num_pairs
    print(f"P-C: {p_c:.3f}  ({p_c_count}/{num_pairs} pairs both correct)")

if __name__ == "__main__":
    main()