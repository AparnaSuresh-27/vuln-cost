from vulncost.data_loader import PairedFunctionLoader
from vulncost.prompt_builder import make_prompt_builder
from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig

loader = PairedFunctionLoader("data/primevul_valid_paired.jsonl")
sample = next(loader.get_samples()) # grab one record
builder = make_prompt_builder(Approach.context, ContextConfig.function_only, sample.func)
print(builder.build())