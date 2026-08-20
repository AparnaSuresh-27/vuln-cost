from enums.approach import Approach
from enums.context_config import ContextConfig

# holds the function and subclasses define how the prompt is built
class PromptBuilder:
    def __init__(self, func: str):
        self.func = func

    def build(self) -> str:
        raise NotImplementedError # subclasses must implement

# for function only scenario in the context approach
class FunctionOnlyPromptBuilder(PromptBuilder):
    def build(self) -> str:
        return (
            "Classify this function as VULNERABLE or SAFE. "
            "Respond with exactly one word: VULNERABLE or SAFE.\n\n"
            f"{self.func}"
        )


# the routing depending on approach and context
def make_prompt_builder(approach: Approach,
                        context_config: ContextConfig,
                        func: str) -> PromptBuilder:
    routing = {
        (Approach.context, ContextConfig.function_only): FunctionOnlyPromptBuilder,
    }
    builder_class = routing.get((approach, context_config))
    if builder_class is None:
        raise ValueError(f"No builder for {approach} / {context_config}")
    return builder_class(func)