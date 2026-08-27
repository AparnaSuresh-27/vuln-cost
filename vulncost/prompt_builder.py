from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig

# holds the function and subclasses define how the prompt is built
class PromptBuilder:
    def __init__(self, func: str):
        self.func = func

    def build(self) -> str:
        raise NotImplementedError  # subclasses must implement

# for function only scenario in the context approach
class FunctionOnlyPromptBuilder(PromptBuilder):
    def build(self) -> str:
        return (
            "Classify this function as VULNERABLE or SAFE. "
            "Respond with exactly one word: VULNERABLE or SAFE.\n\n"
            f"{self.func}"
        )

# function + caller/callee context (resolved bodies only, for a fair comparison)
class CallerCalleePromptBuilder(PromptBuilder):
    def __init__(self, func: str, extracted: dict):
        super().__init__(func)
        self.extracted = extracted  # cached data/extracted/{idx}.json content

    def _resolved(self, names: list, bodies: dict) -> list:
        # keep only entries where a real body was captured (drops library/cross-file)
        return [(n, bodies[n]) for n in names
                if bodies.get(n, "<empty>") not in ("<empty>", "", None)]

    def _render_context(self) -> str:
        callees = self._resolved(
            self.extracted.get("callees", []),
            self.extracted.get("callee_bodies", {}),
        )
        callers = self._resolved(
            self.extracted.get("callers", []),
            self.extracted.get("caller_bodies", {}),
        )

        parts = []
        if callees:
            parts.append("Functions this function calls (callees):")
            for name, body in callees:
                parts.append(f"\n// {name}\n{body}")
        if callers:
            parts.append("\nFunctions that call this function (callers):")
            for name, body in callers:
                parts.append(f"\n// {name}\n{body}")

        return "\n".join(parts) if parts else "(no resolved caller/callee context)"

    def build(self) -> str:
        return (
            "Classify this function as VULNERABLE or SAFE. "
            "Respond with exactly one word: VULNERABLE or SAFE.\n\n"
            "Target function:\n"
            f"{self.func}\n\n"
            "--- Surrounding context ---\n"
            f"{self._render_context()}"
        )

# the routing depending on approach and context
def make_prompt_builder(approach: Approach,
                        context_config: ContextConfig,
                        func: str,
                        extracted: dict = None) -> PromptBuilder:
    # routing = {
    #     (Approach.context, ContextConfig.function_only): FunctionOnlyPromptBuilder,
    #     (Approach.context, ContextConfig.plus_caller_callee) : CallerCalleePromptBuilder
    # }
    # builder_class = routing.get((approach, context_config))
    # if builder_class is None:
    #     raise ValueError(f"No builder for {approach} / {context_config}")
    # return builder_class(func)
    if (approach, context_config) == (Approach.context, ContextConfig.function_only):
        return FunctionOnlyPromptBuilder(func)
    if (approach, context_config) == (Approach.context, ContextConfig.plus_caller_callee):
        if extracted is None:
            raise ValueError("plus_caller_callee requires extracted context")
        return CallerCalleePromptBuilder(func, extracted)
    raise ValueError(f"No builder for {approach} / {context_config}")