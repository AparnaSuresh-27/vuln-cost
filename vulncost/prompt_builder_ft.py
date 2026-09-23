"""
Prompts for the fine-tuned detector.

The template is imported from finetune_vd.py rather than copied, so the ladder
can never drift from what the adapter was trained on. Every rung uses it
verbatim; the rungs differ only in what is placed in {code}.

    function_only       {code} = the target function, exactly as in training
    plus_caller_callee  {code} = the target function, then its resolved callees
                        and callers, laid out as C with // section comments

The old prompt_builder.py (generic "Classify this function..." prompt) is left
untouched so the earlier untrained-baseline runs stay reproducible.
"""
from finetune_vd import PROMPT as FT_TEMPLATE
from vulncost.enums.context_config import ContextConfig

NO_CONTEXT_MARKER = "// ---- Context: none resolved for this function ----"


def _resolved(names, bodies):
    return [(n, bodies[n]) for n in sorted(names)
            if bodies.get(n) not in (None, "", "<empty>")]


def render_caller_callee(func: str, extracted: dict) -> str:
    callees = _resolved(extracted.get("callees", []), extracted.get("callee_bodies", {}))
    callers = _resolved(extracted.get("callers", []), extracted.get("caller_bodies", {}))

    # target first, so the function being judged is always read before context
    parts = ["// Target function", func.rstrip()]
    if callees:
        parts.append("\n// ---- Context: functions called by the target function (callees) ----")
        parts += [f"// {name}\n{body.rstrip()}" for name, body in callees]
    if callers:
        parts.append("\n// ---- Context: functions that call the target function (callers) ----")
        parts += [f"// {name}\n{body.rstrip()}" for name, body in callers]
    if not callees and not callers:
        parts.append("\n" + NO_CONTEXT_MARKER)
    return "\n".join(parts)


def build_ft_prompt(context_config: ContextConfig, func: str, extracted: dict = None) -> str:
    if context_config == ContextConfig.function_only:
        return FT_TEMPLATE.format(code=func)
    if context_config == ContextConfig.plus_caller_callee:
        if extracted is None:
            raise ValueError("plus_caller_callee requires extracted context")
        return FT_TEMPLATE.format(code=render_caller_callee(func, extracted))
    raise ValueError(f"No fine-tuned builder for {context_config}")