from typing import Optional
from data_loader import PairedFunctionLoader, PairedFunction
from enums.approach import Approach
from enums.context_config import ContextConfig

class PromptBuilder:

    def __new__(cls, approach: Approach, context_config: Optional[ContextConfig]):

        # if calling the parent directly intercept and return the correct child
        routing_table = { 
            (Approach.context, ContextConfig.function_only): FunctionOnlyPromptBuilder
        }
        # Look up the class based on the combination
        target_class = routing_table.get((approach, context_config))
        
        if target_class is None:
            raise ValueError(f"Invalid class combination: {approach} with  {context_config}")

        # if child calls
        return super().__new__(target_class)

    def __init__(self, approach: Approach, context_config: Optional[ContextConfig], func: str):
        self.approach = approach
        self.context_config = context_config
        self.func = func
        self.prompt = ""


class FunctionOnlyPromptBuilder(PromptBuilder):
    def prompt(self):
        self.prompt = f"Classify this function as Vulnerable or Safe and identify CVE and CWE if applicable \n {self.func}"