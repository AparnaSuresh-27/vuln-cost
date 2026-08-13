from enum import Enum

class ContextConfig(Enum):
    function_only = "function_only"
    plus_caller_callee = "plus_caller_callee"
    plus_cpg_slice = "plus_cpg_slice"
    plus_nl_description = "plus_nl_description"
