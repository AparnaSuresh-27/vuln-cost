from vulncost.logger import CallRecord, RunLogger
from vulncost.enums.approach import Approach
from vulncost.enums.context_config import ContextConfig
from vulncost.enums.agents import Agent
import json

logger = RunLogger(base_dir="logs", label="scratch")

r1 = CallRecord(
    model="qwen2.5-coder-32b",
    approach=Approach.context,
    context_config=ContextConfig.function_only,
    agent=None,
    sample_id="89023",
    repeat=1,
    call_number=1,
    prompt="fake prompt",
    response="VULNERABLE",
    input_tokens=500,
    output_tokens=3,
    prediction="VULNERABLE",
    label=1,
    status="ok",
    duration_s=1.2,
    temperature=0.0,
    seed=42,
)

r2 = CallRecord(
    model="qwen2.5-coder-32b",
    approach=Approach.architecture,
    context_config=None,
    agent=Agent.mavul,
    sample_id="89023",
    repeat=1,
    call_number=1,
    prompt="fake agent prompt turn 1",
    response="I need to see the callee",
    input_tokens=600,
    output_tokens=20,
    prediction=None,
    label=1,
    status="ok",
    duration_s=2.1,
    temperature=0.0,
    seed=42,
)

r3 = CallRecord(
    model="qwen2.5-coder-32b",
    approach=Approach.architecture,
    context_config=None,
    agent=Agent.mavul,
    sample_id="89023",
    repeat=1,
    call_number=2,
    prompt="fake agent prompt turn 2",
    response="SAFE",
    input_tokens=1400,
    output_tokens=4,
    prediction="SAFE",
    label=1,
    status="banana",
    duration_s=1.8,
    temperature=0.0,
    seed=42,
)

logger.log(r1)
logger.log(r2)
logger.log(r3)
logger.close()

print(f"wrote to {logger.path}\n")
with open(logger.path) as f:
    for line in f:
        print(json.loads(line))