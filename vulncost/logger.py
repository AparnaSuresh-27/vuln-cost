from dataclasses import asdict, dataclass
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional

 

ALLOWED_STATUS = {"ok", "failed"} # whether the interaction with model was success or not
                                  # will add more status if needed

# Structure of the classes
# Run 
#   Sample
#     Repeat
#       Call Number 

@dataclass
class CallRecord: # things that change per interaction with a model
    model: str
    approach: Enum  # if it is context or llm vs agent comparison
    context_config: Optional[Enum] # configuration of context
    agent: Optional[Enum] # agent infrastructure
    sample_id: str # idx from PrimeVul
    repeat: int # skip now
    call_number: int
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    prediction: str
    label: int
    status: str
    duration_s: float
    temperature: float
    seed: int

    def __post_init__(self):
        if self.status not in ALLOWED_STATUS:
            raise ValueError(
                f"status must be one of {ALLOWED_STATUS}, got {self.status!r}"
            )
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not self.model:
            raise ValueError("model is required")


class RunLogger:  # a whole run that goes through the data for a set of configurations

    def __init__(self, base_dir, label):  # label is name assigned for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{label}"

        run_dir = Path(base_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.path = run_dir / "calls.jsonl"
        self.file = open(self.path, "x")

    def log(self, record):
            row = asdict(record)
            row["run_id"] = self.run_id
            row["timestamp"] = datetime.now().isoformat()

            for key, value in row.items():
                if isinstance(value, Enum):
                    row[key] = value.value

            self.file.write(json.dumps(row) + "\n")
            self.file.flush()

    def close(self):
        self.file.close()