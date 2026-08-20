from dataclasses import dataclass, fields
import json
from typing import Dict, Iterator



@dataclass
class PairedFunction: # for structuring each record
    idx: str
    project: str
    commit_id: str
    project_url: str
    commit_url: str
    commit_message: str
    target: int
    func: str

    @classmethod
    def from_dict(cls, record: Dict) -> "PairedFunction":
        allowed = {f.name for f in fields(cls)} # declared field names
        filtered = {k: v for k, v in record.items() if k in allowed}
        filtered["idx"] = str(filtered["idx"]) # CallRecord.sample_id is str
        return cls(**filtered)


class PairedFunctionLoader:
    def __init__(self, idx, data_dir):
        self.data_dir = data_dir

    def get_samples(self) -> Iterator[PairedFunction]:
        with open(self.data_dir) as file:
            for line in file: # one record per line
                if line.strip():
                    record = json.loads(line)
                    yield PairedFunction.from_dict(record)
                    
                    

