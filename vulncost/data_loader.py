from dataclasses import dataclass
import json
import os



@dataclass
class PairedFunction: # for structuring each record
    idx: str
    project: str
    commit_id: str
    project_url: str
    commit_url: str
    commit_message: str
    target: bool
    func: str

        