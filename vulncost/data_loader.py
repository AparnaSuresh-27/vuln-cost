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


class PairedFunctionLoader():
    def __init__(self, idx, data_dir):
        self.data_dir = data_dir

    def get_samples(self):
        with open(self.data_dir) as file:
            for line in file: # one record per line 
                if line.strip():
                    function_record = json.loads(line)
                    
                    

