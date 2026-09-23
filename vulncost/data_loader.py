from dataclasses import dataclass, fields
import json
import re
from typing import Dict, Iterator, Optional



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
    func_hash: str
    file_name: int
    file_hash: int
    cwe: Optional[list[str]]
    cve: Optional[str]
    cve_desc: Optional[str]
    nvd_url: Optional[str]


    @classmethod
    def from_dict(cls, record: Dict) -> "PairedFunction":
        allowed = {f.name for f in fields(cls)} # declared field names
        filtered = {k: v for k, v in record.items() if k in allowed}
        filtered["idx"] = str(filtered["idx"]) # CallRecord.sample_id is str
        return cls(**filtered)


class PairedFunctionLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_samples(self) -> Iterator[PairedFunction]:
        with open(self.data_dir) as file:
            for line in file: # one record per line
                if line.strip():
                    record = json.loads(line)
                    yield PairedFunction.from_dict(record)



class JitVulLoader:
    """
    JITVul loader. One record on disk = one vul/benign pair with
    pre-extracted caller/callee context. Yields (PairedFunction, extracted)
    tuples — vulnerable first, benign second, so pair-adjacency is preserved
    for pairwise metrics. `extracted` is shaped like data/extracted/{idx}.json
    so CallerCalleePromptBuilder works unchanged.
    """
    # 875/879 non-vulnerable bodies start with a stray language tag ("c\n" or
    # "cpp\n", left over from stripped markdown fences); vulnerable bodies never
    # do. Left in, the first token alone separates the two halves of a pair.
    _LANG_TAG = re.compile(r"^(?:cpp|c)\n")

    def __init__(self, data_path, clean=False, dedupe=False):
        self.data_path = data_path
        self.clean = clean    # strip the language-tag artifact from function bodies
        self.dedupe = dedupe  # keep the first record per idx (79 records repeat an idx)

    @staticmethod
    def _neighbours(graph):
        # graph shape: {target_func_name: {inner_key: [neighbour_names]}}
        # trust the outer graph name (caller_graph vs callee_graph); ignore inner key
        names = set()
        if isinstance(graph, dict):
            for _target, inner in graph.items():
                if isinstance(inner, dict):
                    for _k, v in inner.items():
                        if isinstance(v, list):
                            names.update(n for n in v if isinstance(n, str))
        return names

    @classmethod
    def _extracted_from(cls, record, prefix):
        caller_graph = record.get(f"{prefix}_caller_graph", {}) or {}
        callee_graph = record.get(f"{prefix}_callee_graph", {}) or {}
        bodies = record.get(f"{prefix}_function_bodies", {}) or {}

        caller_names = cls._neighbours(caller_graph)
        callee_names = cls._neighbours(callee_graph)

        # drop the target functions themselves (outer keys of the graphs)
        targets = set(caller_graph.keys()) | set(callee_graph.keys())
        caller_names -= targets
        callee_names -= targets

        # keep only names with real bodies; sorted so prompts are identical across
        # runs (set order depends on PYTHONHASHSEED, which changes every process)
        caller_names = sorted(n for n in caller_names if bodies.get(n))
        callee_names = sorted(n for n in callee_names if bodies.get(n))

        return {
            "callers": caller_names,
            "callees": callee_names,
            "caller_bodies": {n: bodies[n] for n in caller_names},
            "callee_bodies": {n: bodies[n] for n in callee_names},
        }

    @staticmethod
    def _make_pf(record, target: int, func: str) -> PairedFunction:
        fixing = record.get("vulnerability_fixing_commits") or []
        first_commit = fixing[0] if fixing else ""
        project_url = record.get("project_url", "") or ""
        commit_url = f"{project_url}/commit/{first_commit}" if (project_url and first_commit) else ""
        return PairedFunction(
            idx=str(record.get("idx", "")),
            project=record.get("project", "") or "",
            commit_id=first_commit,
            project_url=project_url,
            commit_url=commit_url,
            commit_message="",
            target=target,
            func=func,
            func_hash=record.get("func_hash", "") or "",
            file_name=record.get("file_name", "") or "",
            file_hash=record.get("file_hash", "") or "",
            cwe=record.get("cwe"),
            cve=record.get("cve"),
            cve_desc=record.get("cve_desc"),
            nvd_url=record.get("nvd_url"),
        )

    def _body(self, record, key):
        body = record.get(key, "") or ""
        return self._LANG_TAG.sub("", body) if self.clean else body

    def get_samples(self):
        seen = set()
        with open(self.data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if self.dedupe:
                    if r.get("idx") in seen:
                        continue
                    seen.add(r.get("idx"))
                yield (self._make_pf(r, 1, self._body(r, "vulnerable_function_body")),
                       self._extracted_from(r, "vulnerable"))
                yield (self._make_pf(r, 0, self._body(r, "non_vulnerable_function_body")),
                       self._extracted_from(r, "non_vulnerable"))