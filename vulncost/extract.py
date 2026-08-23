import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class CallerCalleeResult: # extraction output for one function in one file
    func_name: str
    callees: list[str] # functions this target calls (same-file resolvable)
    callers: list[str] # functions that call the target (same-file only)
    callee_bodies: dict # name -> source text, for callees defined in the file
    caller_bodies: dict # name -> source text, for callers defined in the file


def _joern_binary() -> str:
    home = os.environ.get("JOERN_HOME")
    if not home:
        raise RuntimeError("JOERN_HOME not set (add it to .env)")
    return os.path.join(home, "joern")


# Joern script template: build CPG on one file, emit callers/callees as JSON.
# {src} = path to the source file, {fn} = target function name.
_JOERN_SCRIPT = '''
importCode(inputPath="{src}", projectName="extract_tmp")

val target = "{fn}"

// callees: named functions the target calls, excluding <operator>.* builtins
val callees = cpg.method.name(target).call.name
  .filterNot(_.startsWith("<operator>"))
  .dedup.l

// callers: methods that call the target
val callers = cpg.method.name(target).caller.name.dedup.l

// bodies of callees that are actually DEFINED in this file
val calleeBodies = callees.flatMap {{ c =>
  cpg.method.name(c).code.headOption.map(code => (c, code))
}}.toMap

// bodies of callers defined in this file
val callerBodies = callers.flatMap {{ c =>
  cpg.method.name(c).code.headOption.map(code => (c, code))
}}.toMap

val result = Map(
  "func_name"     -> target,
  "callees"       -> callees,
  "callers"       -> callers,
  "callee_bodies" -> calleeBodies,
  "caller_bodies" -> callerBodies,
)

import java.io.PrintWriter
new PrintWriter("{out}") {{ write(result.toJson); close() }}
'''


def extract_caller_callee(source: str, func_name: str) -> Optional[CallerCalleeResult]:
    # Joern needs the source on disk; use a temp dir that cleans itself up
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "target.cpp")
        script_path = os.path.join(tmp, "extract.sc")
        out_path = os.path.join(tmp, "result.json")

        with open(src_path, "w") as f:
            f.write(source)

        script = _JOERN_SCRIPT.format(src=src_path, fn=func_name, out=out_path)
        with open(script_path, "w") as f:
            f.write(script)

        proc = subprocess.run(
            [_joern_binary(), "--script", script_path],
            capture_output=True, text=True, timeout=300,
        )

        if not os.path.exists(out_path):
            print(f"[extract fail {func_name}] Joern produced no output")
            print(proc.stderr[-500:])  # tail of the error, for debugging
            return None

        with open(out_path) as f:
            raw = json.load(f)

    # Joern's toJson emits a list of single-key objects; merge into one dict
    data = {}
    if isinstance(raw, list):
        for item in raw:
            data.update(item)
    else:
        data = raw  # already a dict (in case behaviour differs)

    return CallerCalleeResult(
        func_name=data.get("func_name", func_name),
        callees=data.get("callees", []),
        callers=data.get("callers", []),
        callee_bodies=data.get("callee_bodies", {}),
        caller_bodies=data.get("caller_bodies", {}),
    )


# smoke test: fetch preg.cpp, extract preg_quote — we know the expected answer
from pathlib import Path

EXTRACT_DIR = Path("data/extracted")


def _func_name_from_code(func: str) -> str:
    # parse the function name out of the PrimeVul func text.
    # e.g. "String preg_quote(const String& str, ...)" -> "preg_quote"
    before_paren = func.split("(")[0]
    return before_paren.strip().split()[-1]


def extract_and_cache(record) -> Optional[CallerCalleeResult]:
    from vulncost.fetch import fetch_source_for_record

    out_path = EXTRACT_DIR / f"{record.idx}.json"
    if out_path.exists():                       # already done -> skip (resumable)
        print(f"[cached {record.idx}] already extracted")
        with open(out_path) as f:
            data = json.load(f)
        return CallerCalleeResult(**data)

    fetched = fetch_source_for_record(record)
    if not fetched:
        print(f"[skip {record.idx}] fetch failed")
        return None

    func_name = _func_name_from_code(record.func)
    result = extract_caller_callee(fetched.source, func_name)
    if not result:
        print(f"[skip {record.idx}] extraction failed")
        return None

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"[ok {record.idx}] {func_name}: "
          f"{len(result.callees)} callees, {len(result.callers)} callers "
          f"-> {out_path}")
    return result


if __name__ == "__main__":
    from vulncost.data_loader import PairedFunctionLoader

    loader = PairedFunctionLoader("data/primevul_valid_paired.jsonl")
    for i, record in enumerate(loader.get_samples()):
        if i >= 3:                              # first 3 records only for now
            break
        extract_and_cache(record)