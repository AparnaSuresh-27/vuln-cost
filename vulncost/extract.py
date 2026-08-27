import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

EXTRACT_DIR = Path("data/extracted")


@dataclass
class CallerCalleeResult:  # extraction output for one function in one file
    func_name: str
    callees: list[str]
    callers: list[str]
    callee_bodies: dict
    caller_bodies: dict
    # resolution stats (computed automatically for feasibility reporting)
    n_callees: int = 0
    n_callees_resolved: int = 0
    n_callers: int = 0
    n_callers_resolved: int = 0
    unresolved_callees: list = None
    unresolved_callers: list = None

    def __post_init__(self):
        def resolved(names, bodies):
            return [n for n in names
                    if bodies.get(n, "<empty>") not in ("<empty>", "", None)]
        rc = resolved(self.callees, self.callee_bodies)
        rr = resolved(self.callers, self.caller_bodies)
        self.n_callees = len(self.callees)
        self.n_callees_resolved = len(rc)
        self.n_callers = len(self.callers)
        self.n_callers_resolved = len(rr)
        self.unresolved_callees = [n for n in self.callees if n not in rc]
        self.unresolved_callers = [n for n in self.callers if n not in rr]


def _joern_binary() -> str:
    home = os.environ.get("JOERN_HOME")
    if not home:
        raise RuntimeError("JOERN_HOME not set (add it to .env)")
    return os.path.join(home, "joern")


_JOERN_SCRIPT = '''
importCode(inputPath="{src}", projectName="extract_tmp")

val target = "{fn}"

val callees = cpg.method.name(target).call.name
  .filterNot(_.startsWith("<operator>"))
  .dedup.l

val callers = cpg.method.name(target).caller.name.dedup.l

val calleeBodies = callees.flatMap {{ c =>
  cpg.method.name(c).code.headOption.map(code => (c, code))
}}.toMap

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
            print(proc.stderr[-500:])
            return None

        with open(out_path) as f:
            raw = json.load(f)

    data = {}
    if isinstance(raw, list):
        for item in raw:
            data.update(item)
    else:
        data = raw

    return CallerCalleeResult(
        func_name=data.get("func_name", func_name),
        callees=data.get("callees", []),
        callers=data.get("callers", []),
        callee_bodies=data.get("callee_bodies", {}),
        caller_bodies=data.get("caller_bodies", {}),
    )


def _func_name_from_code(func: str) -> str:
    before_paren = func.split("(")[0]
    return before_paren.strip().split()[-1].lstrip("*")  # strip pointer '*'


def extract_and_cache(record) -> Optional[CallerCalleeResult]:
    from vulncost.fetch import fetch_source_for_record

    out_path = EXTRACT_DIR / f"{record.idx}.json"
    if out_path.exists():
        print(f"[cached {record.idx}] already extracted")
        with open(out_path) as f:
            data = json.load(f)
        core = {k: data[k] for k in ("func_name", "callees", "callers",
                                     "callee_bodies", "caller_bodies")}
        return CallerCalleeResult(**core)

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
          f"{result.n_callees} callees ({result.n_callees_resolved} w/ body), "
          f"{result.n_callers} callers ({result.n_callers_resolved} w/ body) "
          f"-> {out_path}")
    return result


if __name__ == "__main__":
    from vulncost.data_loader import PairedFunctionLoader

    loader = PairedFunctionLoader("data/primevul_valid_paired.jsonl")
    N = 200

    ok = fail = 0
    tot_callees = tot_callees_res = 0
    tot_callers = tot_callers_res = 0
    samples_with_callers = 0
    samples_with_resolved_callee = 0

    for i, record in enumerate(loader.get_samples()):
        if i >= N:
            break
        try:
            result = extract_and_cache(record)
        except Exception as e:
            print(f"[error {record.idx}] {e}")
            result = None

        if not result:
            fail += 1
            continue
        ok += 1
        tot_callees += result.n_callees
        tot_callees_res += result.n_callees_resolved
        tot_callers += result.n_callers
        tot_callers_res += result.n_callers_resolved
        if result.n_callers:
            samples_with_callers += 1
        if result.n_callees_resolved:
            samples_with_resolved_callee += 1

    print("\n" + "=" * 55)
    print("FEASIBILITY SUMMARY (intra-file caller/callee)")
    print("=" * 55)
    print(f"Samples attempted:            {ok + fail}")
    print(f"  extracted ok:               {ok}")
    print(f"  failed (fetch/parse):       {fail}")
    print(f"Samples with >=1 caller:      {samples_with_callers}/{ok}")
    print(f"Samples with >=1 resolved callee body: {samples_with_resolved_callee}/{ok}")
    print(f"\nCallees: {tot_callees} total, {tot_callees_res} with bodies "
          f"({100*tot_callees_res/max(tot_callees,1):.0f}% resolved)")
    print(f"Callers: {tot_callers} total, {tot_callers_res} with bodies "
          f"({100*tot_callers_res/max(tot_callers,1):.0f}% resolved)")
    print(f"Avg callees/sample: {tot_callees/max(ok,1):.1f} "
          f"({tot_callees_res/max(ok,1):.1f} with bodies)")
    print(f"Avg callers/sample: {tot_callers/max(ok,1):.1f} "
          f"({tot_callers_res/max(ok,1):.1f} with bodies)")