import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from vulncost.data_loader import PairedFunction

from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"


@dataclass
class FetchedSource:  # what a successful fetch returns
    idx: str
    owner: str
    repo: str
    fetched_commit: str  # the commit actually pulled from (parent or fix)
    file_path: str # full in-repo path, e.g. hphp/runtime/base/preg.cpp
    source: str # the file's full text


def _owner_repo(project_url: str) -> tuple[str, str]:
    # "https://github.com/facebook/hhvm" -> ("facebook", "hhvm")
    parts = urlparse(project_url).path.strip("/").split("/")
    return parts[0], parts[1]


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")  # never hardcoded
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_source_for_record(record: PairedFunction) -> Optional[FetchedSource]:
    owner, repo = _owner_repo(record.project_url)

    # 1. one API call on the fix commit -> parent SHA + full file path
    resp = httpx.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/commits/{record.commit_id}",
        headers=_headers(), timeout=30,
    )
    if resp.status_code != 200:
        print(f"[skip {record.idx}] commit API returned {resp.status_code}")
        return None
    data = resp.json()

    parents = data.get("parents", [])
    if not parents:
        print(f"[skip {record.idx}] no parent commit")
        return None
    parent_sha = parents[0]["sha"]

    # full path = the changed file whose basename matches file_name
    full_path = None
    for f in data.get("files", []):
        if f["filename"].split("/")[-1] == record.file_name:
            full_path = f["filename"]
            break
    if full_path is None:
        print(f"[skip {record.idx}] {record.file_name} not among changed files")
        return None

    # 2. vulnerable (target==1) -> parent commit; patched (target==0) -> fix commit
    fetched_commit = parent_sha if record.target == 1 else record.commit_id

    # 3. fetch just that one raw file (raw host is not rate-limited)
    raw = httpx.get(
        f"{GITHUB_RAW}/{owner}/{repo}/{fetched_commit}/{full_path}",
        headers=_headers(), timeout=30,
    )
    if raw.status_code != 200:
        print(f"[skip {record.idx}] raw fetch returned {raw.status_code}")
        return None

    return FetchedSource(
        idx=record.idx,
        owner=owner,
        repo=repo,
        fetched_commit=fetched_commit,
        file_path=full_path,
        source=raw.text,
    )


# run this file directly to smoke-test on the first few records
if __name__ == "__main__":
    from vulncost.data_loader import PairedFunctionLoader

    loader = PairedFunctionLoader("data/primevul_valid_paired.jsonl")
    for i, record in enumerate(loader.get_samples()):
        if i >= 3:  # just the first 3 records
            break
        result = fetch_source_for_record(record)
        if result:
            print(f"[ok {result.idx}] {result.file_path} @ "
                  f"{result.fetched_commit[:10]} — {len(result.source)} chars "
                  f"(target={record.target})")