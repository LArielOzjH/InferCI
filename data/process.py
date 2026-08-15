#!/usr/bin/env python3
"""Process raw GitHub search JSON files into regression_evidence.jsonl + summary."""
import json, re, glob, os, sys, html

RAW = "/Users/hanzhuojun/WorkSpace/InfraSearch/data/raw"
OUT = "/Users/hanzhuojun/WorkSpace/InfraSearch/data/regression_evidence.jsonl"
SUMMARY = "/Users/hanzhuojun/WorkSpace/InfraSearch/data/regression_evidence_summary.md"

# perf keywords that indicate a numeric claim is performance-related
PERF_KW = [
    "throughput", "tok/s", "token/s", "tokens/s", "tps", "latency",
    "ttft", "tpot", "itl", "ms", "millisecond", "second", "s slower",
    "slower", "dropped", "decrease", "decreased", "degrad", "regress",
    "fps", "req/s", "req/sec", "qps", "it/s", "speed", "slowdown",
    "drop", "token throughput", "ms/tok", "tokens per second",
]

NUM = r"\d"

def clean_body(body: str) -> str:
    if not body:
        return ""
    # remove <details> ... </details> blocks (env dumps)
    body = re.sub(r"<details>.*?</details>", " ", body, flags=re.DOTALL | re.IGNORECASE)
    # remove code fences
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    # inline code markers
    body = body.replace("`", "")
    # html tags
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    # template boilerplate
    body = re.sub(r"_No response_", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"###+", " ", body)
    # collapse whitespace
    body = re.sub(r"\s+", " ", body).strip()
    return body

def split_sentences(text: str):
    # split on newlines, periods, and common delimiters but keep reasonably long pieces
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

def score_candidate(s: str) -> int:
    low = s.lower()
    score = 0
    if not re.search(NUM, s):
        return -1
    # reject pure template junk
    for junk in ["no response", "your current environment", "describe the bug",
                 "proposal to improve", "misc discussion", "before submitting",
                 "collect_env", "check the box", "search for relevant issues"]:
        if junk in low:
            score -= 4
    for kw in PERF_KW:
        if kw in low:
            score += 3
    # strong signals
    if re.search(r"\bfrom\b", low) and re.search(r"\bto\b", low):
        score += 5
    if re.search(r"\d+(\.\d+)?\s*%", s):
        score += 3
    if re.search(r"\d+(\.\d+)?\s*x\s*(slower|faster)", low):
        score += 6
    for w in ["slower", "dropped", "degrad", "regress", "decreased", "down to", "reduced", "fell", "degradation", "downgrade"]:
        if w in low:
            score += 4
    # comparison / arrow operators
    for op in ["→", "->", "→", "vs", "compared", "before", "after", "was ", "now ", "recovered", "went"]:
        if op in low:
            score += 2
    # tok/s or ms/tps concrete units
    if re.search(r"(tok/s|token/s|tokens/s|tps|ms|latency|ttft|ms/tok)", low):
        score += 2
    return score

def extract_claim(body: str) -> str:
    if not body:
        return "N/A"
    cleaned = clean_body(body)
    best = ""
    best_score = 0
    for sent in split_sentences(cleaned):
        if len(sent) > 600:
            # try further splitting on commas/semicolons
            subs = re.split(r"[;,]", sent)
        else:
            subs = [sent]
        for sub in subs:
            sc = score_candidate(sub)
            if sc > best_score:
                best_score = sc
                best = sub.strip()
    if not best or best_score < 6:
        return "N/A"
    if len(best) > 220:
        best = best[:220].rsplit(" ", 1)[0] + "…"
    return best

# --- curation helpers ---
REGRESS_KW = re.compile(
    r"(slower|slowdown|dropped|\bdrop\b|degrad|regress|decreased|decreas|reduced|reduc|"
    r"loss|fell|declin|worse|down to|speed drop|throughput drop)", re.I)
NEGATION = re.compile(
    r"\bno\s+(measurable\s+|significant\s+|notable\s+|obvious\s+)?"
    r"(perf(ormance)?\s+)?(regression|slowdown|degradation|drop|slow|loss)\b", re.I)
EXCLUDE_TITLE = re.compile(
    r"^\s*\[?(feature|roadmap|tracking|track\b|doc|rfc|fr|query)[\]:\s-]", re.I)
EXCLUDE_TITLE_ANYWHERE = re.compile(r"\breadiness\b|how to\b|pr ready\b", re.I)
NUMERIC_UNITS = re.compile(
    r"(tok/s|token/s|tokens/s|tps|ttft|tpot|itl|\bms\b|req/s|qps|it/s|\bt/s\b|"
    r"tokens per second|generation speed|prompt processing|prefill|decode)")


def curation_score(r):
    c = (r["regression_claim"] or "").lower()
    t = (r["title"] or "").lower()
    s = 0
    if re.search(r"\bfrom\b.{0,60}\bto\b", c) and re.search(r"\d", c):
        s += 6
    if "->" in c or "→" in c:
        s += 5
    if re.search(r"\bvs\b|\bversus\b|\bcompared\b|\bbefore\b|\bafter\b", c):
        s += 3
    if re.search(r"\d+(\.\d+)?\s*%", c):
        s += 2
    if re.search(r"\d+(\.\d+)?\s*x\s*(slower|faster)", c):
        s += 4
    if NUMERIC_UNITS.search(c):
        s += 2
    if re.search(r"(regression|slower|drop|degrad|slowdown)", t):
        s += 3
    if re.search(r"(upgrade|since|between|after upgrad|version|0\.\d|b\d{4,5})", c):
        s += 2
    return s


def is_curated(r):
    if r["regression_claim"] == "N/A":
        return False
    t = r["title"] or ""
    c = r["regression_claim"] or ""
    if EXCLUDE_TITLE.match(t):
        return False
    if EXCLUDE_TITLE_ANYWHERE.search(t):
        return False
    if NEGATION.search(c) or NEGATION.search(t):
        return False
    if not (REGRESS_KW.search(c) or REGRESS_KW.search(t)):
        return False
    return True


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.json")))
    seen = {}  # (repo, number) -> record
    order = []
    for f in files:
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"WARN skip {f}: {e}", file=sys.stderr)
            continue
        items = data.get("items", [])
        if not items:
            continue
        # infer repo from first item's repository_url
        repo = ""
        for it in items:
            repo = it.get("repository_url", "").rsplit("/repos/", 1)[-1]
            break
        if not repo:
            repo = os.path.basename(f)
        for it in items:
            num = it.get("number")
            key = (repo, num)
            title = it.get("title") or ""
            # skip PR-only content markers? keep issues as-is
            labels = [l.get("name") for l in it.get("labels", []) or []]
            body = it.get("body") or ""
            if key in seen:
                # merge: keep existing; maybe update if this query's claim better? just skip
                continue
            rec = {
                "repo": repo,
                "number": num,
                "title": title,
                "state": it.get("state"),
                "created_at": it.get("created_at"),
                "comments": it.get("comments"),
                "labels": labels,
                "url": it.get("html_url"),
                "regression_claim": extract_claim(body),
                "body_snippet": clean_body(body)[:400],
            }
            seen[key] = rec
            order.append(key)

    # write FULL pool (all unique issues) for transparency
    FULL = "/Users/hanzhuojun/WorkSpace/InfraSearch/data/regression_evidence_full.jsonl"
    with open(FULL, "w") as fh:
        for key in order:
            fh.write(json.dumps(seen[key], ensure_ascii=False) + "\n")

    # curated selection: genuine regression reports with numeric claims
    curated = [seen[key] for key in order if is_curated(seen[key])]
    curated.sort(key=curation_score, reverse=True)

    # balance across repos: take top N per repo, then merge and sort by score
    per_repo_cap = 26
    picks = []
    repo_pools = {}
    for r in curated:
        repo_pools.setdefault(r["repo"], []).append(r)
    for repo, pool in repo_pools.items():
        picks.extend(pool[:per_repo_cap])
    picks.sort(key=curation_score, reverse=True)

    with open(OUT, "w") as fh:
        for r in picks:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # build summary from curated picks
    by_repo = {}
    with_nums = 0
    total = len(picks)
    for r in picks:
        by_repo.setdefault(r["repo"], []).append(r)
        if r["regression_claim"] != "N/A":
            with_nums += 1

    lines = []
    lines.append("# Performance Regression Evidence Summary\n")
    lines.append(f"- Curated regression reports: **{total}** (full raw pool: {len(order)} unique issues)")
    lines.append(f"- With concrete numeric claim: **{with_nums}**\n")
    lines.append("## Per-repo breakdown (curated)\n")
    for repo in sorted(by_repo):
        items = by_repo[repo]
        nums = sum(1 for r in items if r["regression_claim"] != "N/A")
        lines.append(f"- **{repo}**: {len(items)} issues ({nums} with numeric claim)")

    lines.append("\n## Representative 'slower after upgrade' cases\n")
    ranked = sorted(picks, key=curation_score, reverse=True)
    shown = 0
    for r in ranked:
        if shown >= 6:
            break
        lines.append(f"- **[{r['title']}]({r['url']})** ({r['repo']}#{r['number']}, {r['state']}, created {r['created_at'][:10]})")
        lines.append(f"  - claim: {r['regression_claim']}")
        shown += 1

    with open(SUMMARY, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"FULL_POOL={len(order)} CURATED={total} WITH_NUMS={with_nums}")
    for repo in sorted(by_repo):
        items = by_repo[repo]
        nums = sum(1 for r in items if r["regression_claim"] != "N/A")
        print(f"  {repo}: {len(items)} issues ({nums} with nums)")

if __name__ == "__main__":
    main()
