#!/bin/bash
# Collect performance-regression issue candidates from GitHub search API (unauthenticated).
# Search rate limit: 10/min. We sleep 7s between calls to stay under it.
set -u

OUTDIR="/Users/hanzhuojun/WorkSpace/InfraSearch/data/raw"
mkdir -p "$OUTDIR"

REPOS=(
  "vllm-project/vllm"
  "sgl-project/sglang"
  "ggml-org/llama.cpp"
)

QUERIES=(
  "performance regression"
  "perf regression"
  "throughput drop"
  "latency regression"
  "TTFT regression"
  "slower after upgrade"
  "performance degradation"
  "slower"
)

idx=0
for repo in "${REPOS[@]}"; do
  for q in "${QUERIES[@]}"; do
    idx=$((idx+1))
    fname=$(printf "%02d" "$idx")
    slug=$(echo "$repo" | tr '/' '_')
    qslug=$(echo "$q" | tr ' ' '_')
    outfile="$OUTDIR/${fname}_${slug}_${qslug}.json"

    if [ -f "$outfile" ] && [ -s "$outfile" ]; then
      echo "[$idx] skip existing $slug | $q"
      continue
    fi

    echo "[$idx] search $repo | $q"
    curl -sG -H "Accept: application/vnd.github+json" \
      --data-urlencode "q=repo:${repo} \"${q}\" in:title,body type:issue" \
      --data-urlencode "per_page=100" \
      "https://api.github.com/search/issues" \
      -o "$outfile" \
      -w "    remaining=%header{x-ratelimit-remaining}\n"

    sleep 7
  done
done
echo "DONE collection"
