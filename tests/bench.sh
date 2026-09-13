#!/bin/sh
# Scoreboard: llama.cpp (Metal) vs our engine, same prompt/generation sizes. Run with nothing else on the GPU.
cd "$(dirname "$0")/.." || exit 1
P=${P:-128}; N=${N:-32}; R=${R:-3}
for q in Q4_0 Q8_0 F16; do
    m=models/Llama-3.2-3B-Instruct-$q.gguf
    llama-bench -m $m -p $P -n $N -r $R 2>/dev/null | grep -E "pp$P|tg$N" | awk -v q=$q '{printf "%-36s llama.cpp %s %s ± %s tok/s\n", q, $(NF-5), $(NF-3), $(NF-1)}'
    ./llm bench $m -p $P -n $N -r $R
done
