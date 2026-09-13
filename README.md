# llm — an LLM inference engine from scratch for Apple Silicon

No PyTorch, no MLX, no llama.cpp. Loads a GGUF model file, tokenizes, runs the
transformer, and generates text. CPU reference path first, then Metal GPU kernels
compiled at runtime (no Xcode needed, Command Line Tools are enough).

Target model: Llama 3.2 3B Instruct (Q4_0 for speed, Q8_0/F16 for exact-match testing).
Reference for correctness and speed: llama.cpp (`brew install llama.cpp`).

## Layout

    src/gguf.*        GGUF file parser + mmap of tensor data
    src/tokenizer.*   Llama 3 byte-level BPE tokenizer
    src/model.*       model config + weight table
    src/cpu_*.cpp     CPU reference forward pass
    src/metal_*.mm    Metal backend (Objective-C++)
    src/sampler.*     temperature / top-k / top-p / repetition penalty
    src/chat.*        Llama 3 chat template as token sequences
    src/server.*      HTTP server, OpenAI chat-completions API (streaming via SSE)
    src/json.*        tiny JSON parser/serializer
    src/engine.*      shared glue: backend construction, prefill, UTF-8 streaming
    shaders/*.metal   GPU kernels, compiled at runtime
    tests/            comparison scripts against llama.cpp
    models/           weights (gitignored)

## Usage

    make
    ./llm info models/Llama-3.2-3B-Instruct-Q4_0.gguf
    ./llm run  models/Llama-3.2-3B-Instruct-Q4_0.gguf -p "The capital of France is" -n 32
    ./llm run  models/Llama-3.2-3B-Instruct-Q4_0.gguf --chat -p "Explain RoPE in one paragraph." -n 200
    ./llm run  ... --backend cpu           # reference path, no GPU
    ./llm bench models/Llama-3.2-3B-Instruct-Q4_0.gguf -p 128 -n 32   # same shape as llama-bench
    ./llm chat  models/Llama-3.2-3B-Instruct-Q4_0.gguf                 # multi-turn REPL, KV cache kept across turns
    ./llm serve models/Llama-3.2-3B-Instruct-Q4_0.gguf --port 8080     # OpenAI-compatible API

The server speaks the OpenAI chat API, so any client works:

    curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

Sampling options for run, chat and serve: temperature, top-k, top-p, repeat penalty, seed.

## Tests

    python3 tests/test_tokenizer.py                  # token ids identical to llama-tokenize
    python3 tests/test_generate.py --backend cpu     # greedy text identical to llama-simple
    python3 tests/test_generate.py --backend metal
    tests/bench.sh                                   # scoreboard vs llama.cpp on Metal

## How it works

- **Decode** (one token at a time) is memory-bound: every weight byte is read once per token.
  The matvec kernels stream quantized blocks with one SIMD group per 4 rows, sharing the x
  loads, and never shift nibbles: the masked value is multiplied by a pre-scaled x instead.
- **Prefill** (the prompt) is compute-bound and runs as 64x32x32 tile products on the M5's
  matrix units through the Metal 4 tensor API (`mpp::tensor_ops::matmul2d`). Weights are
  dequantized to f16 tiles in threadgroup memory; the output tile aliases the input tiles so
  a threadgroup uses 8 KB and four of them fit per GPU core.
- **Attention** has two flash-style kernels over an f16 KV cache. Decode splits the keys
  across threadgroups, shares each K/V row across the 3 query heads of a GQA group, keeps a
  running max and sum (online softmax), and merges the partials in a second tiny kernel.
  Prefill runs S = QK^T and O += PV as 8x8 tile products on the SIMD-group matrix units, one
  SIMD group per (head, 8 queries), rescaling O with a diagonal matrix when the max grows.
  Context length is limited only by KV cache memory.
- Everything else (RMSNorm, RoPE, SwiGLU) is a small kernel; ~14 dispatches per layer, one
  command buffer per forward pass.

## Scoreboard (M5, 24 GB, tokens/sec, measured back to back with `tests/bench.sh`)

| Model | Prefill llama.cpp | Prefill ours | Decode llama.cpp | Decode ours |
|---|---|---|---|---|
| Q4_0 | 1496 | 1190 | 51.7 | 46.2 |
| Q8_0 | 1406 | 1051 | 26.0 | 28.6 |
| F16  | 1127 | 864  | 14.8 | 15.6 |

Absolute numbers drift with thermals and memory pressure; the two engines are always run
in the same conditions.

Long context (Q4_0, ours): prefill 1252 tok/s at 1k and 1010 tok/s at 4k; decode 45 tok/s
at 1k and 40 tok/s at 4k. Before the flash kernels these were 739/243 and 41/21.

## Diagnostics

    LLM_PROFILE=1 ./llm bench ...        GPU time vs wall time per forward
    LLM_SKIP=attn,norm ./llm bench ...   skip kernel families (wrong output, right timing)
    LLM_NO_TENSOR=1 ./llm bench ...      force the SIMD-group matmul instead of the tensor API
    LLM_MV_NSG=2 LLM_MV_NR=4 ...         matvec shape (SIMD groups per threadgroup, rows per SIMD group)
    LLM_NAIVE_ATTN=1 ...                 use the simple attention kernel (reference)
    LLM_ATTN_SPLITS=8 ...                key splits for decode attention
    ./llm kcheck <model> [n_tok] [tensor] compare the tensor-API matmul against the SIMD-group one, per token
    ./llm abench <model> [pos]           time decode attention alone at a position
    ./llm bwtest <model> [MB]            GPU read bandwidth per buffer type
