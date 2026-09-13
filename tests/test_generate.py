#!/usr/bin/env python3
"""Greedy-generation regression test: our engine vs llama.cpp (llama-simple) must produce
identical text for every prompt and model. Usage: test_generate.py [--backend cpu|metal] [models...]"""
import subprocess, sys, os, time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
extra = []
if "--backend" in sys.argv:
    extra = ["--backend", sys.argv[sys.argv.index("--backend") + 1]]
    args = [a for a in args if a != extra[1]]
MODELS = args or ["models/Llama-3.2-3B-Instruct-F16.gguf", "models/Llama-3.2-3B-Instruct-Q8_0.gguf",
                  "models/Llama-3.2-3B-Instruct-Q4_0.gguf"]
PROMPTS = [
    ("The capital of France is", 16),
    ("Once upon a time, in a small village,", 48),
    ("def fibonacci(n):\n    ", 40),
    ("Q: What is 17 times 23?\nA:", 24),
    ("The three laws of thermodynamics are:\n1.", 64),
    ("Translate to French: 'The weather is nice today.'\nFrench:", 20),
    ("import numpy as np\n\n# Compute the eigenvalues of a random matrix\n", 48),
    ("Here is a haiku about the ocean:\n", 24),
    # Long prompts: prefill runs the tiled matmul and flash attention over several 32-token tiles,
    # including partial ones. Short prompts never exercise the second half of a tile.
    ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
     "How vexingly quick daft zebras jump. The five boxing wizards jump quickly. "
     "Sphinx of black quartz, judge my vow. The sentence after these pangrams is:", 24),
    ("In 1969, the Apollo 11 mission landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin "
     "spent about two and a quarter hours outside the spacecraft, collecting samples and setting up experiments, "
     "while Michael Collins remained in lunar orbit aboard the command module. The mission fulfilled a national "
     "goal set by President Kennedy in 1961. After returning to Earth, the crew was quarantined for three weeks. "
     "The most important scientific result of the mission was", 32),
]

def ref(model, prompt, n):
    out = subprocess.run(["llama-simple", "-m", model, "-n", str(n), prompt], capture_output=True).stdout.decode("utf-8", "replace")
    prefix = "<|begin_of_text|>" + prompt
    assert out.startswith(prefix), out[:80]
    out = out[len(prefix):]
    return out[:-1] if out.endswith("\n") else out  # llama-simple prints a trailing newline

def ours(model, prompt, n):
    p = subprocess.run(["./llm", "run", model, "-p", prompt, "-n", str(n)] + extra, capture_output=True)
    out = p.stdout.decode("utf-8", "replace")
    stats = p.stderr.decode("utf-8", "replace").strip().splitlines()
    return out[:-1] if out.endswith("\n") else out, (stats[-1] if stats else "")

fails = 0
for model in MODELS:
    print("==", os.path.basename(model))
    for prompt, n in PROMPTS:
        r = ref(model, prompt, n)
        o, stats = ours(model, prompt, n)
        ok = r == o
        fails += not ok
        print(("PASS" if ok else "FAIL"), "%-46r" % prompt[:44], "|", stats.split("|")[-1].strip() if ok else "")
        if not ok:
            print("     ref :", repr(r))
            print("     ours:", repr(o))
print("\n%d failures" % fails)
sys.exit(1 if fails else 0)
