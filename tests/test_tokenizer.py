#!/usr/bin/env python3
"""Compares our tokenizer against llama.cpp's llama-tokenize on a corpus of tricky strings.
Usage: python3 tests/test_tokenizer.py models/<model>.gguf"""
import subprocess, sys, tempfile, os

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/Llama-3.2-3B-Instruct-Q4_0.gguf"

CASES = [
    "Hello world",
    "Hello, world! How's it going?",
    "don't DON'T won't I'm you're we've they'll he'd 's 'S 'T 'LL 'll'd",
    "12345 6789 1 22 333 4444 1234567890 x1 1x",
    "   leading spaces and trailing   ",
    "multiple   spaces\tand\ttabs\t\t  mixed \t \t",
    "line one\nline two\n\n\nline five\r\nwindows\r\r\n\n end",
    "def foo(x):\n    return x * 2  # comment\n\n\nclass Bar:\n\tpass\n",
    "héllo wörld naïve café résumé Ünïcödé ÀÉÎÕÜ",
    "日本語のテキストです。中文文本。한국어 텍스트 第1章 第２章 ３４５",
    "I love 🍕 and 🎉🎉 party!! 👨‍👩‍👧‍👦 🇺🇸",
    "!!! ??? ... --- ### @@@ $$$ %%% ^^^ &&& *** ((( ))) [[[ ]]] {{{ }}} ||| \\\\ ;;; ::: ,,, <<< >>> === +++",
    "The price is $12.99 (about €11) or £10—cheap! 50% off; 3.14159",
    "https://example.com/path?query=1&other=2#frag email@example.com",
    "a b c　d​e",
    "مرحبا بالعالم שלום עולם नमस्ते दुनिया",
    "x² + y² = z², ½ + ¼ = ¾, Ⅳ Ⅸ",
    "a\x01b\x7fc\x1fd",
    "",
    " ",
    "\n",
    "\n\n",
    "  \n  ",
    "hello ",
    "hello\n",
    " hello",
    "hello  world",
    "hello   world",
    "a" * 300,
    " " * 100,
    "Привет мир Γειά σου Κόσμε สวัสดีชาวโลก",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
    "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX. tHe QuIcK.",
    "C++ and C# and F# and .NET and node.js and file.txt and __init__.py",
    "camelCaseWord snake_case_word kebab-case-word SCREAMING_SNAKE",
    "1st 2nd 3rd 4th 10th 100th 1,000,000 1.5e10 0x1F 0b1010",
    "  indented\n    more indented\n\tTabbed\n",
    "\"quoted\" 'single' `backtick` “smart” ‘quotes’ «guillemets»",
    " \\ \\\\ \\\\\\ \\\\\\\\ end",
    "C:\\Users\\name\\file.txt and \\\\server\\share",
    "regex: \\d+\\s*\\w? latex: \\frac{a}{b} \\alpha\\beta json: \"a\\nb\\t\"",
    "tab\\tnewline\\nquote\\' backslash-n at end\\n",
]

def run(cmd, text):
    p = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    lines = [l for l in p.stdout.decode("utf-8", "replace").splitlines() if l.startswith("[")]
    if not lines:
        return None, p.stderr.decode("utf-8", "replace")[-500:]
    return eval(lines[-1]), None

def show(s):
    return repr(s if len(s) < 60 else s[:57] + "...")

fails = 0
for text in CASES:
    ref, err1 = run(["llama-tokenize", "-m", MODEL, "--stdin", "--ids", "--no-bos", "--no-parse-special", "--no-escape"], text)
    ours, err2 = run(["./ingot", "tokenize", MODEL, "--stdin", "--no-bos"], text)
    if ref is None or ours is None:
        fails += 1
        print("ERROR", show(text), err1 or err2)
        continue
    if ref == ours:
        print("PASS %3d tokens  %s" % (len(ref), show(text)))
    else:
        fails += 1
        first = next((i for i in range(min(len(ref), len(ours))) if ref[i] != ours[i]), min(len(ref), len(ours)))
        print("FAIL", show(text))
        print("     ref :", ref)
        print("     ours:", ours)
        print("     first difference at index", first)
print("\n%d/%d passed" % (len(CASES) - fails, len(CASES)))
sys.exit(1 if fails else 0)
