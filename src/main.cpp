#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <chrono>

#include "backend.h"
#include "chat.h"
#include "cpu_backend.h"
#include "engine.h"
#include "gguf.h"
#include "metal_backend.h"
#include "model.h"
#include "server.h"
#include "sampler.h"
#include "tokenizer.h"

static void usage() {
    fprintf(stderr,
            "usage:\n"
            "  ingot info <model.gguf>                          print model metadata and tensor table\n"
            "  ingot tokenize <model.gguf> [opts] [-p TEXT]     print token ids as [a, b, c]\n"
            "       --no-bos       do not prepend <|begin_of_text|>\n"
            "       --file F       read text from file      --stdin   read text from stdin\n"
            "       --pieces       also print each token's text\n"
            "  ingot detokenize <model.gguf> ID...              print the text for token ids\n"
            "  ingot bench <model.gguf> [-p 128] [-n 32] [-r 3] [--backend B]   measure prompt and generation tok/s\n"
            "  ingot kbench <model.gguf> [-n 128] [-i 20] [-t blk.0.ffn_up.weight]   time one matmul kernel\n"
            "  ingot run <model.gguf> -p TEXT [opts]            generate text\n"
            "       -n N           max new tokens (default 32)   --ctx N   context size (default 2048)\n"
            "       --backend B    metal (default) or cpu    --batch N   prompt tokens per GPU pass (default 512)\n"
            "       --chat         wrap the prompt in the Llama 3 chat template\n"
            "       --ids          print prompt and generated token ids to stderr\n"
            "  ingot chat <model.gguf> [opts]                   interactive chat (/reset, /quit)\n"
            "  ingot serve <model.gguf> [--port 8080] [--host 127.0.0.1] [--ctx 8192]   OpenAI-compatible API\n"
            "       --system TEXT  system prompt              -n N      max tokens per reply (default 1024)\n"
            "  sampling options for run and chat (run defaults to greedy, chat to temp 0.7):\n"
            "       --temp T  --top-k K  --top-p P  --repeat-penalty R  --seed S\n");
}

static std::string shape_str(const gguf::TensorInfo& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.ne.size(); i++) {
        if (i) s += ", ";
        s += std::to_string(t.ne[i]);
    }
    return s + "]";
}

static int cmd_info(const std::string& path) {
    gguf::File f = gguf::File::open(path);
    printf("file:        %s\n", path.c_str());
    printf("gguf:        v%u, %.2f GB, tensor data starts at byte %zu\n",
           f.version(), f.file_size() / 1e9, f.data_offset());

    std::vector<std::string> keys;
    for (const auto& [k, v] : f.metadata()) keys.push_back(k);
    std::sort(keys.begin(), keys.end());
    printf("\nmetadata (%zu keys):\n", keys.size());
    for (const std::string& k : keys) {
        const gguf::Value& v = f.get(k);
        printf("  %-48s ", k.c_str());
        switch (v.type) {
            case gguf::ValueType::STRING: {
                std::string s = v.s.substr(0, 60);
                std::replace(s.begin(), s.end(), '\n', ' ');
                printf("\"%s\"%s\n", s.c_str(), v.s.size() > 60 ? "..." : "");
                break;
            }
            case gguf::ValueType::FLOAT32:
            case gguf::ValueType::FLOAT64: printf("%g\n", v.f); break;
            case gguf::ValueType::ARRAY: printf("[array of %zu]\n", v.array_size()); break;
            default: printf("%lld\n", (long long)v.i); break;
        }
    }

    printf("\ntensors (%zu):\n", f.tensors().size());
    std::map<std::string, size_t> bytes_by_type;
    uint64_t params = 0;
    for (const gguf::TensorInfo& t : f.tensors()) {
        bool first_block = t.name.rfind("blk.0.", 0) == 0;
        bool layer = t.name.rfind("blk.", 0) == 0;
        if (!layer || first_block)
            printf("  %-28s %-5s %-22s %10.2f MB\n", t.name.c_str(), gguf::type_name(t.type),
                   shape_str(t).c_str(), t.nbytes() / 1e6);
        bytes_by_type[gguf::type_name(t.type)] += t.nbytes();
        params += t.nelements();
    }
    printf("  ... (blk.1 through blk.N follow the same pattern)\n");
    printf("\ntotal: %.3f B parameters\n", params / 1e9);
    for (const auto& [name, bytes] : bytes_by_type) printf("  %-5s %8.2f MB\n", name.c_str(), bytes / 1e6);
    return 0;
}

static std::string read_all(std::istream& in) {
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static int cmd_tokenize(int argc, char** argv) {
    std::string model = argv[2];
    std::string text;
    bool have_text = false, add_bos = true, pieces = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--no-bos") add_bos = false;
        else if (a == "--pieces") pieces = true;
        else if (a == "--stdin") { text = read_all(std::cin); have_text = true; }
        else if (a == "--file" && i + 1 < argc) {
            std::ifstream in(argv[++i], std::ios::binary);
            if (!in) throw std::runtime_error(std::string("cannot open ") + argv[i]);
            text = read_all(in); have_text = true;
        } else if (a == "-p" && i + 1 < argc) { text = argv[++i]; have_text = true; }
        else throw std::runtime_error("unknown option " + a);
    }
    if (!have_text) throw std::runtime_error("no text given (use -p, --file or --stdin)");

    gguf::File f = gguf::File::open(model);
    Tokenizer tok(f);
    std::vector<int32_t> ids = tok.encode(text, add_bos);
    printf("[");
    for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
    if (pieces) {
        for (int32_t id : ids) {
            std::string p = tok.token_to_piece(id), shown;
            for (unsigned char c : p) {
                if (c == '\n') shown += "\\n";
                else if (c == '\t') shown += "\\t";
                else if (c < 0x20 || c == 0x7f) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); shown += b; }
                else shown += char(c);
            }
            printf("%7d  '%s'\n", id, shown.c_str());
        }
    }
    return 0;
}

static int cmd_detokenize(int argc, char** argv) {
    gguf::File f = gguf::File::open(argv[2]);
    Tokenizer tok(f);
    std::vector<int32_t> ids;
    for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));
    std::string s = tok.decode(ids, /*skip_special=*/false);
    fwrite(s.data(), 1, s.size(), stdout);
    printf("\n");
    return 0;
}


// Writes the longest prefix of buf that ends on a complete UTF-8 sequence, keeps the rest.
static void flush_utf8(std::string& buf) {
    size_t keep = 0;
    for (size_t i = buf.size(); i > 0 && i + 4 > buf.size(); i--) {
        unsigned char c = buf[i - 1];
        if ((c & 0xC0) == 0x80) continue;                 // continuation byte, look further back
        size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (buf.size() - (i - 1) < need) keep = buf.size() - (i - 1);  // incomplete sequence at the end
        break;
    }
    fwrite(buf.data(), 1, buf.size() - keep, stdout);
    fflush(stdout);
    buf.erase(0, buf.size() - keep);
}

static void print_ids(const char* label, const std::vector<int32_t>& ids) {
    fprintf(stderr, "%s [", label);
    for (size_t i = 0; i < ids.size(); i++) fprintf(stderr, "%s%d", i ? ", " : "", ids[i]);
    fprintf(stderr, "]\n");
}

// Parses one sampling option at argv[i]; returns true if it consumed it.
static bool parse_sampling_option(int argc, char** argv, int& i, SamplerParams& sp) {
    std::string a = argv[i];
    if (a == "--temp" && i + 1 < argc) sp.temperature = float(atof(argv[++i]));
    else if (a == "--top-k" && i + 1 < argc) sp.top_k = atoi(argv[++i]);
    else if (a == "--top-p" && i + 1 < argc) sp.top_p = float(atof(argv[++i]));
    else if (a == "--repeat-penalty" && i + 1 < argc) sp.repeat_penalty = float(atof(argv[++i]));
    else if (a == "--seed" && i + 1 < argc) sp.seed = strtoull(argv[++i], nullptr, 10);
    else return false;
    return true;
}

static int cmd_run(int argc, char** argv) {
    std::string model_path = argv[2], prompt, backend_name = "metal";
    int n_predict = 32, n_ctx = 2048, n_batch = 512;
    bool chat = false, show_ids = false, have_prompt = false;
    SamplerParams sp;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (parse_sampling_option(argc, argv, i, sp)) continue;
        if (a == "-p" && i + 1 < argc) { prompt = argv[++i]; have_prompt = true; }
        else if (a == "-n" && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (a == "--batch" && i + 1 < argc) n_batch = atoi(argv[++i]);
        else if (a == "--chat") chat = true;
        else if (a == "--ids") show_ids = true;
        else throw std::runtime_error("unknown option " + a);
    }
    if (!have_prompt) throw std::runtime_error("no prompt given (-p TEXT)");

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    gguf::File f = gguf::File::open(model_path);
    Tokenizer tok(f);
    Model model = Model::load(f);
    std::unique_ptr<Backend> backend = make_backend(backend_name, f, model, n_ctx, n_batch);
    auto t1 = clock::now();

    std::vector<int32_t> ids;
    if (chat) {
        ChatFormat fmt(tok);
        ids = fmt.conversation_start("");
        std::vector<int32_t> turn = fmt.user_turn(prompt);
        ids.insert(ids.end(), turn.begin(), turn.end());
    } else {
        ids = tok.encode(prompt, true);
    }
    if (show_ids) print_ids("prompt:", ids);
    if (int(ids.size()) + n_predict > n_ctx) throw std::runtime_error("prompt + n_predict exceeds --ctx");

    Sampler sampler(sp, model.cfg.n_vocab);
    int pos = 0;
    const float* logits = prefill(*backend, ids, pos, n_batch);
    auto t2 = clock::now();

    std::vector<int32_t> generated;
    std::string buf;
    for (int i = 0; i < n_predict; i++) {
        // The backend owns the logits buffer; the sampler edits it in place, which is fine.
        int32_t next = sampler.sample(const_cast<float*>(logits), generated);
        if (next == tok.eos() || next == tok.eot() || next == tok.eom()) break;
        generated.push_back(next);
        buf += tok.token_to_piece(next);
        flush_utf8(buf);
        if (i + 1 < n_predict) logits = backend->forward(next, pos++);
    }
    fwrite(buf.data(), 1, buf.size(), stdout);
    printf("\n");
    auto t3 = clock::now();

    auto secs = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    if (show_ids) print_ids("generated:", generated);
    fprintf(stderr, "\n[%s] load %.2fs | prompt %zu tok in %.2fs (%.1f tok/s) | gen %zu tok in %.2fs (%.2f tok/s)\n",
            backend->name(), secs(t0, t1), ids.size(), secs(t1, t2), ids.size() / secs(t1, t2), generated.size(), secs(t2, t3),
            generated.size() / secs(t2, t3));
    return 0;
}

// Mirrors llama-bench: pp = tokens/sec processing an n_prompt-token prompt, tg = tokens/sec
// generating n_gen tokens one at a time. Repeated r times; reports mean and stddev.
static int cmd_chat(int argc, char** argv) {
    std::string model_path = argv[2], backend_name = "metal", system;
    int n_predict = 1024, n_ctx = 8192, n_batch = 512;
    SamplerParams sp;
    sp.temperature = 0.7f;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (parse_sampling_option(argc, argv, i, sp)) continue;
        if (a == "--system" && i + 1 < argc) system = argv[++i];
        else if (a == "-n" && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) n_batch = atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else throw std::runtime_error("unknown option " + a);
    }
    gguf::File f = gguf::File::open(model_path);
    Tokenizer tok(f);
    Model model = Model::load(f);
    std::unique_ptr<Backend> backend = make_backend(backend_name, f, model, n_ctx, n_batch);
    ChatFormat fmt(tok);
    Sampler sampler(sp, model.cfg.n_vocab);

    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };

    fprintf(stderr, "%s | %s backend | ctx %d | temp %.2f top-k %d top-p %.2f | /reset clears, /quit exits\n",
            model_path.substr(model_path.find_last_of('/') + 1).c_str(), backend->name(), n_ctx, sp.temperature,
            sp.top_k, sp.top_p);
    int pos = 0;
    std::vector<int32_t> history;  // for the repetition penalty
    std::string line;
    while (true) {
        fprintf(stderr, "\n> ");
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") { pos = 0; history.clear(); fprintf(stderr, "(context cleared)\n"); continue; }
        if (line.empty()) continue;

        std::vector<int32_t> ids;
        if (pos == 0) ids = fmt.conversation_start(system);
        std::vector<int32_t> turn = fmt.user_turn(line);
        ids.insert(ids.end(), turn.begin(), turn.end());
        if (pos + int(ids.size()) + 8 > n_ctx) { fprintf(stderr, "(context full: /reset to start over)\n"); continue; }

        auto t0 = clock::now();
        const float* logits = prefill(*backend, ids, pos, n_batch);
        history.insert(history.end(), ids.begin(), ids.end());
        auto t1 = clock::now();

        std::string buf;
        int generated = 0;
        bool stopped = false;
        while (generated < n_predict && pos < n_ctx) {
            int32_t next = sampler.sample(const_cast<float*>(logits), history);
            if (fmt.is_stop(next)) { stopped = true; break; }
            history.push_back(next);
            generated++;
            buf += tok.token_to_piece(next);
            flush_utf8(buf);
            if (pos >= n_ctx) break;
            logits = backend->forward(next, pos++);
        }
        fwrite(buf.data(), 1, buf.size(), stdout);
        printf("\n");
        fflush(stdout);
        auto t2 = clock::now();
        // Keep the template consistent for the next turn: the reply's <|eot_id|> must be in the
        // cache. If the model stopped by itself we run its stop token; otherwise we append one.
        std::vector<int32_t> eot = fmt.end_of_turn();
        if (pos < n_ctx) {
            backend->forward(eot[0], pos++);
            history.push_back(eot[0]);
        }
        (void)stopped;
        fprintf(stderr, "[prompt %zu tok, %.0f tok/s | reply %d tok, %.1f tok/s | ctx %d/%d]\n", ids.size(),
                ids.size() / secs(t0, t1), generated, generated / secs(t1, t2), pos, n_ctx);
    }
    return 0;
}

static int cmd_bench(int argc, char** argv) {
    std::string model_path = argv[2], backend_name = "metal";
    int n_prompt = 128, n_gen = 32, reps = 3, n_batch = 512;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-p" && i + 1 < argc) n_prompt = atoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (a == "--batch" && i + 1 < argc) n_batch = atoi(argv[++i]);
        else throw std::runtime_error("unknown option " + a);
    }
    gguf::File f = gguf::File::open(model_path);
    Tokenizer tok(f);
    Model model = Model::load(f);
    std::unique_ptr<Backend> backend = make_backend(backend_name, f, model, n_prompt + n_gen + 8, n_batch);

    // A deterministic prompt of ordinary-looking tokens.
    std::string text;
    while (int(tok.encode(text, true).size()) < n_prompt) text += "The quick brown fox jumps over the lazy dog. ";
    std::vector<int32_t> prompt = tok.encode(text, true);
    prompt.resize(n_prompt);

    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    auto stats = [](const std::vector<double>& v) {
        double mean = 0, var = 0;
        for (double x : v) mean += x;
        mean /= v.size();
        for (double x : v) var += (x - mean) * (x - mean);
        return std::make_pair(mean, sqrt(var / v.size()));
    };

    // Warm-up: page in the weights and compile every pipeline (batched and single-token) once.
    backend->forward_batch(prompt.data(), std::min(n_batch, n_prompt), 0);
    backend->forward(prompt[0], 0);
    std::vector<double> pp, tg;
    for (int r = 0; r < reps; r++) {
        int pos = 0;
        auto t0 = clock::now();
        const float* logits = nullptr;
        for (int i = 0; i < n_prompt; i += n_batch) {
            int n = std::min(n_batch, n_prompt - i);
            logits = backend->forward_batch(prompt.data() + i, n, pos);
            pos += n;
        }
        auto t1 = clock::now();
        int32_t next = 0;
        for (int i = 0; i < n_gen; i++) {
            next = int32_t(std::max_element(logits, logits + model.cfg.n_vocab) - logits);
            logits = backend->forward(next, pos++);
        }
        auto t2 = clock::now();
        pp.push_back(n_prompt / secs(t0, t1));
        tg.push_back(n_gen / secs(t1, t2));
    }
    auto [pp_mean, pp_sd] = stats(pp);
    auto [tg_mean, tg_sd] = stats(tg);
    std::string name = model_path.substr(model_path.find_last_of('/') + 1);
    printf("%-36s %-6s pp%-4d %8.2f ± %-6.2f tok/s   tg%-4d %7.2f ± %-5.2f tok/s\n", name.c_str(), backend->name(),
           n_prompt, pp_mean, pp_sd, n_gen, tg_mean, tg_sd);
    return 0;
}

static int cmd_kbench(int argc, char** argv) {
    std::string tensor = "blk.0.ffn_up.weight";
    int n_tok = 128, iters = 20;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) n_tok = atoi(argv[++i]);
        else if (a == "-i" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) tensor = argv[++i];
        else throw std::runtime_error("unknown option " + a);
    }
    gguf::File f = gguf::File::open(argv[2]);
    Model model = Model::load(f);
    const gguf::TensorInfo& W = f.tensor(tensor);
    double gflops = metal_linear_bench(f, model, tensor, n_tok, iters);
    printf("%-24s %-5s [%llu x %llu] n_tok=%-4d %8.1f GFLOP/s\n", tensor.c_str(), gguf::type_name(W.type),
           (unsigned long long)W.ne[1], (unsigned long long)W.ne[0], n_tok, gflops);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "info" && argc == 3) return cmd_info(argv[2]);
        if (cmd == "tokenize" && argc >= 3) return cmd_tokenize(argc, argv);
        if (cmd == "detokenize" && argc >= 3) return cmd_detokenize(argc, argv);
        if (cmd == "run" && argc >= 3) return cmd_run(argc, argv);
        if (cmd == "chat" && argc >= 3) return cmd_chat(argc, argv);
        if (cmd == "serve" && argc >= 3) return cmd_serve(argc, argv);
        if (cmd == "bench" && argc >= 3) return cmd_bench(argc, argv);
        if (cmd == "kbench" && argc >= 3) return cmd_kbench(argc, argv);
        if (cmd == "kcheck" && argc >= 3) {
            gguf::File f = gguf::File::open(argv[2]);
            Model model = Model::load(f);
            metal_linear_check(f, model, argc > 4 ? argv[4] : "blk.0.ffn_up.weight", argc > 3 ? atoi(argv[3]) : 32);
            return 0;
        }
        if (cmd == "abench" && argc >= 3) {
            gguf::File f = gguf::File::open(argv[2]);
            Model model = Model::load(f);
            int pos = argc > 3 ? atoi(argv[3]) : 4095;
            double secs = metal_attention_bench(f, model, pos, 10);
            double bytes = 2.0 * model.cfg.n_layer * double(pos + 1) * model.cfg.n_head_kv * model.cfg.head_dim * 2;
            printf("decode attention at pos %d: %.2f ms/token, KV read %.0f MB -> %.1f GB/s\n", pos, secs * 1e3, bytes / 1e6, bytes / secs / 1e9);
            return 0;
        }
        if (cmd == "bwtest" && argc >= 3) {
            gguf::File f = gguf::File::open(argv[2]);
            Model model = Model::load(f);
            metal_bandwidth_test(f, model, argc > 3 ? size_t(atof(argv[3]) * 1e6) : size_t(512e6));
            return 0;
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
