#include "server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "engine.h"
#include "gguf.h"
#include "json.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace {

struct Request {
    std::string method, path, body;
};

// Reads one HTTP request (headers + Content-Length body). Returns false on EOF / bad request.
bool read_request(int fd, Request& req) {
    std::string data;
    char buf[8192];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        ssize_t n = read(fd, buf, sizeof buf);
        if (n <= 0) return false;
        data.append(buf, size_t(n));
        header_end = data.find("\r\n\r\n");
        if (data.size() > (1u << 20)) return false;
    }
    std::string head = data.substr(0, header_end);
    size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    req.method = head.substr(0, sp1);
    req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t content_length = 0;
    for (size_t pos = head.find("\r\n"); pos != std::string::npos; pos = head.find("\r\n", pos + 2)) {
        std::string line = head.substr(pos + 2, head.find("\r\n", pos + 2) - pos - 2);
        std::string lower;
        for (char c : line) lower += char(tolower(c));
        if (lower.rfind("content-length:", 0) == 0) content_length = size_t(atol(line.c_str() + 15));
    }
    req.body = data.substr(header_end + 4);
    while (req.body.size() < content_length) {
        ssize_t n = read(fd, buf, sizeof buf);
        if (n <= 0) return false;
        req.body.append(buf, size_t(n));
        if (req.body.size() > (64u << 20)) return false;
    }
    return true;
}

void write_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = write(fd, s.data() + off, s.size() - off);
        if (n <= 0) throw std::runtime_error("client went away");
        off += size_t(n);
    }
}

void respond(int fd, int status, const std::string& type, const std::string& body) {
    const char* reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : "Internal Server Error";
    write_all(fd, "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\nContent-Type: " + type +
                      "\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: close\r\n\r\n" + body);
}

void respond_error(int fd, int status, const std::string& message) {
    Json err = Json::object();
    err["error"] = Json::object();
    err["error"]["message"] = message;
    err["error"]["type"] = status == 400 ? "invalid_request_error" : "server_error";
    respond(fd, status, "application/json", err.dump());
}

std::string chunk(const std::string& s) {
    char len[32];
    snprintf(len, sizeof len, "%zx\r\n", s.size());
    return len + s + "\r\n";
}

struct Server {
    gguf::File file;
    Tokenizer tok;
    Model model;
    std::unique_ptr<Backend> backend;
    ChatFormat fmt;
    std::string model_name;
    int n_ctx, n_batch;

    Server(const std::string& path, const std::string& backend_name, int ctx, int batch)
        : file(gguf::File::open(path)), tok(file), model(Model::load(file)),
          backend(make_backend(backend_name, file, model, ctx, batch)), fmt(tok),
          model_name(path.substr(path.find_last_of('/') + 1)), n_ctx(ctx), n_batch(batch) {}

    // Builds the prompt from an OpenAI messages array. Returns false with `error` set if invalid.
    bool build_prompt(const Json& messages, std::vector<int32_t>& ids, std::string& error) {
        if (!messages.is_array() || messages.items().empty()) { error = "messages must be a non-empty array"; return false; }
        std::string system;
        size_t first = 0;
        if (messages.items()[0]["role"].as_string() == "system") { system = messages.items()[0]["content"].as_string(); first = 1; }
        ids = fmt.conversation_start(system);
        for (size_t i = first; i < messages.items().size(); i++) {
            const Json& m = messages.items()[i];
            std::string role = m["role"].as_string();
            if (role != "user" && role != "assistant") { error = "unsupported role: " + role; return false; }
            if (!m["content"].is_string()) { error = "message content must be a string"; return false; }
            std::vector<int32_t> part = fmt.message(role, m["content"].as_string());
            ids.insert(ids.end(), part.begin(), part.end());
        }
        std::vector<int32_t> start = fmt.assistant_start();
        ids.insert(ids.end(), start.begin(), start.end());
        return true;
    }

    void handle_chat(int fd, const Json& req) {
        std::vector<int32_t> ids;
        std::string error;
        if (!build_prompt(req["messages"], ids, error)) { respond_error(fd, 400, error); return; }
        SamplerParams sp;
        sp.temperature = float(req["temperature"].as_number(0.7));
        sp.top_p = float(req["top_p"].as_number(0.95));
        sp.seed = uint64_t(req["seed"].as_number(0));
        int max_tokens = int(req["max_tokens"].as_number(512));
        bool stream = req["stream"].as_bool(false);
        if (int(ids.size()) + 1 >= n_ctx) { respond_error(fd, 400, "prompt is " + std::to_string(ids.size()) + " tokens, context is " + std::to_string(n_ctx)); return; }
        max_tokens = std::min(max_tokens, n_ctx - int(ids.size()) - 1);

        const std::string id = "chatcmpl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000000);
        const int64_t created = int64_t(time(nullptr));
        auto make_chunk = [&](const std::string& content, const char* finish, bool role) {
            Json j = Json::object();
            j["id"] = id; j["object"] = "chat.completion.chunk"; j["created"] = created; j["model"] = model_name;
            Json choice = Json::object();
            choice["index"] = 0;
            choice["delta"] = Json::object();
            if (role) choice["delta"]["role"] = "assistant";
            if (!content.empty()) choice["delta"]["content"] = content;
            choice["finish_reason"] = finish ? Json(finish) : Json();
            j["choices"] = Json::array();
            j["choices"].push(choice);
            return "data: " + j.dump() + "\n\n";
        };

        if (stream) {
            write_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                          "Access-Control-Allow-Origin: *\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
            write_all(fd, chunk(make_chunk("", nullptr, true)));
        }

        auto t0 = std::chrono::steady_clock::now();
        int pos = 0;
        const float* logits = prefill(*backend, ids, pos, n_batch);
        auto t1 = std::chrono::steady_clock::now();
        Sampler sampler(sp, model.cfg.n_vocab);
        std::vector<int32_t> history(ids);
        std::string content, pending;
        const char* finish = "length";
        int generated = 0;
        while (generated < max_tokens) {
            int32_t next = sampler.sample(const_cast<float*>(logits), history);
            if (fmt.is_stop(next)) { finish = "stop"; break; }
            history.push_back(next);
            generated++;
            pending += tok.token_to_piece(next);
            std::string piece;
            take_complete_utf8(pending, piece);
            content += piece;
            if (stream && !piece.empty()) write_all(fd, chunk(make_chunk(piece, nullptr, false)));
            if (generated < max_tokens) logits = backend->forward(next, pos++);
        }
        content += pending;
        auto t2 = std::chrono::steady_clock::now();
        auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        fprintf(stderr, "[%s] prompt %zu tok (%.0f tok/s) | reply %d tok (%.1f tok/s) | %s\n", id.c_str(), ids.size(),
                ids.size() / secs(t0, t1), generated, generated / std::max(1e-9, secs(t1, t2)), finish);

        if (stream) {
            if (!pending.empty()) write_all(fd, chunk(make_chunk(pending, nullptr, false)));
            write_all(fd, chunk(make_chunk("", finish, false)));
            write_all(fd, chunk("data: [DONE]\n\n"));
            write_all(fd, "0\r\n\r\n");
            return;
        }
        Json j = Json::object();
        j["id"] = id; j["object"] = "chat.completion"; j["created"] = created; j["model"] = model_name;
        Json choice = Json::object();
        choice["index"] = 0;
        choice["message"] = Json::object();
        choice["message"]["role"] = "assistant";
        choice["message"]["content"] = content;
        choice["finish_reason"] = finish;
        j["choices"] = Json::array();
        j["choices"].push(choice);
        j["usage"] = Json::object();
        j["usage"]["prompt_tokens"] = int(ids.size());
        j["usage"]["completion_tokens"] = generated;
        j["usage"]["total_tokens"] = int(ids.size()) + generated;
        respond(fd, 200, "application/json", j.dump());
    }

    void handle(int fd) {
        Request req;
        if (!read_request(fd, req)) return;
        try {
            if (req.method == "OPTIONS") {
                write_all(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                              "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n");
            } else if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/health")) {
                Json j = Json::object();
                j["object"] = "list";
                j["data"] = Json::array();
                Json m = Json::object();
                m["id"] = model_name; m["object"] = "model"; m["owned_by"] = "local";
                j["data"].push(m);
                respond(fd, 200, "application/json", j.dump());
            } else if (req.method == "POST" && req.path == "/v1/chat/completions") {
                Json body;
                try { body = Json::parse(req.body); } catch (const std::exception& e) { respond_error(fd, 400, e.what()); return; }
                handle_chat(fd, body);
            } else {
                respond_error(fd, 404, "no route for " + req.method + " " + req.path);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "request failed: %s\n", e.what());
            try { respond_error(fd, 500, e.what()); } catch (...) {}
        }
    }
};

}  // namespace

int cmd_serve(int argc, char** argv) {
    std::string model_path = argv[2], backend_name = "metal", host = "127.0.0.1";
    int port = 8080, n_ctx = 8192, n_batch = 512;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) n_batch = atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else throw std::runtime_error("unknown option " + a);
    }
    signal(SIGPIPE, SIG_IGN);
    Server server(model_path, backend_name, n_ctx, n_batch);

    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error("socket() failed");
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("bad --host address");
    if (bind(listener, (sockaddr*)&addr, sizeof addr) != 0) throw std::runtime_error("bind failed on port " + std::to_string(port));
    if (listen(listener, 16) != 0) throw std::runtime_error("listen failed");
    fprintf(stderr, "serving %s on http://%s:%d/v1/chat/completions (%s backend, ctx %d)\n", server.model_name.c_str(),
            host.c_str(), port, server.backend->name(), n_ctx);
    while (true) {
        int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) continue;
        server.handle(fd);
        close(fd);
    }
}
