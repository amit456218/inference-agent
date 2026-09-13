#include "chat.h"

#include <ctime>
#include <stdexcept>

#include "tokenizer.h"

namespace {
int32_t special(const Tokenizer& tok, const char* name) {
    int32_t id = tok.token_id(name);
    if (id < 0) throw std::runtime_error(std::string("chat: tokenizer lacks ") + name);
    return id;
}
std::string today() {
    char buf[32];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%d %b %Y", std::localtime(&t));
    return buf;
}
}  // namespace

ChatFormat::ChatFormat(const Tokenizer& tok)
    : tok_(tok), bos_(tok.bos()), start_header_(special(tok, "<|start_header_id|>")),
      end_header_(special(tok, "<|end_header_id|>")), eot_(special(tok, "<|eot_id|>")),
      eos_(tok.eos()), eom_(tok.eom()) {}

void ChatFormat::add_text(std::vector<int32_t>& ids, const std::string& s) const {
    std::vector<int32_t> part = tok_.encode(s, false);
    ids.insert(ids.end(), part.begin(), part.end());
}

std::vector<int32_t> ChatFormat::conversation_start(const std::string& system) const {
    std::vector<int32_t> ids{bos_, start_header_};
    add_text(ids, "system");
    ids.push_back(end_header_);
    std::string body = "\n\nCutting Knowledge Date: December 2023\nToday Date: " + today() + "\n\n" + system;
    add_text(ids, body);
    ids.push_back(eot_);
    return ids;
}

std::vector<int32_t> ChatFormat::message(const std::string& role, const std::string& text) const {
    std::vector<int32_t> ids{start_header_};
    add_text(ids, role);
    ids.push_back(end_header_);
    add_text(ids, "\n\n" + text);
    ids.push_back(eot_);
    return ids;
}

std::vector<int32_t> ChatFormat::assistant_start() const {
    std::vector<int32_t> ids{start_header_};
    add_text(ids, "assistant");
    ids.push_back(end_header_);
    add_text(ids, "\n\n");
    return ids;
}

std::vector<int32_t> ChatFormat::user_turn(const std::string& text) const {
    std::vector<int32_t> ids = message("user", text);
    std::vector<int32_t> start = assistant_start();
    ids.insert(ids.end(), start.begin(), start.end());
    return ids;
}

std::vector<int32_t> ChatFormat::end_of_turn() const { return {eot_}; }

bool ChatFormat::is_stop(int32_t id) const { return id == eot_ || id == eos_ || id == eom_; }
