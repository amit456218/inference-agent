#include "tokenizer.h"

#include <climits>
#include <stdexcept>

#include "gguf.h"
#include "unicode.h"

namespace {

enum TokenType { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6 };

bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
bool is_ws(uint32_t c) { return unicode::is_whitespace(c); }
bool is_L(uint32_t c) { return unicode::is_letter(c); }
bool is_N(uint32_t c) { return unicode::is_number(c); }

// Hand-coded matcher for the Llama 3 pre-tokenizer regex, evaluated at position i over
// codepoints. Alternatives are tried in order, as regex alternation does. Returns the
// number of codepoints matched (always >= 1).
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
size_t match_at(const std::vector<unicode::Codepoint>& cps, size_t i) {
    const size_t n = cps.size();
    auto at = [&](size_t k) -> uint32_t { return k < n ? cps[k].cp : 0; };
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };

    // 1. contractions
    if (at(i) == '\'' && i + 1 < n) {
        uint32_t a = lower(at(i + 1)), b = lower(at(i + 2));
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    // 2. optional non-letter/number/newline prefix, then letters
    {
        size_t j = i;
        if (!is_crlf(at(j)) && !is_L(at(j)) && !is_N(at(j))) j++;
        if (j < n && is_L(at(j))) {
            while (j < n && is_L(at(j))) j++;
            return j - i;
        }
    }
    // 3. up to three digits
    if (is_N(at(i))) {
        size_t j = i;
        while (j < n && j - i < 3 && is_N(at(j))) j++;
        return j - i;
    }
    // 4. optional space, then a run of "other" chars, then trailing newlines
    {
        size_t j = i;
        if (at(j) == ' ') j++;
        if (j < n && !is_ws(at(j)) && !is_L(at(j)) && !is_N(at(j))) {
            while (j < n && !is_ws(at(j)) && !is_L(at(j)) && !is_N(at(j))) j++;
            while (j < n && is_crlf(at(j))) j++;
            return j - i;
        }
    }
    // 5, 6, 7 all start with whitespace
    if (!is_ws(at(i))) return 1;  // unreachable in practice: everything else matched above
    size_t j = i;
    while (j < n && is_ws(at(j))) j++;  // maximal whitespace run [i, j)
    // 5. \s*[\r\n]+  -> ends after the last newline inside the run
    for (size_t k = j; k > i; k--) {
        if (is_crlf(at(k - 1))) return k - i;
    }
    // 6. \s+(?!\S)   -> whole run if at end of text, else leave the last char for the next token
    if (j == n) return j - i;
    if (j - i > 1) return j - i - 1;
    // 7. \s+
    return 1;
}

}  // namespace

Tokenizer::Tokenizer(const gguf::File& f) {
    if (f.get_str("tokenizer.ggml.model", "") != "gpt2" || f.get_str("tokenizer.ggml.pre", "") != "llama-bpe")
        throw std::runtime_error("tokenizer: only Llama 3 style BPE (model=gpt2, pre=llama-bpe) is supported");

    tokens_ = f.get("tokenizer.ggml.tokens").arr_s;
    const auto& types = f.get("tokenizer.ggml.token_type").arr_i;
    types_.assign(types.begin(), types.end());
    if (types_.size() != tokens_.size()) throw std::runtime_error("tokenizer: token/type count mismatch");

    token_to_id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); i++) token_to_id_[tokens_[i]] = int32_t(i);

    const auto& merges = f.get("tokenizer.ggml.merges").arr_s;
    merge_rank_.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); i++) merge_rank_[merges[i]] = int32_t(i);

    // GPT-2 byte encoder: printable Latin-1 bytes map to themselves, the other 68 bytes
    // map to U+0100 upward in increasing byte order.
    uint32_t next = 256;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = printable ? uint32_t(b) : next++;
        unicode::append_utf8(byte_to_char_[b], cp);
        char_to_byte_[byte_to_char_[b]] = uint8_t(b);
    }

    bos_ = int32_t(f.get_int("tokenizer.ggml.bos_token_id", -1));
    eos_ = int32_t(f.get_int("tokenizer.ggml.eos_token_id", -1));
    eot_ = token_id("<|eot_id|>");
    eom_ = token_id("<|end_of_text|>");
}

int32_t Tokenizer::token_id(const std::string& text) const {
    auto it = token_to_id_.find(text);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool Tokenizer::is_control(int32_t id) const {
    return id >= 0 && size_t(id) < types_.size() && (types_[id] == CONTROL || types_[id] == USER_DEFINED);
}

std::vector<std::pair<uint32_t, uint32_t>> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<unicode::Codepoint> cps = unicode::decode_utf8(text);
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (size_t i = 0; i < cps.size();) {
        size_t len = match_at(cps, i);
        uint32_t start = cps[i].offset;
        uint32_t end = cps[i + len - 1].offset + cps[i + len - 1].len;
        out.emplace_back(start, end);
        i += len;
    }
    return out;
}

std::vector<int32_t> Tokenizer::bpe(const std::string& chunk) const {
    // Symbols start as one byte-encoding char per byte.
    std::vector<std::string> symbols;
    symbols.reserve(chunk.size());
    std::string whole;
    for (unsigned char b : chunk) { symbols.push_back(byte_to_char_[b]); whole += byte_to_char_[b]; }

    // If the entire chunk is a vocab entry, use it without running merges. tiktoken does this,
    // and so does llama.cpp for Llama 3 (ignore_merges). The merge path does not always reach it.
    if (int32_t id = token_id(whole); id >= 0) return {id};

    // Merge the lowest-ranked adjacent pair, leftmost on ties, until nothing merges.
    std::string key;
    while (symbols.size() > 1) {
        int best = -1, best_rank = INT_MAX;
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            key.assign(symbols[i]); key += ' '; key += symbols[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) { best_rank = it->second; best = int(i); }
        }
        if (best < 0) break;
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + best + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const std::string& s : symbols) {
        int32_t id = token_id(s);
        if (id >= 0) { ids.push_back(id); continue; }
        // Not a vocab entry: fall back to the single-byte tokens.
        for (size_t p = 0; p < s.size();) {
            size_t len = unicode::utf8_len(uint8_t(s[p]));
            int32_t bid = token_id(s.substr(p, len));
            if (bid < 0) throw std::runtime_error("tokenizer: byte token missing from vocab");
            ids.push_back(bid);
            p += len;
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_ >= 0) ids.push_back(bos_);
    for (auto [start, end] : pretokenize(text)) {
        std::vector<int32_t> part = bpe(text.substr(start, end - start));
        ids.insert(ids.end(), part.begin(), part.end());
    }
    return ids;
}

std::string Tokenizer::token_to_piece(int32_t id) const {
    if (id < 0 || size_t(id) >= tokens_.size()) return "";
    const std::string& t = tokens_[id];
    if (is_control(id)) return t;
    std::string out;
    out.reserve(t.size());
    for (size_t p = 0; p < t.size();) {
        size_t len = unicode::utf8_len(uint8_t(t[p]));
        auto it = char_to_byte_.find(t.substr(p, len));
        if (it == char_to_byte_.end()) out.append(t, p, len);  // shouldn't happen; keep the text
        else out += char(it->second);
        p += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (skip_special && is_control(id)) continue;
        out += token_to_piece(id);
    }
    return out;
}
