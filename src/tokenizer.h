// Llama 3 tokenizer: byte-level BPE in the tiktoken/GPT-4 style.
//
// Encoding pipeline:
//   1. Pre-tokenize: split text into chunks with the Llama 3 regex (words, numbers in
//      groups of up to 3 digits, punctuation runs, whitespace runs, contractions).
//   2. Byte-encode each chunk: every byte becomes one printable Unicode char (GPT-2 scheme),
//      so the vocab and merge table are plain strings with no raw control bytes.
//   3. BPE: repeatedly merge the adjacent pair with the lowest merge rank until none apply.
//   4. Look up each final symbol in the vocab.
// Decoding reverses step 2 on each token's text.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gguf { class File; }

class Tokenizer {
public:
    explicit Tokenizer(const gguf::File& f);

    // Special tokens in `text` are NOT parsed; add them by id. add_bos prepends <|begin_of_text|>.
    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
    // Raw bytes for a token. Control tokens return their literal text, e.g. "<|eot_id|>".
    std::string token_to_piece(int32_t id) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

    int32_t token_id(const std::string& text) const;  // -1 if not in vocab
    bool is_control(int32_t id) const;
    size_t vocab_size() const { return tokens_.size(); }
    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }  // <|eot_id|> for instruct models: end of assistant turn
    int32_t eot() const { return eot_; }
    int32_t eom() const { return eom_; }  // <|end_of_text|>

    // Exposed for tests: the pre-tokenizer split, as byte ranges [start, end) of `text`.
    std::vector<std::pair<uint32_t, uint32_t>> pretokenize(const std::string& text) const;

private:
    std::vector<int32_t> bpe(const std::string& chunk) const;

    std::vector<std::string> tokens_;   // byte-encoded token text, index = id
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
    std::string byte_to_char_[256];                        // UTF-8 of the byte-encoding char
    std::unordered_map<std::string, uint8_t> char_to_byte_;
    int32_t bos_ = -1, eos_ = -1, eot_ = -1, eom_ = -1;
};
