// Llama 3 chat template as token sequences, so special tokens are inserted by id and the
// text of each turn is tokenized on its own.
//
//   <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
//   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>\n\n{reply}<|eot_id|>
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Tokenizer;

class ChatFormat {
public:
    explicit ChatFormat(const Tokenizer& tok);
    // BOS plus the system turn. Llama 3.2's official template always emits the system header
    // with the knowledge-cutoff and today's date; `system` may be empty.
    std::vector<int32_t> conversation_start(const std::string& system) const;
    // One complete message: header, text, <|eot_id|>. role is "user", "assistant", "system", ...
    std::vector<int32_t> message(const std::string& role, const std::string& text) const;
    std::vector<int32_t> assistant_start() const;                    // header the model continues from
    std::vector<int32_t> user_turn(const std::string& text) const;   // message("user") + assistant_start()
    std::vector<int32_t> end_of_turn() const;                        // <|eot_id|>
    bool is_stop(int32_t id) const;

private:
    void add_text(std::vector<int32_t>& ids, const std::string& s) const;
    const Tokenizer& tok_;
    int32_t bos_, start_header_, end_header_, eot_, eos_, eom_;
};
