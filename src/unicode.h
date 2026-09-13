// Minimal Unicode support for the tokenizer: UTF-8 decode/encode and the three
// character classes the Llama 3 pre-tokenizer regex uses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace unicode {

struct Range { uint32_t lo, hi; };
extern const Range kLetterRanges[];     extern const size_t kLetterRangesCount;      // \p{L}
extern const Range kNumberRanges[];     extern const size_t kNumberRangesCount;      // \p{N}
extern const Range kWhitespaceRanges[]; extern const size_t kWhitespaceRangesCount;  // \s

bool is_letter(uint32_t cp);
bool is_number(uint32_t cp);
bool is_whitespace(uint32_t cp);

// One decoded codepoint and where it came from in the source bytes.
struct Codepoint {
    uint32_t cp;
    uint32_t offset;  // byte offset in the source string
    uint32_t len;     // byte length (1..4). Invalid bytes decode as U+FFFD with len 1.
};
std::vector<Codepoint> decode_utf8(const std::string& s);
void append_utf8(std::string& out, uint32_t cp);
// Length in bytes of the UTF-8 sequence starting with byte b (1 for invalid lead bytes).
size_t utf8_len(uint8_t b);

}  // namespace unicode
