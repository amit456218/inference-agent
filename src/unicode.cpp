#include "unicode.h"

namespace unicode {

namespace {
bool in_ranges(const Range* r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}
}  // namespace

bool is_letter(uint32_t cp) { return in_ranges(kLetterRanges, kLetterRangesCount, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumberRanges, kNumberRangesCount, cp); }
bool is_whitespace(uint32_t cp) { return in_ranges(kWhitespaceRanges, kWhitespaceRangesCount, cp); }

size_t utf8_len(uint8_t b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1;
}

std::vector<Codepoint> decode_utf8(const std::string& s) {
    std::vector<Codepoint> out;
    out.reserve(s.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t n = s.size();
    for (size_t i = 0; i < n;) {
        uint8_t b = p[i];
        size_t len = utf8_len(b);
        uint32_t cp;
        bool ok = i + len <= n && len > 0 && (b < 0x80 || (b & 0xC0) != 0x80);
        if (ok) {
            if (len == 1) cp = b;
            else {
                cp = b & (0xFF >> (len + 1));
                for (size_t k = 1; k < len; k++) {
                    if ((p[i + k] & 0xC0) != 0x80) { ok = false; break; }
                    cp = (cp << 6) | (p[i + k] & 0x3F);
                }
            }
        }
        if (!ok) { cp = 0xFFFD; len = 1; }
        out.push_back({cp, uint32_t(i), uint32_t(len)});
        i += len;
    }
    return out;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += char(cp);
    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
}

}  // namespace unicode
