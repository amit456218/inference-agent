#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i)); }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++; }
    bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { i++; return true; } return false; }
    void expect(char c) { if (!eat(c)) fail("unexpected character"); }

    static void utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
        else { out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("bad unicode escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        i += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            i++;
            Json obj = Json::object();
            if (eat('}')) return obj;
            do {
                std::string key = string();
                expect(':');
                obj[key] = value();
            } while (eat(','));
            expect('}');
            return obj;
        }
        if (c == '[') {
            i++;
            Json arr = Json::array();
            if (eat(']')) return arr;
            do arr.push(value()); while (eat(','));
            expect(']');
            return arr;
        }
        if (c == '"') return Json(string());
        if (s.compare(i, 4, "true") == 0) { i += 4; return Json(true); }
        if (s.compare(i, 5, "false") == 0) { i += 5; return Json(false); }
        if (s.compare(i, 4, "null") == 0) { i += 4; return Json(); }
        char* end = nullptr;
        double d = strtod(s.c_str() + i, &end);
        if (end == s.c_str() + i) fail("unexpected token");
        i = size_t(end - s.c_str());
        return Json(d);
    }
};

void dump_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += char(c);
        }
    }
    out += '"';
}

void dump(const Json& j, std::string& out) {
    switch (j.type()) {
        case Json::NUL: out += "null"; break;
        case Json::BOOL: out += j.as_bool() ? "true" : "false"; break;
        case Json::NUMBER: {
            double v = j.as_number();
            char b[32];
            if (v == std::floor(v) && std::fabs(v) < 1e15) snprintf(b, sizeof b, "%lld", (long long)v);
            else snprintf(b, sizeof b, "%.17g", v);
            out += b;
            break;
        }
        case Json::STRING: dump_string(out, j.as_string()); break;
        case Json::ARRAY: {
            out += '[';
            bool first = true;
            for (const Json& v : j.items()) { if (!first) out += ','; first = false; dump(v, out); }
            out += ']';
            break;
        }
        case Json::OBJECT: {
            out += '{';
            bool first = true;
            for (const auto& [k, v] : j.fields()) { if (!first) out += ','; first = false; dump_string(out, k); out += ':'; dump(v, out); }
            out += '}';
            break;
        }
    }
}

}  // namespace

const Json& Json::operator[](const std::string& key) const {
    static const Json null;
    if (type_ != OBJECT) return null;
    auto it = o_.find(key);
    return it == o_.end() ? null : it->second;
}

Json& Json::operator[](const std::string& key) {
    type_ = OBJECT;
    return o_[key];
}

std::string Json::dump() const {
    std::string out;
    ::dump(*this, out);
    return out;
}

Json Json::parse(const std::string& text) {
    Parser p{text};
    Json v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}
