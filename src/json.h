// Minimal JSON value, parser and serializer. Enough for OpenAI-style request/response bodies.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

class Json {
public:
    enum Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };

    Json() = default;
    Json(bool b) : type_(BOOL), b_(b) {}
    Json(int v) : type_(NUMBER), n_(v) {}
    Json(int64_t v) : type_(NUMBER), n_(double(v)) {}
    Json(double v) : type_(NUMBER), n_(v) {}
    Json(const char* s) : type_(STRING), s_(s) {}
    Json(std::string s) : type_(STRING), s_(std::move(s)) {}
    static Json array() { Json j; j.type_ = ARRAY; return j; }
    static Json object() { Json j; j.type_ = OBJECT; return j; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == NUL; }
    bool is_string() const { return type_ == STRING; }
    bool is_number() const { return type_ == NUMBER; }
    bool is_array() const { return type_ == ARRAY; }
    bool is_object() const { return type_ == OBJECT; }

    bool as_bool(bool def = false) const { return type_ == BOOL ? b_ : def; }
    double as_number(double def = 0) const { return type_ == NUMBER ? n_ : def; }
    const std::string& as_string() const { return s_; }
    const std::vector<Json>& items() const { return a_; }
    const std::map<std::string, Json>& fields() const { return o_; }

    // Object access; a missing key yields a null Json.
    const Json& operator[](const std::string& key) const;
    Json& operator[](const std::string& key);
    bool has(const std::string& key) const { return type_ == OBJECT && o_.count(key); }
    void push(Json v) { type_ = ARRAY; a_.push_back(std::move(v)); }

    std::string dump() const;
    // Throws std::runtime_error on malformed input.
    static Json parse(const std::string& text);

private:
    Type type_ = NUL;
    bool b_ = false;
    double n_ = 0;
    std::string s_;
    std::vector<Json> a_;
    std::map<std::string, Json> o_;
};
