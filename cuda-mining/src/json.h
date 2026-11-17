// Tiny JSON reader, enough for Stratum v1 messages. Header-only, no dependencies.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj };
    Type type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    bool is_null() const { return type == Null; }
    bool is_str() const { return type == Str; }
    bool is_num() const { return type == Num; }
    bool is_arr() const { return type == Arr; }
    bool is_true() const { return type == Bool && b; }

    // Out-of-range / missing lookups return a shared Null value instead of throwing.
    const Json& operator[](int i) const { return (type == Arr && i >= 0 && (size_t)i < arr.size()) ? arr[i] : null_value(); }
    const Json& operator[](const char* key) const {
        if (type == Obj)
            for (const auto& kv : obj)
                if (kv.first == key) return kv.second;
        return null_value();
    }

    static bool parse(const std::string& s, Json& out) {
        size_t i = 0;
        if (!value(s, i, out, 0)) return false;
        ws(s, i);
        return i == s.size();
    }

    static std::string escape(const std::string& s) {
        std::string o;
        for (unsigned char c : s) {
            if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
            else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof(buf), "\\u%04x", c); o += buf; }
            else o += (char)c;
        }
        return o;
    }

  private:
    static const Json& null_value() { static const Json n; return n; }
    static void ws(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }
    static bool lit(const std::string& s, size_t& i, const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }
    static void utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 0x3F)); }
        else { o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    }
    static bool string(const std::string& s, size_t& i, std::string& out) {
        if (i >= s.size() || s[i] != '"') return false;
        for (i++; i < s.size(); i++) {
            char c = s[i];
            if (c == '"') { i++; return true; }
            if (c != '\\') { out += c; continue; }
            if (++i >= s.size()) return false;
            switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (i + 4 >= s.size()) return false;
                    utf8(out, (unsigned)strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16));
                    i += 4;
                    break;
                }
                default: out += s[i];
            }
        }
        return false;
    }
    static bool value(const std::string& s, size_t& i, Json& out, int depth) {
        if (depth > 32) return false;
        ws(s, i);
        if (i >= s.size()) return false;
        char c = s[i];
        if (c == '{') {
            out.type = Obj;
            i++; ws(s, i);
            if (i < s.size() && s[i] == '}') { i++; return true; }
            while (true) {
                std::string key;
                Json v;
                ws(s, i);
                if (!string(s, i, key)) return false;
                ws(s, i);
                if (i >= s.size() || s[i++] != ':') return false;
                if (!value(s, i, v, depth + 1)) return false;
                out.obj.emplace_back(std::move(key), std::move(v));
                ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ',') { i++; continue; }
                return s[i++] == '}';
            }
        }
        if (c == '[') {
            out.type = Arr;
            i++; ws(s, i);
            if (i < s.size() && s[i] == ']') { i++; return true; }
            while (true) {
                Json v;
                if (!value(s, i, v, depth + 1)) return false;
                out.arr.push_back(std::move(v));
                ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ',') { i++; continue; }
                return s[i++] == ']';
            }
        }
        if (c == '"') { out.type = Str; return string(s, i, out.str); }
        if (c == 't') { out.type = Bool; out.b = true; return lit(s, i, "true"); }
        if (c == 'f') { out.type = Bool; out.b = false; return lit(s, i, "false"); }
        if (c == 'n') { out.type = Null; return lit(s, i, "null"); }
        const char* start = s.c_str() + i;
        char* end = nullptr;
        out.num = strtod(start, &end);
        if (end == start) return false;
        out.type = Num;
        i += (size_t)(end - start);
        return true;
    }
};
