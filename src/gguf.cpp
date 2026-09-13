#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace gguf {

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" read as little-endian u32

struct TypeTraits {
    const char* name;
    size_t block;
    size_t bytes;
};

const TypeTraits* traits(TensorType t) {
    static const TypeTraits F32{"F32", 1, 4}, F16{"F16", 1, 2}, BF16{"BF16", 1, 2},
        Q4_0{"Q4_0", 32, 18}, Q4_1{"Q4_1", 32, 20}, Q5_0{"Q5_0", 32, 22}, Q5_1{"Q5_1", 32, 24},
        Q8_0{"Q8_0", 32, 34}, Q8_1{"Q8_1", 32, 36}, Q2_K{"Q2_K", 256, 84}, Q3_K{"Q3_K", 256, 110},
        Q4_K{"Q4_K", 256, 144}, Q5_K{"Q5_K", 256, 176}, Q6_K{"Q6_K", 256, 210}, Q8_K{"Q8_K", 256, 292};
    switch (t) {
        case TensorType::F32: return &F32;
        case TensorType::F16: return &F16;
        case TensorType::BF16: return &BF16;
        case TensorType::Q4_0: return &Q4_0;
        case TensorType::Q4_1: return &Q4_1;
        case TensorType::Q5_0: return &Q5_0;
        case TensorType::Q5_1: return &Q5_1;
        case TensorType::Q8_0: return &Q8_0;
        case TensorType::Q8_1: return &Q8_1;
        case TensorType::Q2_K: return &Q2_K;
        case TensorType::Q3_K: return &Q3_K;
        case TensorType::Q4_K: return &Q4_K;
        case TensorType::Q5_K: return &Q5_K;
        case TensorType::Q6_K: return &Q6_K;
        case TensorType::Q8_K: return &Q8_K;
    }
    return nullptr;
}

const TypeTraits& traits_or_throw(TensorType t) {
    const TypeTraits* tt = traits(t);
    if (!tt) throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(uint32_t(t)));
    return *tt;
}

// Bounds-checked reader over the mapped file.
struct Cursor {
    const uint8_t* base;
    size_t size;
    size_t pos = 0;

    void need(size_t n) const {
        if (n > size - pos) throw std::runtime_error("gguf: file truncated while parsing header");
    }
    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }
    std::string read_string() {
        uint64_t len = read<uint64_t>();
        need(len);
        std::string s(reinterpret_cast<const char*>(base + pos), len);
        pos += len;
        return s;
    }
};

void read_scalar_into(Cursor& c, ValueType t, int64_t& i, double& f, std::string& s) {
    switch (t) {
        case ValueType::UINT8:   i = c.read<uint8_t>(); f = double(i); break;
        case ValueType::INT8:    i = c.read<int8_t>(); f = double(i); break;
        case ValueType::UINT16:  i = c.read<uint16_t>(); f = double(i); break;
        case ValueType::INT16:   i = c.read<int16_t>(); f = double(i); break;
        case ValueType::UINT32:  i = c.read<uint32_t>(); f = double(i); break;
        case ValueType::INT32:   i = c.read<int32_t>(); f = double(i); break;
        case ValueType::FLOAT32: f = c.read<float>(); i = int64_t(f); break;
        case ValueType::BOOL:    i = c.read<uint8_t>() != 0; f = double(i); break;
        case ValueType::STRING:  s = c.read_string(); break;
        case ValueType::UINT64:  i = int64_t(c.read<uint64_t>()); f = double(i); break;
        case ValueType::INT64:   i = c.read<int64_t>(); f = double(i); break;
        case ValueType::FLOAT64: f = c.read<double>(); i = int64_t(f); break;
        case ValueType::ARRAY:   throw std::runtime_error("gguf: nested arrays are not supported");
    }
}

bool is_string_type(ValueType t) { return t == ValueType::STRING; }
bool is_float_type(ValueType t) { return t == ValueType::FLOAT32 || t == ValueType::FLOAT64; }

Value read_value(Cursor& c) {
    Value v;
    v.type = ValueType(c.read<uint32_t>());
    if (v.type != ValueType::ARRAY) {
        read_scalar_into(c, v.type, v.i, v.f, v.s);
        return v;
    }
    v.elem_type = ValueType(c.read<uint32_t>());
    uint64_t n = c.read<uint64_t>();
    if (v.elem_type == ValueType::ARRAY) throw std::runtime_error("gguf: nested arrays are not supported");
    if (n > c.size) throw std::runtime_error("gguf: array length larger than file");
    if (is_string_type(v.elem_type)) v.arr_s.reserve(n);
    else if (is_float_type(v.elem_type)) v.arr_f.reserve(n);
    else v.arr_i.reserve(n);
    for (uint64_t k = 0; k < n; k++) {
        int64_t i = 0; double f = 0; std::string s;
        read_scalar_into(c, v.elem_type, i, f, s);
        if (is_string_type(v.elem_type)) v.arr_s.push_back(std::move(s));
        else if (is_float_type(v.elem_type)) v.arr_f.push_back(f);
        else v.arr_i.push_back(i);
    }
    return v;
}

}  // namespace

const char* type_name(TensorType t) {
    const TypeTraits* tt = traits(t);
    return tt ? tt->name : "unknown";
}
size_t type_block_size(TensorType t) { return traits_or_throw(t).block; }
size_t type_block_bytes(TensorType t) { return traits_or_throw(t).bytes; }

size_t Value::array_size() const {
    return arr_i.size() + arr_f.size() + arr_s.size();
}

uint64_t TensorInfo::nelements() const {
    uint64_t n = 1;
    for (uint64_t d : ne) n *= d;
    return n;
}

size_t TensorInfo::nbytes() const {
    const TypeTraits& tt = traits_or_throw(type);
    uint64_t n = nelements();
    if (n % tt.block != 0)
        throw std::runtime_error("gguf: tensor " + name + " size not a multiple of its block size");
    return size_t(n / tt.block * tt.bytes);
}

File::~File() { close(); }

File::File(File&& o) noexcept { *this = std::move(o); }

File& File::operator=(File&& o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_);
        fd_ = o.fd_; o.fd_ = -1;
        map_ = o.map_; o.map_ = nullptr;
        map_size_ = o.map_size_; o.map_size_ = 0;
        version_ = o.version_;
        data_offset_ = o.data_offset_;
        kv_ = std::move(o.kv_);
        tensors_ = std::move(o.tensors_);
        index_ = std::move(o.index_);
    }
    return *this;
}

void File::close() {
    if (map_) { munmap(map_, map_size_); map_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

File File::open(const std::string& path) {
    File f;
    f.path_ = path;
    f.fd_ = ::open(path.c_str(), O_RDONLY);
    if (f.fd_ < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st{};
    if (fstat(f.fd_, &st) != 0 || st.st_size <= 0) throw std::runtime_error("gguf: cannot stat " + path);
    f.map_size_ = size_t(st.st_size);
    f.map_ = mmap(nullptr, f.map_size_, PROT_READ, MAP_SHARED, f.fd_, 0);
    if (f.map_ == MAP_FAILED) { f.map_ = nullptr; throw std::runtime_error("gguf: mmap failed for " + path); }
    f.parse();
    return f;
}

void File::parse() {
    Cursor c{static_cast<const uint8_t*>(map_), map_size_};
    if (c.read<uint32_t>() != kMagic) throw std::runtime_error("gguf: bad magic, not a GGUF file: " + path_);
    version_ = c.read<uint32_t>();
    if (version_ < 2 || version_ > 3)
        throw std::runtime_error("gguf: unsupported version " + std::to_string(version_));
    uint64_t n_tensors = c.read<uint64_t>();
    uint64_t n_kv = c.read<uint64_t>();
    if (n_tensors > (1u << 20) || n_kv > (1u << 20)) throw std::runtime_error("gguf: implausible header counts");

    for (uint64_t k = 0; k < n_kv; k++) {
        std::string key = c.read_string();
        kv_[key] = read_value(c);
    }

    tensors_.reserve(n_tensors);
    for (uint64_t k = 0; k < n_tensors; k++) {
        TensorInfo t;
        t.name = c.read_string();
        uint32_t n_dims = c.read<uint32_t>();
        if (n_dims > 4) throw std::runtime_error("gguf: tensor " + t.name + " has " + std::to_string(n_dims) + " dims");
        for (uint32_t d = 0; d < n_dims; d++) t.ne.push_back(c.read<uint64_t>());
        t.type = TensorType(c.read<uint32_t>());
        t.offset = c.read<uint64_t>();
        index_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    size_t alignment = size_t(get_int("general.alignment", 32));
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::runtime_error("gguf: alignment must be a power of two");
    data_offset_ = (c.pos + alignment - 1) / alignment * alignment;

    for (TensorInfo& t : tensors_) {
        size_t nbytes = t.nbytes();
        if (t.offset > map_size_ || data_offset_ + t.offset + nbytes > map_size_)
            throw std::runtime_error("gguf: tensor " + t.name + " extends past end of file");
        t.data = static_cast<const uint8_t*>(map_) + data_offset_ + t.offset;
    }
}

bool File::has(const std::string& key) const { return kv_.count(key) != 0; }

const Value& File::get(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing metadata key " + key);
    return it->second;
}

int64_t File::get_int(const std::string& key) const { return get(key).i; }
int64_t File::get_int(const std::string& key, int64_t def) const { return has(key) ? get(key).i : def; }
double File::get_float(const std::string& key) const { return get(key).f; }
double File::get_float(const std::string& key, double def) const { return has(key) ? get(key).f : def; }
const std::string& File::get_str(const std::string& key) const { return get(key).s; }
std::string File::get_str(const std::string& key, const std::string& def) const { return has(key) ? get(key).s : def; }

const TensorInfo* File::find_tensor(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& File::tensor(const std::string& name) const {
    const TensorInfo* t = find_tensor(name);
    if (!t) throw std::runtime_error("gguf: missing tensor " + name);
    return *t;
}

}  // namespace gguf
