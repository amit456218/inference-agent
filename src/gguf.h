// GGUF file format reader.
//
// GGUF is the single-file model format used by llama.cpp. Layout:
//
//   magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
//   n_kv    x ( key string | value_type u32 | value )
//   n_tensors x ( name string | n_dims u32 | dims u64[n_dims] | ggml_type u32 | offset u64 )
//   padding to general.alignment (default 32)
//   tensor data (each tensor at data_start + offset)
//
// Strings are u64 length followed by UTF-8 bytes, no terminator. Everything is little-endian.
// Tensor dims are in ggml order: ne[0] is the contiguous/fastest dimension. A weight matrix
// used as y = W x with n_in inputs and n_out outputs is stored with ne = [n_in, n_out].
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gguf {

enum class ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

// Tensor element types. Numeric values match ggml's enum so files parse directly.
enum class TensorType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15, BF16 = 30,
};

const char* type_name(TensorType t);
size_t type_block_size(TensorType t);   // elements per quantization block (1 for float types)
size_t type_block_bytes(TensorType t);  // bytes per block

struct Value {
    ValueType type = ValueType::UINT8;
    int64_t i = 0;       // integer and bool scalars
    double f = 0;        // float scalars
    std::string s;       // string scalar
    ValueType elem_type = ValueType::UINT8;  // arrays: element type, data in exactly one of:
    std::vector<int64_t> arr_i;
    std::vector<double> arr_f;
    std::vector<std::string> arr_s;
    size_t array_size() const;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;   // ggml dims, ne[0] fastest
    TensorType type = TensorType::F32;
    uint64_t offset = 0;        // relative to the data section
    const uint8_t* data = nullptr;
    uint64_t nelements() const;
    size_t nbytes() const;
    uint64_t dim(size_t i) const { return i < ne.size() ? ne[i] : 1; }
};

class File {
public:
    File() = default;
    ~File();
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& o) noexcept;
    File& operator=(File&& o) noexcept;

    // Opens and mmaps the file, parses header, metadata and tensor table.
    // Throws std::runtime_error on any problem.
    static File open(const std::string& path);

    uint32_t version() const { return version_; }
    size_t file_size() const { return map_size_; }
    size_t data_offset() const { return data_offset_; }
    const uint8_t* base() const { return static_cast<const uint8_t*>(map_); }  // start of the mmap'd file
    const std::string& path() const { return path_; }

    const std::unordered_map<std::string, Value>& metadata() const { return kv_; }
    bool has(const std::string& key) const;
    const Value& get(const std::string& key) const;  // throws if missing
    int64_t get_int(const std::string& key) const;
    int64_t get_int(const std::string& key, int64_t def) const;
    double get_float(const std::string& key) const;
    double get_float(const std::string& key, double def) const;
    const std::string& get_str(const std::string& key) const;
    std::string get_str(const std::string& key, const std::string& def) const;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find_tensor(const std::string& name) const;
    const TensorInfo& tensor(const std::string& name) const;  // throws if missing

private:
    void parse();
    void close();

    std::string path_;
    int fd_ = -1;
    void* map_ = nullptr;
    size_t map_size_ = 0;
    uint32_t version_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, Value> kv_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

}  // namespace gguf
