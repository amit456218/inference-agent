// Shared glue for the CLI commands and the server: backend construction and prompt prefill.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf.h"
#include "model.h"

std::unique_ptr<Backend> make_backend(const std::string& name, const gguf::File& f, const Model& model, int n_ctx, int n_batch);
// Runs `ids` through the model starting at `pos` in batches; returns the final logits. Advances pos.
const float* prefill(Backend& backend, const std::vector<int32_t>& ids, int& pos, int n_batch);
// Writes the longest prefix of buf that is complete UTF-8 to `out`, keeps the rest in buf.
void take_complete_utf8(std::string& buf, std::string& out);
