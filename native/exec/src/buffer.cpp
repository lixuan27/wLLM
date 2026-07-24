/* wLLM exec — buffers (the only "state" primitive). */
#include "internal.h"
#include "backend.h"

wrt_buffer wrt_buffer_alloc(wrt_ctx c, const char* name, size_t bytes) {
    if (!c || bytes == 0) return nullptr;
    void* d = frt::be::malloc(bytes);
    if (!d) return nullptr;
    auto* b = new wrt_buffer_s();
    b->ctx = c;
    b->name = name ? name : "";
    b->dptr = d;
    b->bytes = bytes;
    b->owned = true;
    c->buffers.push_back(b);
    return b;
}

wrt_buffer wrt_buffer_wrap(wrt_ctx c, const char* name, void* dptr, size_t bytes) {
    if (!c || !dptr) return nullptr;
    auto* b = new wrt_buffer_s();
    b->ctx = c;
    b->name = name ? name : "";
    b->dptr = dptr;
    b->bytes = bytes;
    b->owned = false;  // external pointer; never freed by us
    c->buffers.push_back(b);
    return b;
}

void* wrt_buffer_dptr(wrt_buffer b)  { return b ? b->dptr : nullptr; }
size_t wrt_buffer_bytes(wrt_buffer b) { return b ? b->bytes : 0; }
const char* wrt_buffer_name(wrt_buffer b) { return b ? b->name.c_str() : ""; }

int wrt_buffer_copy(wrt_ctx c, wrt_buffer dst, size_t dst_off,
                    wrt_buffer src, size_t src_off, size_t bytes, int stream_id) {
    if (!c || !dst || !src) return WRT_ERR_INVALID;
    if (dst_off + bytes > dst->bytes || src_off + bytes > src->bytes)
        return WRT_ERR_INVALID;
    if (!c->has_stream(stream_id)) return WRT_ERR_INVALID;
    void* s = c->stream(stream_id);
    void* d = static_cast<char*>(dst->dptr) + dst_off;
    const void* sp = static_cast<const char*>(src->dptr) + src_off;
    return frt::be::memcpy_dtod_async(d, sp, bytes, s) ? WRT_OK : WRT_ERR_BACKEND;
}

// Note: no wrt_buffer_destroy in the public ABI yet (Phase A) — the ctx owns
// all buffers and frees owned device memory at wrt_ctx_destroy. Add per-buffer
// destroy when a real model needs finer lifetime than the ctx scope.
