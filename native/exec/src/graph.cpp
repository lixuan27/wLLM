/* wLLM exec — Graph: a ShapeKey -> graph-exec variant table with LRU. */
#include "internal.h"
#include "backend.h"

void wrt_graph_s::touch(wrt_shape_key key) {
    for (auto it = lru.begin(); it != lru.end(); ++it) {
        if (*it == key) { lru.erase(it); break; }
    }
    lru.push_back(key);  // back = most recently used
}

void wrt_graph_s::evict_one() {
    if (lru.empty()) return;
    wrt_shape_key old = lru.front();
    lru.pop_front();
    auto it = variants.find(old);
    if (it != variants.end()) {
        if (it->second.owned)                         // never free an adopted exec
            frt::be::graph_exec_destroy(it->second.exec);
        variants.erase(it);
    }
}

void wrt_graph_s::put(wrt_shape_key key, void* exec, bool owned) {
    auto it = variants.find(key);
    if (it != variants.end()) {
        if (it->second.owned) frt::be::graph_exec_destroy(it->second.exec);
        it->second = wrt_variant{exec, owned};
    } else {
        variants.emplace(key, wrt_variant{exec, owned});
    }
    touch(key);
    if (max_variants > 0 && variants.size() > max_variants) evict_one();
}

wrt_graph wrt_graph_create(wrt_ctx c, const char* name, size_t max_variants) {
    if (!c) return nullptr;
    auto* g = new wrt_graph_s();
    g->ctx = c;
    g->name = name ? name : "";
    g->max_variants = max_variants;
    c->graphs.push_back(g);
    return g;
}

void wrt_graph_destroy(wrt_graph g) {
    if (!g) return;
    auto& gs = g->ctx->graphs;
    for (auto it = gs.begin(); it != gs.end(); ++it) {
        if (*it == g) { gs.erase(it); break; }
    }
    for (auto& kv : g->variants)
        if (kv.second.owned) frt::be::graph_exec_destroy(kv.second.exec);
    delete g;
}

int wrt_graph_capture(wrt_graph g, wrt_shape_key key,
                      void (*record)(void*, void*), void* user) {
    if (!g || !record) return WRT_ERR_INVALID;
    void* cap_stream = g->ctx->stream(0);  // capture on the default stream
    if (!cap_stream) return WRT_ERR_INVALID;

    if (!frt::be::capture_begin(cap_stream)) return WRT_ERR_CAPTURE;
    record(user, cap_stream);  // model enqueues its kernels onto cap_stream
    void* exec = frt::be::capture_end(cap_stream);
    if (!exec) return WRT_ERR_CAPTURE;
    g->put(key, exec, /*owned=*/true);
    return WRT_OK;
}

int wrt_graph_adopt(wrt_graph g, wrt_shape_key key, void* external_graph_exec) {
    if (!g || !external_graph_exec) return WRT_ERR_INVALID;
    g->put(key, external_graph_exec, /*owned=*/false);  // never freed by frt
    return WRT_OK;
}

int wrt_graph_evict(wrt_graph g, wrt_shape_key key) {
    if (!g) return WRT_ERR_INVALID;
    auto it = g->variants.find(key);
    if (it == g->variants.end()) return WRT_ERR_NO_VARIANT;
    if (it->second.owned) frt::be::graph_exec_destroy(it->second.exec);
    g->variants.erase(it);
    for (auto lit = g->lru.begin(); lit != g->lru.end(); ++lit)
        if (*lit == key) { g->lru.erase(lit); break; }
    return WRT_OK;
}

int wrt_graph_evict_lru(wrt_graph g) {
    if (!g) return WRT_ERR_INVALID;
    if (g->lru.empty()) return WRT_ERR_NO_VARIANT;
    g->evict_one();
    return WRT_OK;
}

size_t wrt_graph_variant_count(wrt_graph g) {
    return g ? g->variants.size() : 0;
}

int wrt_graph_bind(wrt_graph g, const char* port, wrt_buffer b) {
    if (!g || !port || !b) return WRT_ERR_INVALID;
    g->bindings[port] = b;  // bookkeeping + lifetime ref; pointers were baked at capture
    return WRT_OK;
}

int wrt_graph_replay(wrt_graph g, wrt_shape_key key, int stream_id) {
    if (!g) return WRT_ERR_INVALID;
    auto it = g->variants.find(key);
    if (it == g->variants.end()) return WRT_ERR_NO_VARIANT;  // never a silent no-op
    if (!g->ctx->has_stream(stream_id)) return WRT_ERR_INVALID;
    g->touch(key);
    return frt::be::graph_launch(it->second.exec, g->ctx->stream(stream_id))
           ? WRT_OK : WRT_ERR_BACKEND;
}

int wrt_graph_has_variant(wrt_graph g, wrt_shape_key key) {
    if (!g) return 0;
    return g->variants.count(key) ? 1 : 0;
}
