/* wLLM exec — internal object definitions backing the opaque C ABI handles.
 * Not installed / not public. */
#ifndef WLLM_EXEC_INTERNAL_H
#define WLLM_EXEC_INTERNAL_H

#include "wllmrt/exec.h"

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

struct wrt_buffer_s {
    wrt_ctx     ctx   = nullptr;
    std::string name;
    void*       dptr  = nullptr;
    size_t      bytes = 0;
    bool        owned = false;   // true if we cudaMalloc'd it (free on destroy)
};

struct wrt_event_s {
    wrt_ctx ctx    = nullptr;    // for stream_id resolution
    void*   handle = nullptr;    // backend event
};

struct wrt_variant {
    void* exec  = nullptr;   // graph-exec handle
    bool  owned = true;      // false if adopted from an external owner (torch)
};

struct wrt_graph_s {
    wrt_ctx     ctx = nullptr;
    std::string name;
    size_t      max_variants = 0;                       // 0 = unbounded
    std::unordered_map<wrt_shape_key, wrt_variant> variants;  // key -> exec
    std::list<wrt_shape_key> lru;                       // front = oldest
    std::unordered_map<std::string, wrt_buffer> bindings;  // port -> buffer (refs)

    void touch(wrt_shape_key key);   // move key to MRU
    void evict_one();                // drop the oldest variant
    void put(wrt_shape_key key, void* exec, bool owned);  // insert/replace + LRU
};

struct wrt_plan_node {
    wrt_graph     graph;
    wrt_shape_key key;
    int           stream_id;
};

struct wrt_plan_s {
    wrt_ctx ctx = nullptr;
    std::vector<wrt_plan_node> nodes;
    std::vector<std::pair<int, int>> deps;  // (node_idx, dep_node_idx)
};

struct wrt_ctx_s {
    std::vector<void*> streams;            // stream_id -> backend stream; [0]=default
    std::vector<char>  stream_owned;       // parallel: 1 if frt created it (destroy), 0 if wrapped
    std::vector<wrt_event_s*> events;      // tracked for cleanup safety
    std::vector<wrt_buffer_s*> buffers;    // ctx owns all buffers (freed at destroy)
    std::vector<wrt_graph_s*> graphs;      // tracked for cleanup safety
    std::vector<wrt_plan_s*>  plans;       // tracked for cleanup safety

    bool has_stream(int id) const {
        return id >= 0 && id < (int)streams.size();
    }
    // Returns the backend stream handle for id; may legitimately be null
    // (handle 0 == the CUDA default stream). Validate the id with has_stream
    // first — do NOT treat a null return as "invalid", since 0 is a real stream.
    void* stream(int id) const {
        if (!has_stream(id)) return nullptr;
        return streams[id];
    }
};

#endif  /* WLLM_EXEC_INTERNAL_H */
