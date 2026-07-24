/* wLLM exec — pybind module (`_wllm_exec`).
 *
 * Dev / research / migration only. The hot path for real deployment is the C
 * ABI linked directly by C++/Rust/robot hosts. Here we wrap the same C ABI in
 * thin Python-friendly classes; the Ctx owns everything and frees it on
 * destruction, so Buffer/Graph/Plan/Event Python objects do not self-destroy
 * (avoids GC-order use-after-free).
 */
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include "wllmrt/exec.h"
#include "backend.h"

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

void check(int rc, const char* what) {
    if (rc < 0) throw std::runtime_error(std::string(what) + " failed: rc=" + std::to_string(rc));
}

// Trampoline: the C record callback forwards to a Python callable, passing the
// capture stream as an integer the caller can wrap (torch.cuda.ExternalStream).
void py_record_trampoline(void* user, void* stream) {
    auto* fn = reinterpret_cast<py::function*>(user);
    (*fn)(reinterpret_cast<std::uintptr_t>(stream));
}

struct PyBuffer { wrt_buffer h; };
struct PyEvent  { wrt_event  h; };
struct PyGraph  { wrt_graph  h; };
struct PyPlan   { wrt_plan   h; };

struct PyCtx {
    wrt_ctx h;
    PyCtx() { h = wrt_ctx_create(); if (!h) throw std::runtime_error("wrt_ctx_create failed"); }
    ~PyCtx() { if (h) wrt_ctx_destroy(h); }
};

}  // namespace

PYBIND11_MODULE(_wllm_exec, m) {
    m.doc() = "wLLM execution-contract C ABI (dev binding)";

    py::class_<PyBuffer>(m, "Buffer")
        .def("dptr",  [](PyBuffer& b) { return reinterpret_cast<std::uintptr_t>(wrt_buffer_dptr(b.h)); })
        .def("nbytes",[](PyBuffer& b) { return wrt_buffer_bytes(b.h); })
        .def("name",  [](PyBuffer& b) { return std::string(wrt_buffer_name(b.h)); })
        .def("raw",   [](PyBuffer& b) { return reinterpret_cast<std::uintptr_t>(b.h); },
             "Opaque wrt_buffer handle (uintptr) — for the runtime-export builder.");

    py::class_<PyEvent>(m, "Event")
        .def("record", [](PyEvent& e, int stream_id) { check(wrt_event_record(e.h, stream_id), "event_record"); });

    py::class_<PyGraph>(m, "Graph")
        .def("capture", [](PyGraph& g, std::uint64_t key, py::function record) {
            check(wrt_graph_capture(g.h, key, &py_record_trampoline, &record), "graph_capture");
        }, py::arg("key"), py::arg("record"))
        .def("adopt", [](PyGraph& g, std::uint64_t key, std::uintptr_t graph_exec) {
            check(wrt_graph_adopt(g.h, key, reinterpret_cast<void*>(graph_exec)), "graph_adopt");
        }, py::arg("key"), py::arg("graph_exec"),
           "Register an external graph-exec (e.g. torch CUDAGraph.raw_cuda_graph_exec()).")
        .def("bind", [](PyGraph& g, const std::string& port, PyBuffer& b) {
            check(wrt_graph_bind(g.h, port.c_str(), b.h), "graph_bind");
        })
        .def("replay", [](PyGraph& g, std::uint64_t key, int stream_id) {
            return wrt_graph_replay(g.h, key, stream_id);  // return rc (e.g. NO_VARIANT) to caller
        }, py::arg("key"), py::arg("stream_id") = 0)
        .def("has_variant", [](PyGraph& g, std::uint64_t key) { return wrt_graph_has_variant(g.h, key) != 0; })
        .def("evict", [](PyGraph& g, std::uint64_t key) { return wrt_graph_evict(g.h, key); },
             "Drop one variant (host eviction policy; evict at a safe point only).")
        .def("evict_lru", [](PyGraph& g) { return wrt_graph_evict_lru(g.h); })
        .def("variant_count", [](PyGraph& g) { return wrt_graph_variant_count(g.h); })
        .def("raw", [](PyGraph& g) { return reinterpret_cast<std::uintptr_t>(g.h); },
             "Opaque wrt_graph handle (uintptr) — for the runtime-export builder.");

    py::class_<PyPlan>(m, "Plan")
        .def("add", [](PyPlan& p, PyGraph& g, std::uint64_t key, int stream_id) {
            int idx = wrt_plan_add(p.h, g.h, key, stream_id);
            check(idx, "plan_add"); return idx;
        }, py::arg("graph"), py::arg("key"), py::arg("stream_id") = 0)
        .def("after", [](PyPlan& p, int node_idx, int dep_idx) { check(wrt_plan_after(p.h, node_idx, dep_idx), "plan_after"); })
        .def("execute", [](PyPlan& p, std::uint64_t key) { check(wrt_plan_execute(p.h, key), "plan_execute"); }, py::arg("key") = (std::uint64_t)WRT_KEY_INHERIT)
        .def("sync", [](PyPlan& p) { check(wrt_plan_sync(p.h), "plan_sync"); });

    py::class_<PyCtx>(m, "Ctx")
        .def(py::init<>())
        .def("stream", [](PyCtx& c, int priority) { int id = wrt_ctx_stream(c.h, priority); check(id, "ctx_stream"); return id; }, py::arg("priority") = 0)
        .def("wrap_stream", [](PyCtx& c, std::uintptr_t external_stream) {
            int id = wrt_ctx_wrap_stream(c.h, reinterpret_cast<void*>(external_stream));
            check(id, "ctx_wrap_stream"); return id;
        }, py::arg("external_stream"), "Wrap an external stream (e.g. torch stream cuda handle) as a stream_id.")
        .def("event", [](PyCtx& c) { PyEvent e; e.h = wrt_ctx_event(c.h); if (!e.h) throw std::runtime_error("ctx_event failed"); return e; })
        .def("stream_wait", [](PyCtx& c, int stream_id, PyEvent& e) { check(wrt_stream_wait(c.h, stream_id, e.h), "stream_wait"); })
        .def("buffer", [](PyCtx& c, const std::string& name, size_t nbytes) {
            PyBuffer b; b.h = wrt_buffer_alloc(c.h, name.c_str(), nbytes);
            if (!b.h) throw std::runtime_error("buffer_alloc failed"); return b;
        })
        .def("wrap", [](PyCtx& c, const std::string& name, std::uintptr_t dptr, size_t nbytes) {
            PyBuffer b; b.h = wrt_buffer_wrap(c.h, name.c_str(), reinterpret_cast<void*>(dptr), nbytes);
            if (!b.h) throw std::runtime_error("buffer_wrap failed"); return b;
        })
        .def("copy", [](PyCtx& c, PyBuffer& dst, size_t dst_off, PyBuffer& src, size_t src_off, size_t nbytes, int stream_id) {
            check(wrt_buffer_copy(c.h, dst.h, dst_off, src.h, src_off, nbytes, stream_id), "buffer_copy");
        }, py::arg("dst"), py::arg("dst_off"), py::arg("src"), py::arg("src_off"), py::arg("nbytes"), py::arg("stream_id") = 0)
        .def("graph", [](PyCtx& c, const std::string& name, size_t max_variants) {
            PyGraph g; g.h = wrt_graph_create(c.h, name.c_str(), max_variants);
            if (!g.h) throw std::runtime_error("graph_create failed"); return g;
        }, py::arg("name"), py::arg("max_variants") = 0)
        .def("plan", [](PyCtx& c) { PyPlan p; p.h = wrt_plan_create(c.h); if (!p.h) throw std::runtime_error("plan_create failed"); return p; })
        .def("raw", [](PyCtx& c) { return reinterpret_cast<std::uintptr_t>(c.h); },
             "Opaque wrt_ctx handle (uintptr) — for the runtime-export builder.");

    // --- dev/test helpers: allocation-free, capture-safe ops on a raw stream
    //     (an integer cudaStream_t). Used by record callbacks in tests so we
    //     can validate the contract without a real model kernel. ---
    m.def("memset_async", [](std::uintptr_t dptr, int value, size_t nbytes, std::uintptr_t stream) {
        if (!frt::be::memset_async(reinterpret_cast<void*>(dptr), value, nbytes, reinterpret_cast<void*>(stream)))
            throw std::runtime_error("memset_async failed");
    });
    m.def("memcpy_async", [](std::uintptr_t dst, std::uintptr_t src, size_t nbytes, std::uintptr_t stream) {
        if (!frt::be::memcpy_dtod_async(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), nbytes, reinterpret_cast<void*>(stream)))
            throw std::runtime_error("memcpy_async failed");
    });
}
