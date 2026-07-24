// Link the wLLM execution-contract C ABI (libwllm_exec).
// Point WLLM_EXEC_LIB_DIR at the directory holding libwllm_exec.so
// (e.g. <repo>/exec/build).
fn main() {
    if let Ok(dir) = std::env::var("WLLM_EXEC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=wllm_exec");
}
