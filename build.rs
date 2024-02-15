use std::env;

// This is evil ground. Tread carefully.
fn main() {
    let trt_libs = env::var("TENSORRT_LIBRARIES").expect("TENSORRT_LIBRARIES not set");
    let cuda_libs = env::var("CUDA_LIBRARIES").expect("CUDA_LIBRARIES not set");
    let cuda_incl = env::var("CUDA_INCLUDE_DIRS").expect("CUDA_INCLUDE_DIRS not set");

    let fmt = pkg_config::probe_library("fmt").unwrap();
    let spdlog = pkg_config::probe_library("spdlog").unwrap();

    cxx_build::bridge("src/lib.rs")
        .file("src/engine.cpp")
        .include(cuda_incl)
        .includes(spdlog.include_paths)
        .includes(fmt.include_paths)
        .flag_if_supported("-std=c++17")
        .flag("-O3")
        .flag("-Wall")
        .flag("-Werror")
        .define("SPDLOG_FMT_EXTERNAL", Some("1"))
        .compile("libinfer-bridge");

    println!("cargo:rustc-link-search={}", trt_libs);
    println!("cargo:rustc-link-search={}", cuda_libs);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvinfer_plugin");
    println!("cargo:rustc-link-lib=nvonnxparser");

    println!("cargo:rerun-if-changed=src/engine.cpp");
    println!("cargo:rerun-if-changed=include/engine.h");
}
