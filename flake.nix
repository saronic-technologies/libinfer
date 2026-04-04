{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
  };
  outputs = { nixpkgs, flake-utils, jetpack-nixos, ... }:
    let
      supported-systems = with flake-utils.lib.system; [
        x86_64-linux
        aarch64-linux
      ];
    in
    flake-utils.lib.eachSystem supported-systems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        cudaPackages = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages else pkgs.cudaPackages;
        tensorrt = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages.tensorrt else pkgs.cudaPackages.tensorrt;
        inherit (cudaPackages) cudatoolkit cudnn cuda_cudart;

        python = pkgs.python3.withPackages (ps: with ps; [
          numpy
          onnx
        ]);

        inputs = with pkgs; [
          bacon
          cmake
          cmake-format
          clang
          clang-tools
          linuxHeaders
          llvmPackages.compiler-rt
          nixpkgs-fmt
          openssl
          pkg-config
          cudatoolkit
          tensorrt
          rustc
          cargo
          rustfmt
          spdlog
          fmt
          cxx-rs
          python
        ];

        libs = pkgs.lib.makeLibraryPath (with pkgs; [
          addDriverRunpath.driverLink
          stdenv.cc.cc.lib
          cudaPackages.cudatoolkit
          cudaPackages.cuda_nvrtc
          libGL
          glib
          zlib
          tensorrt.lib
          cudnn
          cuda_cudart
          spdlog
          fmt
        ]);

        includes = pkgs.lib.makeSearchPath "include" [
          pkgs.glibc.dev
        ];
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = inputs;
            LIBCLANG_PATH = "${pkgs.libclang.lib}/lib/";
            TENSORRT_LIBRARIES = "${tensorrt.lib}/lib";
            CUDA_INCLUDE_DIRS = "${cudatoolkit}/include";
            CUDA_LIBRARIES = "${cudatoolkit}/lib";
            LD_LIBRARY_PATH = libs;
            LIBRARY_PATH = libs;
            CPLUS_INCLUDE_PATH = "${pkgs.gcc}/include/c++/${pkgs.gcc.version}:${pkgs.gcc}/include/c++/${pkgs.gcc.version}/${pkgs.stdenv.hostPlatform.config}:${includes}";
            C_INCLUDE_PATH = includes;
            shellHook = ''
              export CC="clang"
              export CXX="clang++"
            '';

          };
        };
      });
}
