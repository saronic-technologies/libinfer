{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    jetpack-nixos.url = "github:anduril/jetpack-nixos";
  };
  outputs = { nixpkgs, nixpkgs-unstable, flake-utils, jetpack-nixos, ... }:
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

        pkgs-unstable = import nixpkgs-unstable { 
          inherit system; 
          config.allowUnfree = true;
        };

        cudaPackages = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages else pkgs-unstable.cudaPackages;
        tensorrt = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages.tensorrt else pkgs-unstable.cudaPackages.tensorrt;
        l4t-cuda = jetpack-nixos.legacyPackages.aarch64-linux.l4t-cuda;
        inherit (cudaPackages) cudatoolkit cudnn cuda_cudart;

        inputs = [
          pkgs.bacon
          pkgs.cmake
          pkgs.cmake-format
          pkgs.clang
          pkgs.clang-tools
          pkgs.linuxHeaders
          pkgs.llvmPackages.compiler-rt
          pkgs.nixpkgs-fmt
          pkgs.openssl
          pkgs.pkg-config
          cudatoolkit
          tensorrt
          pkgs.rustc
          pkgs.cargo
          pkgs.rustfmt
          pkgs.spdlog
          pkgs.fmt
          pkgs.cxx-rs
        ];

        libs = pkgs.lib.makeLibraryPath [
          pkgs.addDriverRunpath.driverLink
          pkgs.stdenv.cc.cc.lib
          cudaPackages.cudatoolkit
          cudaPackages.cuda_nvrtc
          pkgs.libGL
          pkgs.glib
          pkgs.glibc
          pkgs.zlib
          tensorrt.lib
          cudnn
          cuda_cudart
        ];
      in
      {
        devShells = {
          default = pkgs.mkShell rec {
            nativeBuildInputs = inputs;
            LIBCLANG_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.libclang.lib}/lib/";
            TENSORRT_LIBRARIES = "${tensorrt.lib}/lib";
            CUDA_INCLUDE_DIRS = "${cudatoolkit}/include";
            CUDA_LIBRARIES = "${cudatoolkit}/lib";
            LD_LIBRARY_PATH = libs;
            CPLUS_INCLUDE_PATH = "${pkgs.gcc}/include/c++/${pkgs.gcc.version}:${pkgs.gcc}/include/c++/${pkgs.gcc.version}/x86_64-unknown-linux-gnu:${pkgs.glibc.dev}/include";
            C_INCLUDE_PATH = "${pkgs.glibc.dev}/include";
            shellHook = ''
              export CC="clang"
              export CXX="clang++"
            '';
            
          };
        };
      });
}
