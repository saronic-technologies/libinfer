{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
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
        pkgs = import nixpkgs { inherit system; };
        cudaPackages = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages else pkgs.cudaPackages;
        tensorrt = if system == "aarch64-linux" then jetpack-nixos.legacyPackages.aarch64-linux.cudaPackages.tensorrt else pkgs.cudaPackages.tensorrt;
        l4t-cuda = jetpack-nixos.legacyPackages.aarch64-linux.l4t-cuda;
        inherit (cudaPackages) cudatoolkit cudnn cuda_cudart;

        inputs = with pkgs; [
          bacon
          cmake
          cmake-format
          clang
          clang-tools
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
        ];
      in
      {
        devShells = {
          default = pkgs.mkShell {
            nativeBuildInputs = inputs;
            LIBCLANG_PATH = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.libclang.lib}/lib/";
            TENSORRT_LIBRARIES = "${tensorrt.lib}/lib";
            CUDA_INCLUDE_DIRS = "${cudatoolkit}/include";
            CUDA_LIBRARIES = "${cudatoolkit}/lib";
            LD_LIBRARY_PATH = "${tensorrt.lib}/lib";
          };
        };
      });
}
