{ pkgs ? import <nixpkgs> {} }:

let

    # pkgs = import (builtins.fetchGit {
    #     name = "nixos-22.11";
    #     url = "https://github.com/nixos/nixpkgs/";
    #     # Choose the branch name
    #     ref = "refs/heads/nixos-22.11";
    #     # Commit hash obtained with
    #     # `git ls-remote https://github.com/nixos/nixpkgs nixos-unstable`
    #     rev = "ce20e9ebe1903ea2ba1ab006ec63093020c761cb";
    # }) {};

    my-python-packages = ps: with ps; [
        numpy
        ipykernel
        jupyterlab
        pytorch
    ];

    my-python = pkgs.python3.withPackages my-python-packages;

in

    pkgs.mkShell {
        buildInputs = [ pkgs.wget my-python ];
    }
