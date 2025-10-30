{
  description = "Python environment for optimization algorithms (Nix-native)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };

    pythonEnv = pkgs.python311.withPackages (ps: [
      ps.numpy
      ps.scipy
      ps.matplotlib
    ]);

  in {
    
    devShells.${system}.default = pkgs.mkShell {
      name = "py-opt-env";

      packages = [
        pythonEnv
      ];

    };
  };
}
