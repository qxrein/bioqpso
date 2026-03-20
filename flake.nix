{
  description = "Python environment for optimization algorithms (Nix-native)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "aarch64-darwin";
    pkgs = import nixpkgs { inherit system; };

    pythonEnv = pkgs.python311.withPackages (ps: [
      ps.numpy
      ps.scipy
      ps.matplotlib
      ps.scikit-learn
    ]);

  in {

    devShells.${system}.default = pkgs.mkShell {
      name = "py-opt-env";

      packages = [
        pythonEnv
        pkgs.python311Packages.venvShellHook
      ];

      shellHook = ''
        # Create and activate a local virtual environment with pip available
        if [ ! -d ".venv" ]; then
          python -m venv .venv
        fi
        . .venv/bin/activate
        python -m pip install --upgrade pip
      '';
    };
  };
}
