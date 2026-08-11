"""Check H3 image-derived resolution rounding."""

import importlib.util
import os
import sys

import torch

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
sys.path.insert(0, COMFY_DIR)

import comfy.cli_args

comfy.cli_args.args.cpu = True


def main():
    spec = importlib.util.spec_from_file_location(
        "h3mc_resolution_test", os.path.join(PACKAGE_DIR, "__init__.py"),
        submodule_search_locations=[PACKAGE_DIR])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    image = torch.zeros((1, 720, 1280, 3))
    result = package.NODE_CLASS_MAPPINGS[
        "MiniMaxH3ResolutionFromImage"].execute(
            image=image, megapixels=0.5, multiple=32)
    assert result.args == (960, 544)
    print("H3 resolution: image aspect preserved at 0.5 Mpix on 32-pixel grid")


if __name__ == "__main__":
    main()
