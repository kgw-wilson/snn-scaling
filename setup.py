import os
import subprocess
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_EXTRA_COMPILE_ARGS = ["-O3", "-march=native"]
_TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), "lib")


def _torch_openmp_library() -> str:
    """Gets the path of the OpenMP runtime that PyTorch loads

    A second runtime (e.g. from Homebrew or the system) in the same process
    means two competing thread pools.

    macOS: Apple clang rejects a plain -fopenmp, so -Xpreprocessor passes it straight
    to the compiler. That skips the usual link step, so the extension is linked
    explicitly against PyTorch's bundled libomp.dylib (omp.h comes from PyTorch's
    include directory, which CppExtension already adds).

    Linux: Finds the exact libgomp that libtorch_cpu.so loads (found with ldd).

    Raises RuntimeError rather than falling back to a different runtime if PyTorch's
    OpenMP library can't be found.
    """

    if sys.platform == "darwin":
        path = os.path.join(_TORCH_LIB_DIR, "libomp.dylib")
        if not os.path.exists(path):
            raise RuntimeError(f"PyTorch's OpenMP runtime not found at {path}")
        return path

    libtorch_cpu = os.path.join(_TORCH_LIB_DIR, "libtorch_cpu.so")
    ldd = subprocess.run(
        ["ldd", libtorch_cpu], capture_output=True, text=True, check=True
    )
    for line in ldd.stdout.splitlines():
        name, _, location = line.partition("=>")
        if "libgomp" in name and location:
            path = location.split("(")[0].strip()
            if os.path.exists(path):
                return path
    raise RuntimeError(f"Could not find the libgomp that {libtorch_cpu} links against.")


ext_modules = [
    CppExtension(
        "simulations.clock_driven_dense.backend",
        ["simulations/clock_driven_dense/backend.cpp"],
        extra_compile_args=_EXTRA_COMPILE_ARGS,
    ),
    CppExtension(
        "simulations.clock_driven_sparse.backend",
        ["simulations/clock_driven_sparse/backend.cpp"],
        extra_compile_args=_EXTRA_COMPILE_ARGS,
    ),
    CppExtension(
        "simulations.clock_driven_openmp.backend",
        ["simulations/clock_driven_openmp/backend.cpp"],
        extra_compile_args=_EXTRA_COMPILE_ARGS
        + (
            ["-Xpreprocessor", "-fopenmp"] if sys.platform == "darwin" else ["-fopenmp"]
        ),
        extra_link_args=[_torch_openmp_library()],
    ),
    CppExtension(
        "simulations.event_driven.backend",
        ["simulations/event_driven/backend.cpp"],
        extra_compile_args=_EXTRA_COMPILE_ARGS,
    ),
]

setup(
    name="simulation_backends",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
