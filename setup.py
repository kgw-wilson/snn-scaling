from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

_EXTRA_COMPILE_ARGS = ["-O3", "-march=native"]

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
        extra_compile_args=_EXTRA_COMPILE_ARGS,
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
