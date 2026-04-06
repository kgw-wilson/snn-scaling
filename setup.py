from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "simulations.event_driven_cpu_backend",
    ["simulations/event_driven_cpu_backend.cpp"],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="event_driven_cpu_backend",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)