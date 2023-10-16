import contextlib
import logging
import multiprocessing
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from typing import ClassVar

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.develop

PYTHON_DIR = os.path.realpath(os.path.dirname(__file__))
TOP_DIR = os.path.dirname(os.path.dirname(PYTHON_DIR))
CMAKE_BUILD_DIR = os.path.join(PYTHON_DIR, ".setuptools-cmake-build")

WINDOWS = os.name == "nt"

CMAKE = shutil.which("cmake3") or shutil.which("cmake")

################################################################################
# Global variables for controlling the build variant
################################################################################

DEBUG = os.getenv("DEBUG", "0") == "1"

# Customize the wheel plat-name, usually needed for MacOS builds.
# See usage in .github/workflows/release_mac.yml
FR_WHEEL_PLATFORM_NAME = os.getenv("FR_WHEEL_PLATFORM_NAME")

################################################################################
# Pre Check
################################################################################

assert CMAKE, "Could not find cmake in PATH"

################################################################################
# Utilities
################################################################################


@contextlib.contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f"Can only cd to absolute path, got: {path}")
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def get_ext_suffix():
    if sys.version_info < (3, 8) and sys.platform == "win32":
        # Workaround for https://bugs.python.org/issue39825
        # Reference: https://github.com/pytorch/pytorch/commit/4b96fc060b0cb810965b5c8c08bc862a69965667
        import distutils

        return distutils.sysconfig.get_config_var("EXT_SUFFIX")
    return sysconfig.get_config_var("EXT_SUFFIX")


################################################################################
# Customized commands
################################################################################


class CmakeBuild(setuptools.Command):
    """Compiles everything when `python setup.py build` is run using cmake.

    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.

    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options: ClassVar[list] = [
        ("jobs=", "j", "Specifies the number of jobs to use with make")
    ]

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            build_type = "Release"
            if "CI" not in os.environ:
                generator_arg = ["-GNinja"]
            else:
                generator_arg = []
            # configure
            cmake_args = [
                CMAKE,
                *generator_arg,
                f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DFR_ENABLE_ONNX_EXPORTING=ON",
                "-DFR2ONNX_BUILD_PYTHON=ON",
                "-DFR_BUILD_PYTHON=OFF",
                "-DFR_BUILD_JNI=OFF",
                "-DFR_BUILD_ONNX=OFF",
                "-DFR_ENABLE_NCNN=OFF",
                # "-DFR_ENABLE_CUDA=ON",
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DPY_EXT_SUFFIX={get_ext_suffix() or ''}",
            ]
            if DEBUG:
                build_type = "Debug"
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
            if WINDOWS:
                cmake_args.extend(
                    [
                        # we need to link with libpython on windows, so
                        # passing python version to window in order to
                        # find python in cmake
                        f"-DPY_VERSION={'{}.{}'.format(*sys.version_info[:2])}",
                    ]
                )
                if platform.architecture()[0] == "64bit":
                    if "arm" in platform.machine().lower():
                        cmake_args.extend(["-A", "ARM64"])
                    else:
                        cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:  # noqa: PLR5501
                    if "arm" in platform.machine().lower():
                        cmake_args.extend(["-A", "ARM"])
                    else:
                        cmake_args.extend(["-A", "Win32", "-T", "host=x86"])
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                logging.info("Extra cmake args: %s", extra_cmake_args)
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            logging.info("Using cmake args: %s", cmake_args)
            if (
                "PYTHONPATH" in os.environ
                and "pip-build-env" in os.environ["PYTHONPATH"]
            ):
                # When the users use `pip install -e .` to install onnx and
                # the cmake executable is a python entry script, there will be
                # `Fix ModuleNotFoundError: No module named 'cmake'` from the cmake script.
                # This is caused by the additional PYTHONPATH environment variable added by pip,
                # which makes cmake python entry script not able to find correct python cmake packages.
                # Actually, sys.path is well enough for `pip install -e .`.
                # Therefore, we delete the PYTHONPATH variable.
                del os.environ["PYTHONPATH"]
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, "--build", os.curdir, "--target", "rwkv2onnx_python"]
            if WINDOWS:
                build_args.extend(["--config", build_type])
                build_args.extend(["--", f"/maxcpucount:{self.jobs}"])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            subprocess.check_call(build_args)


class BuildExt(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()

    def build_extensions(self):
        # We override this method entirely because the actual building is done
        # by cmake_build. Here we just copy the built extensions to the final
        # destination.
        build_lib = self.build_lib
        # extension_dst_dir = os.path.join(build_lib, "onnx")
        extension_dst_dir = build_lib
        os.makedirs(extension_dst_dir, exist_ok=True)

        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = os.path.basename(self.get_ext_filename(fullname))

            # our python library is defined in export_onnx/python/CMakelists.txt
            lib_dir = os.path.join(CMAKE_BUILD_DIR, "export_onnx", "python")
            if WINDOWS:
                # Windows compiled extensions are stored in Release/Debug subfolders
                debug_lib_dir = os.path.join(lib_dir, "Debug")
                release_lib_dir = os.path.join(lib_dir, "Release")
                if os.path.exists(debug_lib_dir):
                    lib_dir = debug_lib_dir
                elif os.path.exists(release_lib_dir):
                    lib_dir = release_lib_dir
            src = os.path.join(lib_dir, filename)
            dst = os.path.join(extension_dst_dir, filename)
            self.copy_file(src, dst)

        self.copy_file(os.path.join(TOP_DIR, "tools", "convert_weight.py"),
                       os.path.join(PYTHON_DIR, "rwkv2onnx", "convert_to_fr.py"))


CMD_CLASS = {
    "cmake_build": CmakeBuild,
    "build_ext": BuildExt,
}

################################################################################
# Extensions
################################################################################

EXT_MODULES = [setuptools.Extension(name="rwkv2onnx_python", sources=[])]


################################################################################
# Final
################################################################################

setuptools.setup(
    ext_modules=EXT_MODULES,
    cmdclass=CMD_CLASS,
    options={"bdist_wheel": {"plat_name": FR_WHEEL_PLATFORM_NAME}}
    if FR_WHEEL_PLATFORM_NAME is not None
    else {},
)


