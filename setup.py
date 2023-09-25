from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
# import multiprocessing
# import os
# import subprocess
# import sys
# from setuptools.command.build_ext import build_ext


# Using the BuildExtension from torch instead of cmake due below issue  
# refer why retuning tensor is problem https://github.com/pytorch/pytorch/issues/73016

# class CMakeExtension(Extension):
#     def __init__(self, name, source_dir=""):
#         # don't invoke the original build_ext for this special extension
#         super().__init__(name, sources=[])
#         self.source_dir = os.path.abspath(source_dir)


# class CMakeBuild(build_ext):
#     def run(self):
#         try:
#             _ = subprocess.check_output(["cmake", "--version"])
#         except OSError:
#             raise RuntimeError("CMake must be installed to build the following extensions: " +
#                                ", ".join(ext.name for ext in self.extensions))

#         try:
#             import torch
#         except ImportError:
#             sys.stderr.write("Pytorch is required to build this package\n")
#             sys.exit(-1)

#         self.pytorch_dir = os.path.dirname(torch.__file__)
#         self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()

#         for ext in self.extensions:
#             self.build_cmake(ext)

#     def build_cmake(self, ext):
#         ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
#         cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
#                       "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
#                       "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
#                       "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
#                       ]

#         config = "Debug" if self.debug else "Release"
#         build_args = ["--config", config]

        
#         cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         cwd = os.getcwd()
#         os.chdir(os.path.dirname(ext_dir))
#         self.spawn(["cmake", ext.source_dir] + cmake_args)
#         if not self.dry_run:
#             self.spawn(["cmake", "--build", ".", "--", "-j{}".format(multiprocessing.cpu_count())])
#         os.chdir(cwd)


# setup(
#     name="quantize",
#     packages=find_packages(),
#     ext_modules=[CMakeExtension("quantize")],
#     cmdclass={"build_ext": CMakeBuild, }
# )


setup(
    name='quantize',
    ext_modules=[
        CUDAExtension(
            name ='quantize', 
            sources = [
            'csrc/quantize/quantize.cpp',
            'csrc/quantize/src/quantize.cu'],
            include_dirs=["csrc/quantize/include"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
