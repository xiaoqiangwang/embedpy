Demo of embedding python in C
=============================

processplugin_python.cpp prepares the C structure for Python. C arrays are wrapped to numpy arrays without data copying. The results from the Python process plugin (processplugin.py) is of dict type. These values could then be used to fill the C structure as needed. processplugin_python.cpp is compiled into a shared library and later loaded into a main program.

For the convenience of testing, a test main program is provided. The test image is a 2D elliptical gaussian image. The process python plugin calculates the momentum and derives the center, sigma and orientation information.

Build & Run
-----------

You could need a compiler, a Python installation with numpy and cmake.

Make sure Python is in the PATH.

On Windows::

 > mkdir build
 > pushd build
 > cmake -G "Visual Studio 14 2015 Win64" ..
 > popd
 > cmake --build build --config Release
 > set PYTHONPATH=.;%PYTHONPATH%
 > build\Release\test_program.exe

On Linux or macOS::

 $ mkdir build
 $ pushd build
 $ cmake ..
 $ popd
 $ cmake --build build --config Release
 $ PYTHONPATH=. build/test_program
