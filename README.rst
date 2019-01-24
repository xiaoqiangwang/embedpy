Demo of embedding python in C
=============================

processplugin_python.cpp prepares the C structure for Python. C arrays are wrapped to numpy arrays without data copying.

The results from the Python process plugin (processplugin.py) is of dict type. These values could then be used to fill the C structure as needed.

processplugin_python.cpp in reality would be compiled into a shared library and later loaded into a main program. But for the convenience of testing, a test function is provided and it is compiled as a main program.

The test image is a 2D elliptical gaussian image. The process python plugin calculates the momentum and derives the center, sigma and orientation information.

Build & Run
-----------

You could need a compiler, a Python installation with numpy and cmake.

Make sure Python is in the PATH.

On Windows,::

 > mkdir build
 > pushd build
 > cmake -G "Visual Studio 14 2015 Win64" ..
 > popd
 > cmake --build build --config Release
 > set PYTHONPATH=.;%PYTHONPATH%
 > build\Release\processplugin_python.exe
