cmake_minimum_required(VERSION 2.8)

# Include custom cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(embedpy)

find_package(PythonLibs REQUIRED)
find_package(numpy REQUIRED)

# create the process plugin
set(SOURCES
    processplugin_python.cpp
)

add_library(processplugin_python SHARED
    ${SOURCES}
)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

target_compile_definitions(processplugin_python
    PUBLIC PROCESSPLUGIN_PYTHON_EXPORT
)

target_link_libraries(processplugin_python
    ${PYTHON_LIBRARIES}
)

# create the test program
add_executable(test_program test_program.cpp)
target_link_libraries(test_program processplugin_python)
