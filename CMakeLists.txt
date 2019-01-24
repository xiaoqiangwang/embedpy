cmake_minimum_required(VERSION 2.8)

# Include custom cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(embedpy)

find_package(PythonLibs REQUIRED)
find_package(numpy REQUIRED)

set(SOURCES
    processplugin_python.cpp
)

add_executable(processplugin_python
    ${SOURCES}
)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

target_link_libraries(processplugin_python
    ${PYTHON_LIBRARIES}
)
