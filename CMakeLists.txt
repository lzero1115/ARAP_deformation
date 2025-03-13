cmake_minimum_required(VERSION 3.16)
project(ARAP VERSION 1.0.0 LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Release build type (overrides any command-line settings)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Define the dependencies folder
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)

# Add Geometry Central
#add_subdirectory(${DEPS_DIR}/geometry-central)

# Configure libigl options: we only use its core functionality
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" ON)
option(LIBIGL_WITH_OPENGL         "Use OpenGL"           OFF)
option(LIBIGL_WITH_OPENGL_GLFW    "Use GLFW"             OFF)
option(LIBIGL_WITH_VIEWER         "Use viewer"           OFF)
option(LIBIGL_WITH_PNG            "Use PNG"              OFF)
option(LIBIGL_WITH_IMGUI          "Use ImGui"            OFF)
add_subdirectory(${DEPS_DIR}/libigl)

# Add Polyscope for visualization (added after libigl so its settings take precedence)
option(POLYSCOPE_BACKEND_OPENGL3_GLFW "Use OpenGL3 backend with GLFW" ON)
add_subdirectory(${DEPS_DIR}/polyscope)

# Include project's header directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files from the src folder and headers from include (optional)
file(GLOB_RECURSE CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CORE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

# Create a library from your source files (core functionality)
add_library(core ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link the core library with dependencies that its code might use.
# Using PUBLIC ensures that if main.cpp uses these libraries too,
# the link dependencies are transitive.
target_link_libraries(core PUBLIC
        #geometry-central
        polyscope
        igl::core
)

# Create the executable using main.cpp
add_executable(${PROJECT_NAME} main.cpp)
# Create a polyscope example program
add_executable(polyscope_example polyscope_example.cpp)

# Link the executable with the core library and also explicitly link
# Geometry Central and libigl (igl::core) in case main.cpp uses them.
target_link_libraries(${PROJECT_NAME} PRIVATE
        core
        #geometry-central
        igl::core
        polyscope
)

target_link_libraries(polyscope_example PRIVATE
        core
        #geometry-central
        igl::core
        polyscope
)



# Add this line to propagate the definition to both targets
target_compile_definitions(core PUBLIC USE_POLYSCOPE_VISUALIZATION)
# And keep the existing one for the executable
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_POLYSCOPE_VISUALIZATION)
target_compile_definitions(polyscope_example PRIVATE USE_POLYSCOPE_VISUALIZATION)
