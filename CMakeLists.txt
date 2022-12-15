 cmake_minimum_required(VERSION 3.11)

project("Boxed")

include(FetchContent)

FetchContent_Declare(
  glew
  URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz
)

FetchContent_GetProperties(glew)
if(NOT glew_POPULATED)
  FetchContent_Populate(glew)
  set(BUILD_UTILS
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(${glew_SOURCE_DIR}/build/cmake EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw
  GIT_PROGRESS TRUE
  GIT_SHALLOW TRUE
  GIT_TAG origin/master)

if(NOT glfw_POPULATED)
  FetchContent_Populate(glfw)
  set(GLFW_BUILD_DOCS
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_INSTALL
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(${glfw_SOURCE_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_PROGRESS TRUE
  GIT_SHALLOW TRUE
  GIT_TAG origin/master)

if(NOT json_POPULATED)
  FetchContent_Populate(json)
  set(JSON_Install
      OFF
      CACHE INTERNAL "" FORCE)
  set(JSON_BuildTests
      OFF
      CACHE INTERNAL "" FORCE)
  set(JSON_ImplicitConversions
      OFF
      CACHE INTERNAL "" FORCE)
  add_subdirectory(${json_SOURCE_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui
  GIT_PROGRESS TRUE
  GIT_SHALLOW TRUE
  GIT_TAG origin/master)

if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)
endif()

FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot
  GIT_PROGRESS TRUE
  GIT_SHALLOW TRUE
  GIT_TAG origin/master)

if(NOT implot_POPULATED)
  FetchContent_Populate(implot)
endif()

add_executable(
  boxed
  "boxed.cpp"
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imstb_textedit.h
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
  ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot.cpp)

set_target_properties(boxed PROPERTIES CXX_STANDARD 20)

target_link_libraries(boxed glfw glew_s nlohmann_json::nlohmann_json)

if(MSVC)
  target_link_options(boxed PRIVATE "/SUBSYSTEM:WINDOWS"
                      "/ENTRY:mainCRTStartup")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcoroutines")
endif()

target_include_directories(
  boxed
  PRIVATE ${implot_SOURCE_DIR}
          ${imgui_SOURCE_DIR}
          ${json_SOURCE_DIR}
          ${glew_SOURCE_DIR}/include)