cmake_minimum_required (VERSION 3.8)

project ("Boxed")

set(GLFW glfw-3.3.5)
set(GLEW glew-2.2.0)
set(IMGUI imgui-1.85)
set(JSON json-3.10.4)

add_subdirectory(${PROJECT_SOURCE_DIR}/${GLFW} EXCLUDE_FROM_ALL)
add_subdirectory(${PROJECT_SOURCE_DIR}/${GLEW}/build/cmake EXCLUDE_FROM_ALL)
add_subdirectory(${PROJECT_SOURCE_DIR}/${JSON} EXCLUDE_FROM_ALL)

add_executable (boxed "boxed.cpp" "implot.h" "implot_internal.h" "implot.cpp" "implot_items.cpp" ${IMGUI}/imconfig.h ${IMGUI}/imgui.cpp ${IMGUI}/imgui.h ${IMGUI}/imgui_draw.cpp ${IMGUI}/imgui_internal.h ${IMGUI}/imgui_tables.cpp ${IMGUI}/imgui_widgets.cpp ${IMGUI}/imstb_rectpack.h ${IMGUI}/imstb_textedit.h ${IMGUI}/imstb_truetype.h ${IMGUI}/backends/imgui_impl_glfw.cpp ${IMGUI}/backends/imgui_impl_glfw.h ${IMGUI}/backends/imgui_impl_opengl2.cpp ${IMGUI}/backends/imgui_impl_opengl2.h)

set_target_properties(boxed PROPERTIES CXX_STANDARD 20)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

set(BUILD_UTILS OFF CACHE BOOL "" FORCE)

set(JSON_Install OFF CACHE INTERNAL "" FORCE)
set(JSON_BuildTests OFF CACHE INTERNAL "" FORCE)
set(JSON_ImplicitConversions OFF CACHE INTERNAL "" FORCE)

target_link_libraries(boxed glfw glew_s nlohmann_json::nlohmann_json)

IF (WIN32)
    target_link_options(boxed PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
ENDIF()

target_include_directories(boxed PRIVATE
    ${PROJECT_SOURCE_DIR}/${GLEW}/include
)

target_include_directories(boxed PRIVATE
    ${PROJECT_SOURCE_DIR}/${IMGUI}/
)

target_include_directories(boxed PRIVATE
    ${PROJECT_SOURCE_DIR}/${JSON}/
)