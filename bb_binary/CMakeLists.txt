file(GLOB_RECURSE src RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
list(REMOVE_ITEM src  "src/main.cpp")
file(GLOB hdr "${CMAKE_CURRENT_SOURCE_DIR}/bb_binary/" "*.h")

add_library(${CPM_LIB_TARGET_NAME} ${src} ${hdr} )
target_link_libraries(${CPM_LIB_TARGET_NAME} ${LIBS})

add_dependencies(${CPM_LIB_TARGET_NAME} build_fbs)

add_executable(bbb "src/main.cpp" ${hdr})
target_link_libraries(bbb ${CPM_LIB_TARGET_NAME} ${LIBS})
