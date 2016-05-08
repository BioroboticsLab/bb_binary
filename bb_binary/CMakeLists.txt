file(GLOB_RECURSE src RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
list(REMOVE_ITEM src  "src/main.cpp")
file(GLOB hdr "${CMAKE_CURRENT_SOURCE_DIR}/bb_binary/" "*.h")

set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS ../bb_binary/bb_binary_schema.capnp)

message("${CAPNP_SRCS}")
message("${CAPNP_HDRS}")

list(APPEND src ${CAPNP_SRCS})
list(APPEND hdr ${CAPNP_HDRS})
add_library(${CPM_LIB_TARGET_NAME} ${src} ${hdr})
target_link_libraries(${CPM_LIB_TARGET_NAME} ${LIBS})

add_executable(bbb "src/main.cpp" ${hdr})
target_link_libraries(bbb ${CPM_LIB_TARGET_NAME} ${LIBS})
