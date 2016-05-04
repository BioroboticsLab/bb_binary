message("${CMAKE_CURRENT_SOURCE_DIR}/bb_binary")

message("${src}")
file(GLOB_RECURSE src RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
# list(REMOVE_ITEM src  "src/caffe_stats.cpp")
file(GLOB hdr "${CMAKE_CURRENT_SOURCE_DIR}/bb_binary/" "*.h")
message("${src}")
set(hdr "${hdr};${CMAKE_CURRENT_SOURCE_DIR}/bb_binary_generated.h")
message("${hdr}")

add_library(${CPM_LIB_TARGET_NAME} ${src} ${hdr} )
target_link_libraries(${CPM_LIB_TARGET_NAME} ${LIBS})

add_dependencies(${CPM_LIB_TARGET_NAME} build_fbs)

#add_executable(caffe_stats EXCLUDE_FROM_ALL "src/caffe_stats.cpp" ${hdr})
#target_link_libraries(caffe_stats ${CPM_LIB_TARGET_NAME} ${LIBS})
