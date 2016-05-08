cmake_minimum_required(VERSION 2.8.7)

cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0009 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(bb_binary)

#-----------------------------------------------------------------------
# CPM configuration
#-----------------------------------------------------------------------
set(CPM_MODULE_NAME "bb_binary")
set(CPM_LIB_TARGET_NAME ${CPM_MODULE_NAME})

if ((DEFINED CPM_DIR) AND (DEFINED CPM_UNIQUE_ID) AND (DEFINED CPM_TARGET_NAME))
  set(CPM_LIB_TARGET_NAME ${CPM_TARGET_NAME})
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CPM_DIR})
  include(CPM)
  else()
  set(CPM_DIR "${CMAKE_CURRENT_BINARY_DIR}/cpm-packages" CACHE TYPE STRING)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "CPM requires Git.")
  endif()
  if (NOT EXISTS ${CPM_DIR}/CPM.cmake)
    message(STATUS "Cloning repo (https://github.com/iauns/cpm)")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" clone https://github.com/iauns/cpm ${CPM_DIR}
      RESULT_VARIABLE error_code
      OUTPUT_QUIET ERROR_QUIET)
    if(error_code)
      message(FATAL_ERROR "CPM failed to get the hash for HEAD")
    endif()
  endif()
  include(${CPM_DIR}/CPM.cmake)
endif()

if(NOT DEFINED CMAKECONFIG_PATH)
    CPM_AddModule("cmakeconfig"
        GIT_REPOSITORY "https://github.com/BioroboticsLab/cmakeconfig.git"
        GIT_TAG "master")
else()
    CPM_AddModule("cmakeconfig"
        SOURCE_DIR "${CMAKECONFIG_PATH}")
endif()

CPM_AddModule("cpm_fast_cpp_csv_parser"
    GIT_REPOSITORY "https://github.com/BioroboticsLab/cpm_fast_cpp_csv_parser"
    USE_EXISTING_VER TRUE)

CPM_AddModule("cpm_catch"
    GIT_REPOSITORY "https://github.com/bcachet/cpm-catch")

CPM_InitModule(${CPM_MODULE_NAME})

biorobotics_config()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ ${CMAKE_MODULE_PATH})

# project dependecies
find_package(Threads REQUIRED)
# find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(FlatBuffers REQUIRED)
find_package(LZ4 REQUIRED)

build_flatbuffers(
    "${CMAKE_CURRENT_SOURCE_DIR}/schema/bb_binary_schema.fbs"
    ""
    "build_fbs"
    ""
    "${CMAKE_CURRENT_SOURCE_DIR}/bb_binary/"
    ""
    "")

include_directories(
    SYSTEM ${CATCH_INCLUDE_DIR}
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${CUDA_INCLUDE_DIRS}
    SYSTEM ${FLATBUFFERS_INCLUDE_DIRS}
    SYSTEM ${LZ4_INCLUDE_DIR}
)

set(LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CUDA_LIBRARIES}
    ${LZ4_LIBS}
    ${CPM_LIBRARIES}
)
add_subdirectory(bb_binary)

set(TEST_LIBS ${LIBS} ${CPM_LIB_TARGET_NAME})
add_subdirectory(test)
