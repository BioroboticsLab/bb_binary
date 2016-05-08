enable_testing(true)

include_directories(
    ${PROJECT_SOURCE_DIR}/bb_binary
)

file(GLOB tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_*.cpp)

foreach(test ${tests})
    get_filename_component(name ${test} NAME_WE)
    add_executable(${name} ${test})
    target_link_libraries(${name} ${TEST_LIBS})
    set(test_bin ${CMAKE_CURRENT_BINARY_DIR}/${name})
    add_test(${name} ${test_bin})
endforeach()
