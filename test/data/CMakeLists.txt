file(GLOB all_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(file ${all_files})
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/${file} COPYONLY)
endforeach()
