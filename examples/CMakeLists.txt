set(EXAMPLES_SOURCES 
      "${CMAKE_CURRENT_SOURCE_DIR}/01_callback.cpp"
)

foreach(SOURCE ${EXAMPLES_SOURCES})
    # Get the filename without the extension
    get_filename_component(FILENAME ${SOURCE} NAME_WE)

    # Create the target
    add_executable(Example_${FILENAME} ${SOURCE})
    
    #Link with the lib
    target_link_libraries(Example_${FILENAME} PRIVATE ${LIB_NAME})

endforeach()
