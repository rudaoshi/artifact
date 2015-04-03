





IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
	SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
	include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/include /usr/include/eigen3)
	add_executable(test_dnn_func test_dnn_functionality.cpp)
	target_link_libraries(test_dnn_func artifact boost_filesystem boost_system boost_program_options)

	add_executable(test_dnn_effi test_dnn_efficiency.cpp)
	target_link_libraries(test_dnn_effi artifact boost_filesystem boost_system boost_program_options boost_timer)

	add_executable(test_utils test_utils.cpp)
	target_link_libraries(test_utils artifact )

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++11 -O2")
	SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
	include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/local/include /usr/local/include/eigen3)
	link_directories(/usr/local/lib)
	add_executable(test_dnn_func test_dnn_functionality.cpp)
	target_link_libraries(test_dnn_func artifact boost_filesystem boost_system boost_program_options)

	add_executable(test_dnn_effi test_dnn_efficiency.cpp)
	target_link_libraries(test_dnn_effi artifact boost_filesystem boost_system boost_program_options boost_timer)

	add_executable(test_utils test_utils.cpp)
	target_link_libraries(test_utils artifact )

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	include_directories(${CMAKE_SOURCE_DIR}/include D:/Lib/include D:/Lib/include/hdf5 D:/Lib/include/camp D:/Lib/include/log4cpp)
	link_directories(D:/Lib/lib/boost D:/Lib/lib/camp D:/Lib/lib/hdf5 D:/Lib/lib/log4cpp)
	add_executable(test_dnn_func test_dnn_functionality.cpp)
	target_link_libraries(test_dnn_func artifact boost_filesystem boost_system boost_program_options)

	add_executable(test_dnn_effi test_dnn_efficiency.cpp)
	target_link_libraries(test_dnn_effi artifact boost_filesystem boost_system boost_program_options boost_timer)

	add_executable(test_utils test_utils.cpp)
	target_link_libraries(test_utils artifact )

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")

add_test(NAME test_utils COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_utils WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_dnn_func COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_dnn_func WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_dnn_effi COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_dnn_effi WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})