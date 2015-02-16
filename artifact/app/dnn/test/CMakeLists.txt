
set(srcfiles 
    ./test_dnn.cpp

 )




IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
			SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
			include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/include /usr/include/eigen3)
			add_executable(test_dnn ${srcfiles})
			target_link_libraries(test_dnn artifact boost_filesystem boost_system boost_program_options)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++11 -O2")
			SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
			include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/local/include /usr/local/include/eigen3)
            link_directories(/usr/local/lib)
            add_executable(test_dnn ${srcfiles})
            target_link_libraries(test_dnn artifact boost_filesystem boost_system boost_program_options)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
			include_directories(${CMAKE_SOURCE_DIR}/include D:/Lib/include D:/Lib/include/hdf5 D:/Lib/include/camp D:/Lib/include/log4cpp)
			link_directories(D:/Lib/lib/boost D:/Lib/lib/camp D:/Lib/lib/hdf5 D:/Lib/lib/log4cpp)
			add_executable(test_dnn ${srcfiles})
			target_link_libraries(test_dnn artifact hdf5 hdf5_cpp camp log4cpp,boost_program_options)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")


add_test(NAME test_dnn COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_dnn WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
