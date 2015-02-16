
set(srcfiles 
    optimization/gd_optimizer.cpp
    optimization/numerical_gradient.cpp
	network/layer/mlp_layer.cpp
#    ./artifact/lib/network/restricted_boltzmann_machine.cpp
    network/deep_network.cpp
    network/network_creator.cpp
    network/network_trainer.cpp
    network/layer/activator.cpp
    loss/loss.cpp
    utils/matrix_io_txt.cpp
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

include_directories(EIGEN3_INCLUDE_DIR)


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
    SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
    include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/include/eigen3)
    add_library(artifact SHARED ${srcfiles})
    target_link_libraries(artifact boost_filesystem boost_system)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")



IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_FULLY_DYNAMIC_STRING -std=c++0x -O2")
			SET(CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_RELEASE)
			include_directories(${CMAKE_SOURCE_DIR}/artifact/include /usr/local/include /usr/local/include/eigen3)
			link_directories(/usr/local/lib)
			add_library(artifact SHARED ${srcfiles})
			target_link_libraries(artifact boost_filesystem boost_system)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
			include_directories(${CMAKE_SOURCE_DIR}/include D:/Lib/include D:/Lib/include/hdf5 D:/Lib/include/camp D:/Lib/include/log4cpp)
			link_directories(D:/Lib/lib/boost D:/Lib/lib/camp D:/Lib/lib/hdf5 D:/Lib/lib/log4cpp)
			add_library(learning SHARED ${srcfiles})
			target_link_libraries(learning hdf5 hdf5_cpp camp log4cpp)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")



