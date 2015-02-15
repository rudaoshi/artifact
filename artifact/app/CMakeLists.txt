cmake_minimum_required(VERSION 2.8)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
			add_subdirectory(dnn)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
			add_subdirectory(dnn)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
			add_subdirectory(dnn)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")



