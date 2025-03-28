cmake_minimum_required(VERSION 3.2)
project(guide)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")

set(3RD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rd)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/debug)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin/debug/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG  ${PROJECT_SOURCE_DIR}/bin/debug/)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DALIGNMENT_REQUIRED -march=native -mtune=native")

file(GLOB_RECURSE HEAD_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE SOURCES_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

set(SOURCE_FILE
${HEAD_LIST}
${SOURCES_LIST}
)

find_package(PCL 1.14.0 REQUIRED)
if(PCL_FOUND)
	include_directories(${PCL_INCLUDE_DIRS})
    MESSAGE("PCL include directories: ${PCL_INCLUDE_DIRS}")
	link_directories(${PCL_LIBRARY_DIRS})
	add_definitions(${PCL_DEFINITIONS} -DGLOG_NO_ABBREVIATED_SEVERITIES)
	#add_definitions(${PCL_DEFINITIONS})
endif()
add_definitions(-DBUILD_DLL)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}/src 
${3RD_PATH}/opencv-4.8.0/include
${3RD_PATH}/Boost/include/boost-1_84/
${3RD_PATH}/eigen
${3RD_PATH}/VTK/include/vtk-9.3/
${3RD_PATH}/FLANN/include/
${PCL_INCLUDE_DIRS}
)

set(3RD_LIB 
${3RD_PATH}/opencv-4.8.0/lib/opencv_world480.lib
${3RD_PATH}/VTK/lib/
${3RD_PATH}/FLANN/lib/flann.lib
${3RD_PATH}/Boost/lib/libboost_serialization-vc143-mt-x64-1_84.lib
${PCL_LIBRARIES}
)

if(MSVC)
    add_compile_options(/bigobj)
endif()

add_compile_definitions(GUIDE_EXPORTS)
add_library(${PROJECT_NAME} ${SOURCE_FILE})
add_executable(guide_main ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(guide_main
${3RD_LIB}
${WIN32LIB}
${LINUXLIB}
${PROJECT_NAME}
)

target_compile_options(guide_main PRIVATE -mavx)
