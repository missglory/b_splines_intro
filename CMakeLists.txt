cmake_minimum_required(VERSION 2.8)

project(opencv__test)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(${PROJECT_NAME} "main.cpp")
message("OpenCV_LIBS: " ${OpenCV_LIBS})
message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
