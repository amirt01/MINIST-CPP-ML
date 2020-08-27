cmake_minimum_required(VERSION 3.17)
project(MNIST_ML)

set(CMAKE_CXX_STANDARD 20)

add_executable(knn KNN/src/knn.cpp KNN/include/knn.hpp)
add_executable(k-means K-Means/src/KMeans.cpp K-Means/include/KMeans.hpp)
add_executable(MINIST_Loader_Example
        include/Data.hpp
        include/DataHandler.hpp
        src/Data.cpp
        src/DataHandler.cpp)
add_library(MINIST_Loader_Library STATIC
        include/Data.hpp
        include/DataHandler.hpp
        include/CommonAlg.hpp
        src/Data.cpp
        src/DataHandler.cpp
        src/CommonAlg.cpp)

include_directories(MINIST_Loader_Library/include KNN/include K-Means/include)

find_library(MINIST_Loader_Library lib)
target_link_libraries(knn MINIST_Loader_Library)
target_link_libraries(k-means MINIST_Loader_Library)