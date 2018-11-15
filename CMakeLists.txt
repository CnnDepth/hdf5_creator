cmake_minimum_required( VERSION 3.0.0 )

project( test_hdf5 VERSION 0.0.1 )

find_package( HDF5 COMPONENTS C CXX HL REQUIRED )
find_package( Boost COMPONENTS system filesystem regex REQUIRED )
find_package( OpenCV REQUIRED )

if( OPENCV_CUDA_VERSION )
  message( STATUS "Found OpenCV compiled with CUDA version:  ${OPENCV_CUDA_VERSION}" )
endif()

file( GLOB TOOLS_SOURCE_FILES src/data_transformer/*.cpp )
file( GLOB TOOLS_HEADER_FILES include/data_transformer/*.hpp )

file( GLOB COMPOSE_HDF5_SOURCE_FILE src/compose_hdf5.cpp )
file( GLOB CROP_DATA_SOURCE_FILE src/crop_data.cpp )
#file( GLOB COMPOSE_HDF5_HEADER_FILES include/.hpp )

link_libraries( ${HDF5_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} )
include_directories( include ${HDF5_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )

add_executable( compose_hdf5 ${TOOLS_SOURCE_FILES} ${COMPOSE_HDF5_SOURCE_FILE} )
add_executable( crop_data ${TOOLS_SOURCE_FILES} ${CROP_DATA_SOURCE_FILE} )

target_link_libraries( compose_hdf5 ${HDF5_CXX_LIBRARIES} )

