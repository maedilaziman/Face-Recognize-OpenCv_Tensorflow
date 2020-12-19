
# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

project(FACEDETECT)

get_filename_component(SRC_DIR  ${CMAKE_SOURCE_DIR}/..  ABSOLUTE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -frtti")

#cmake_minimum_required(VERSION 3.4.1)

# OpenCV stuff
#file(GLOB_RECURSE jnilibs_sources ${PROJECT_SOURCE_DIR}/app/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)
file(GLOB_RECURSE facedetect_sources ${SRC_DIR}/jni/facedetect/cpp/native-lib.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/native/jni/include)
#include_directories(${PROJECT_SOURCE_DIR}/native/jni/include)
add_library( lib_opencv SHARED IMPORTED )
#set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION  ${CMAKE_CURRENT_SOURCE_DIR}/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             ${facedetect_sources} )


# Searches for a specified prebuilt library and stores the path as a
# variable. Because system libraries are included in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       # OpenCV lib
                       lib_opencv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )
