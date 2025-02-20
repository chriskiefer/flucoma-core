# Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
# Copyright 2017-2019 University of Huddersfield.
# Licensed under the BSD-3 License.
# See license.md file in the project root for full license information.
# This project has received funding from the European Research Council (ERC)
# under the European Union’s Horizon 2020 research and innovation programme
# (grant agreement No 725899).

cmake_minimum_required (VERSION 3.11)

get_directory_property(hasParent PARENT_DIRECTORY)

#set module path at top level so wrapper projects can easily include fluid_version script
if(hasParent)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/script PARENT_SCOPE)
endif()

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.8" CACHE STRING "")
  #A consequence of targetting 10.8. Needs to be set globally from 10.15 onwards in order for the test program to compile successfully during configure 
  string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
endif()

project (flucoma-core LANGUAGES CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/script/flucoma-buildtype.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/FlucomaVersion.cmake")

include(FetchContent)
set(HISS_PATH "" CACHE PATH "The path to a HISSTools_Library folder. Will pull from github if not set")
set(EIGEN_PATH "" CACHE PATH "The path to an Eigen installation (>=3.3.5). Will pull from github if not set")
set(SPECTRA_PATH "" CACHE PATH "The path to aa Spectra installation. Will pull from github if not set")
IF(APPLE)
  find_library(ACCELERATE Accelerate)
  IF (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate framework not found")
  ENDIF()
ENDIF (APPLE)

# Grab the Fluid Decpomposition header files so they can be added to IDE builds
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

set(FETCHCONTENT_QUIET FALSE)

# Either download or point to dependencies
FetchContent_Declare(
  HISSTools
  GIT_REPOSITORY https://github.com/AlexHarker/HISSTools_Library
  GIT_PROGRESS TRUE
  GIT_TAG 5dd8530
)

FetchContent_Declare(
  Eigen
  GIT_SHALLOW TRUE
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_PROGRESS TRUE
  GIT_BRANCH "3.4"
  GIT_TAG "3.4.0"
)

FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_PROGRESS TRUE
  GIT_BRANCH "master"
  GIT_TAG "v0.9.0"
)

if(HISS_PATH) #if hiss path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_HISSTOOLS ${HISS_PATH} ABSOLUTE)
endif()

if(EIGEN_PATH) #if eigen path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_EIGEN ${EIGEN_PATH} ABSOLUTE)
endif()

if(SPECTRA_PATH) #if spectra path is set, this will stop it downloading
  get_filename_component(FETCHCONTENT_SOURCE_DIR_SPECTRA ${SPECTRA_PATH} ABSOLUTE)
endif()


FetchContent_GetProperties(HISSTools)
if(NOT hisstools_POPULATED)
  FetchContent_Populate(HISSTools)
endif()

FetchContent_GetProperties(Eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(Eigen)
endif()

FetchContent_GetProperties(Spectra)
if(NOT spectra_POPULATED)
  FetchContent_Populate(Spectra)
endif()


# HISSTools FFT target
add_library(
  HISSTools_FFT STATIC "${hisstools_SOURCE_DIR}/HISSTools_FFT/HISSTools_FFT.cpp"
)

target_link_libraries(
  HISSTools_FFT PRIVATE ${ACCELERATE}
)

target_include_directories(HISSTools_FFT PUBLIC "${hisstools_SOURCE_DIR}")

set_target_properties(HISSTools_FFT PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Brute force staic runtime on windwos
if(MSVC)
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif()
  endforeach()
endif()

#HISSTools Audiofile Target
add_library(
  HISSTools_AudioFile STATIC
  "${hisstools_SOURCE_DIR}/AudioFile/BaseAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/IAudioFile.cpp"
  "${hisstools_SOURCE_DIR}/AudioFile/OAudioFile.cpp"
)

set_target_properties(HISSTools_AudioFile PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

#Fluid Decomposition header-only target
add_library(FLUID_DECOMPOSITION INTERFACE)

target_include_directories(
  FLUID_DECOMPOSITION INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty"
)
target_include_directories(
  FLUID_DECOMPOSITION SYSTEM INTERFACE #we don't want warnings from Eigen or HissTools
  "${eigen_SOURCE_DIR}"
  "${spectra_SOURCE_DIR}/include"
  "${hisstools_SOURCE_DIR}"
)
target_link_libraries(
  FLUID_DECOMPOSITION INTERFACE HISSTools_FFT flucoma_VERSION_LIB
)
target_sources(
  FLUID_DECOMPOSITION INTERFACE ${HEADERS}
)

if(MSVC)
  target_compile_definitions(
    FLUID_DECOMPOSITION INTERFACE NOMINMAX _USE_MATH_DEFINES
  )
endif()

#GCC vomits on using HostVector = HostVector<U> without this flag on
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -fpermissive)
endif()

target_compile_definitions(FLUID_DECOMPOSITION INTERFACE EIGEN_MPL2_ONLY=1)

if(APPLE)
  #targeting <= 10.9, need to really emphasise that we want libc++ both to compiler and linker
  target_compile_options(HISSTools_FFT PUBLIC -stdlib=libc++)
  target_compile_options(HISSTools_AudioFile PUBLIC -stdlib=libc++)
  target_compile_options(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++)
  target_link_libraries(FLUID_DECOMPOSITION INTERFACE -stdlib=libc++)


endif()

#Apply any vector instruction flags
if(DEFINED FLUID_ARCH)
  target_compile_options(HISSTools_FFT PUBLIC ${FLUID_ARCH})
  target_compile_options(HISSTools_AudioFile PUBLIC ${FLUID_ARCH})
  target_compile_options(FLUID_DECOMPOSITION INTERFACE ${FLUID_ARCH})
endif()

#Examples
add_subdirectory(
   "${CMAKE_CURRENT_SOURCE_DIR}/examples"
)

enable_testing()

if(FLUCOMA_TESTS)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
