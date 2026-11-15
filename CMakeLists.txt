cmake_minimum_required(VERSION 3.20)
project(keel_fft VERSION 0.1.0 LANGUAGES CXX)

option(KEEL_FFT_WITH_HIGHWAY "Enable Google Highway SIMD acceleration in keel_fft" ON)
option(KEEL_FFT_NO_EXCEPTIONS_RTTI "Compile keel_fft without C++ exceptions and RTTI" OFF)
option(KEEL_FFT_BUILD_TESTS "Build keel_fft unit tests and header purity checks" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET keel::keel)
    find_package(keel QUIET CONFIG)
endif()
if(NOT TARGET keel::keel)
    include(FetchContent)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../keel/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_KEEL)
        set(FETCHCONTENT_SOURCE_DIR_KEEL "${CMAKE_CURRENT_SOURCE_DIR}/../keel" CACHE PATH "Local keel checkout")
    endif()
    set(KEEL_NO_EXCEPTIONS_RTTI ${KEEL_FFT_NO_EXCEPTIONS_RTTI} CACHE BOOL "" FORCE)
    FetchContent_Declare(
        keel
        GIT_REPOSITORY https://github.com/keelcpp/keel.git
        GIT_TAG        ca7fb3ed8f781d0b7af113422cdaa70a6da6eae4
    )
    FetchContent_MakeAvailable(keel)
endif()

if(KEEL_FFT_WITH_HIGHWAY)
    if(NOT TARGET hwy)
        find_package(hwy QUIET CONFIG)
        if(TARGET hwy::hwy AND NOT TARGET hwy)
            add_library(hwy ALIAS hwy::hwy)
        endif()
    endif()
    if(NOT TARGET hwy)
        include(FetchContent)
        set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
        set(_keel_fft_saved_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
        set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            highway
            GIT_REPOSITORY https://github.com/google/highway.git
            GIT_TAG        1.2.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(highway)
        if(DEFINED _keel_fft_saved_warn_deprecated AND NOT "${_keel_fft_saved_warn_deprecated}" STREQUAL "")
            set(CMAKE_WARN_DEPRECATED "${_keel_fft_saved_warn_deprecated}" CACHE BOOL "" FORCE)
        endif()
    endif()
endif()

set(KEEL_FFT_SOURCES
    src/fft.cpp
)

if(KEEL_FFT_WITH_HIGHWAY)
    list(APPEND KEEL_FFT_SOURCES
        src/fft_highway.cpp
        src/fft_r2_highway.cpp
        src/fft_r2_soa_highway.cpp
        src/fft_r4_highway.cpp
        src/fft_r4_soa_highway.cpp
        src/fft_stockham_highway.cpp
        src/fft_twist_highway.cpp
    )
else()
    list(APPEND KEEL_FFT_SOURCES
        src/fft_portable.cpp
    )
endif()

add_library(keel_fft STATIC ${KEEL_FFT_SOURCES})
add_library(keel::fft ALIAS keel_fft)

target_link_libraries(keel_fft PUBLIC keel::keel)
if(KEEL_FFT_WITH_HIGHWAY)
    target_link_libraries(keel_fft PRIVATE $<BUILD_INTERFACE:hwy>)
    target_compile_definitions(keel_fft PRIVATE KEEL_FFT_WITH_HIGHWAY=1)
    get_target_property(_keel_fft_hwy_inc hwy INTERFACE_INCLUDE_DIRECTORIES)
    if(_keel_fft_hwy_inc)
        target_include_directories(keel_fft SYSTEM PRIVATE ${_keel_fft_hwy_inc})
    endif()
endif()

target_compile_features(keel_fft PUBLIC cxx_std_20)
set_target_properties(keel_fft PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(keel_fft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(MSVC)
    target_compile_options(keel_fft PRIVATE /W4 /utf-8 /bigobj)
    if(KEEL_FFT_NO_EXCEPTIONS_RTTI)
        string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REGEX REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        target_compile_options(keel_fft PUBLIC /EHs-c- /GR-)
        target_compile_definitions(keel_fft PUBLIC _HAS_EXCEPTIONS=0)
    endif()
else()
    target_compile_options(keel_fft PRIVATE -Wall -Wextra -Wpedantic)
    if(KEEL_FFT_NO_EXCEPTIONS_RTTI)
        target_compile_options(keel_fft PUBLIC -fno-exceptions -fno-rtti)
    endif()
endif()

include(CMakePackageConfigHelpers)
install(TARGETS keel_fft EXPORT keel_fftTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/keel DESTINATION include)
install(EXPORT keel_fftTargets
    FILE keel_fftTargets.cmake
    NAMESPACE keel::
    DESTINATION lib/cmake/keel_fft
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/keel_fftConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_fftConfig.cmake"
    INSTALL_DESTINATION lib/cmake/keel_fft
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/keel_fftConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/keel_fftConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/keel_fftConfigVersion.cmake"
    DESTINATION lib/cmake/keel_fft
)

if(KEEL_FFT_BUILD_TESTS)
    enable_testing()

    add_executable(test_fft tests/test_fft.cpp)
    target_link_libraries(test_fft PRIVATE keel::fft)
    if(MSVC)
        target_compile_options(test_fft PRIVATE /W4 /utf-8)
    else()
        target_compile_options(test_fft PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME test_fft COMMAND test_fft)

    get_target_property(KEEL_CORE_INC_DIRS keel::keel INTERFACE_INCLUDE_DIRECTORIES)
    set(KEEL_CORE_INC_CLEAN "")
    foreach(IDIR IN LISTS KEEL_CORE_INC_DIRS)
        string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" IDIR_STRIPPED "${IDIR}")
        if(NOT IDIR_STRIPPED MATCHES "\\$<INSTALL_INTERFACE:" AND EXISTS "${IDIR_STRIPPED}")
            list(APPEND KEEL_CORE_INC_CLEAN "${IDIR_STRIPPED}")
        endif()
    endforeach()
    string(REPLACE ";" "|" KEEL_DEP_INC_JOINED "${KEEL_CORE_INC_CLEAN}")
    string(REPLACE ";" "|" KEEL_STD_INC_JOINED "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")

    add_test(
        NAME test_header_purity
        COMMAND ${CMAKE_COMMAND}
            "-DKEEL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-DKEEL_DEP_INC_DIRS=${KEEL_DEP_INC_JOINED}"
            "-DCOMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCOMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DSTD_INC_DIRS=${KEEL_STD_INC_JOINED}"
            "-DCMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckHeaderPurity.cmake"
    )
endif()
