# CMakeList.txt : CMake project for my-project, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
include(FetchContent)

project ("soya-pareto")

# Add source to this project's executable.
add_executable (find-pareto "src/find-pareto.cpp")
add_executable (analyze-front "src/analyze-front.cpp")
add_executable (find-dominated "src/find-dominated.cpp")
add_executable (merge-dominated "src/merge-undominated.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET find-pareto PROPERTY CXX_STANDARD 23)
  set_property(TARGET analyze-front PROPERTY CXX_STANDARD 23)
  set_property(TARGET find-dominated PROPERTY CXX_STANDARD 23)
  set_property(TARGET merge-dominated PROPERTY CXX_STANDARD 23)
endif()

# Fetch csv-parser and make it available to my-executable
FetchContent_Declare(
  csv  
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE 
  GIT_TAG 2.2.1
) 

FetchContent_MakeAvailable(csv)

target_include_directories(find-pareto PRIVATE /home/gkatsi/pkgs/CPLEX_Studio127/cplex/include)
target_link_directories(find-pareto PRIVATE /home/gkatsi/pkgs/CPLEX_Studio127/cplex/lib/x86-64_linux/static_pic)
target_link_libraries(find-pareto cplex)

target_include_directories(analyze-front PRIVATE csv)
target_link_libraries(analyze-front csv  ${CMAKE_SOURCE_DIR}/src/fpli_hv.a)

target_include_directories(find-dominated PRIVATE csv)
target_link_libraries(find-dominated csv)

target_include_directories(merge-dominated PRIVATE csv)
target_link_libraries(merge-dominated csv)
