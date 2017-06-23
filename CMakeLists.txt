
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


IF(NOT (DEAL_II_WITH_PETSC OR DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_PETSC = ON
    DEAL_II_WITH_P4EST = ON
or
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
One or both of these combinations are OFF in your installation but at least one is required for this tutorial step."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
#SET(CLEAN_UP_FILES *.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2 *.vtu *.pvtu)
PROJECT(ipds)

# lib
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/lib)
ADD_SUBDIRECTORY(src/lib)
SET_TARGET_PROPERTIES (
    lib
    PROPERTIES LINKER_LANGUAGE CXX
)

# targets
ADD_EXECUTABLE(eaglefrac-solid ${CMAKE_SOURCE_DIR}/src/eaglefrac-solid.cc)
DEAL_II_SETUP_TARGET(eaglefrac-solid)

ADD_EXECUTABLE(eaglefrac-pressurized ${CMAKE_SOURCE_DIR}/src/eaglefrac-pressurized.cc)
DEAL_II_SETUP_TARGET(eaglefrac-pressurized)

ADD_EXECUTABLE(eaglefrac-fluid ${CMAKE_SOURCE_DIR}/src/eaglefrac-fluid.cc)
DEAL_II_SETUP_TARGET(eaglefrac-fluid)

ADD_EXECUTABLE(eaglefrac-acid ${CMAKE_SOURCE_DIR}/src/eaglefrac-acid.cc)
DEAL_II_SETUP_TARGET(eaglefrac-acid)

# linking
TARGET_LINK_LIBRARIES(eaglefrac-solid lib boost_filesystem)
TARGET_LINK_LIBRARIES(eaglefrac-pressurized lib boost_filesystem)
TARGET_LINK_LIBRARIES(eaglefrac-fluid lib boost_filesystem)
TARGET_LINK_LIBRARIES(eaglefrac-acid lib boost_filesystem)

# DEAL_II_INVOKE_AUTOPILOT()
