# lib/CMakeLists.txt



ADD_LIBRARY(lib
  PhaseFieldSolver.cc
  InputData.cc
    )

DEAL_II_SETUP_TARGET(lib)
