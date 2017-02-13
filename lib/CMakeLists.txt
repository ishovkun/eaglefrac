# lib/CMakeLists.txt



ADD_LIBRARY(lib
  PhaseFieldSolver.cc
  InputData.cc
  LinearSolver.cc
    )

DEAL_II_SETUP_TARGET(lib)
