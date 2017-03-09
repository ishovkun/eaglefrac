# lib/CMakeLists.txt



ADD_LIBRARY(lib
  PhaseFieldSolver.hpp
  InputData.hpp
  LinearSolver.hpp
  Postprocessing.hpp
  Mesher.hpp
  BitMap.hpp
)

DEAL_II_SETUP_TARGET(lib)
