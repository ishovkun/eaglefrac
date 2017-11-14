# lib/CMakeLists.txt



ADD_LIBRARY(lib
  PhaseFieldSolver.hpp
  InputData.hpp
  PhaseFieldPressurizedData.hpp
  LinearSolver.hpp
  Postprocessing.hpp
  Mesher.hpp
  BitMap.hpp
  InitialValues.hpp
  PressureSolver.hpp
  SinglePhaseData.hpp
  Well.hpp
  Scheduler.hpp
  Parsers.hpp
  TemperatureSolver.hpp
  FEFunction.hpp
)

DEAL_II_SETUP_TARGET(lib)
