#include <deal.II/base/parameter_handler.h>
// #include <boost/algorithm/string.hpp>
#include <cstdlib>


namespace input_data {
  using namespace dealii;

  class InputData
  {
  public:
    void read_input_file(std::string file_name);

  private:
    void declare_parameters();

  public:
    // variables
    std::string input_file_name;
    int initial_refinement_level, max_refinement_level;

  private:
    ParameterHandler prm;
  };


  class PhaseFieldData : public InputData
  {
  public:
    double young_modulus, poisson_ratio, biot_coef,
           lame_constant, shear_modulus;
    double penalty_parameter, energy_release_rate;
    double regularization_parameter_epsilon, regularization_parameter_kappa;

    std::vector<int>    displacement_boundary_labels = {2, 3};
    std::vector<int>    displacement_boundary_components = {1 ,1};

  };

  class NotchedTestData : public PhaseFieldData
  {
  public:
    NotchedTestData();

  public:
    std::vector<double> displacement_boundary_velocities;
    double newton_tolerance;
    unsigned int max_newton_iter, max_linear_solver_iter;

  };

  NotchedTestData::NotchedTestData()
  {
    // young_modulus = 1e10;
    // poisson_ratio = 0.3;
    // double E = young_modulus;
    // double nu = poisson_ratio;
    // lame_constant = E*nu/((1. + nu)*(1. - 2.*nu));
    // shear_modulus = 0.5*E/(1 + nu);
    lame_constant = 121.15*1e3/(1e-6);
    shear_modulus = 80.77*1e3/(1e-6);

    displacement_boundary_labels =     {0, 1,    1};
    displacement_boundary_components = {0 ,0,    1};
    displacement_boundary_velocities = {0, 1e-3, 0};
    // regularization_parameter_epsilon = 1;
    // penalty_parameter = 0.125;
    penalty_parameter = 0.25;
    // penalty_parameter = 10;
    regularization_parameter_kappa = 1e-10;
    energy_release_rate = 2.7/(1e-3);

    initial_refinement_level = 5;

    // Solver
    // max_newton_iter = 50;
  }  // EOM

}  // end of namespace
