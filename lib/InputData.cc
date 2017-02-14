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

  };

  NotchedTestData::NotchedTestData()
  {
    young_modulus = 1e10;
    poisson_ratio = 0.3;
    displacement_boundary_labels = {2, 3};
    displacement_boundary_components = {1 ,1};
    displacement_boundary_velocities = {0, 1e-3};
    regularization_parameter_epsilon = 1;
  }  // EOM

}  // end of namespace
