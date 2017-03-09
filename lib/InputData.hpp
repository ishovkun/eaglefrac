#pragma once

#include <deal.II/base/parameter_handler.h>
// #include <boost/algorithm/string.hpp>
#include <cstdlib>

// custom modules
#include <BitMap.hpp>


namespace input_data {
  using namespace dealii;

  // template <int dim>
  // class InputData
  // {
  // // public:
  //   // void read_input_file(std::string &file_name);
  //
  // // private:
  // //   void declare_parameters();
  //
  // public:
  //   // variables
  //   std::string input_file_name;
  //
  // // private:
  // //   parameterhandler prm;
  // };


  template <int dim>
  class PhaseFieldData
  {
  public:
    double young_modulus, poisson_ratio, biot_coef,
           lame_constant, shear_modulus;
    double penalty_parameter;
    double regularization_parameter_kappa;
    double regularization_parameter_epsilon;
    std::string mesh_file_name;

    std::vector<int>    displacement_boundary_labels;
    std::vector<int>    displacement_boundary_components;

    Function<dim>
      *get_young_modulus,
      *get_fracture_toughness;

  };

  template <int dim>
  class NotchedTestData : public PhaseFieldData<dim>
  {
  public:
    NotchedTestData();
    void compute_mesh_dependent_parameters(double);

  public:
    std::vector<double> displacement_boundary_velocities;

    // Solver
    double newton_tolerance;
    int max_newton_iter;
    double t_max, initial_time_step;

    // Mesh
    int initial_refinement_level,
      n_prerefinement_steps,
      n_adaptive_steps;
    double phi_refinement_value;
    std::vector<std::pair<double,double>> local_prerefinement_region;
    std::string mesh_file_name;

  private:
    void compute_runtime_parameters();
    bool homogeneous_fracture_toughness, homogeneous_youngs_modulus;
    // Properties
    double fracture_toughness_constant;
    std::pair<double,double> fracture_toughness_limits;
    std::pair<double,double> regularization_epsilon_coefficients;
    // Files
    std::string fracture_toughness_file_name;
    // Other
    std::vector<std::vector<double>> bitmap_range;
    std::vector<std::string> postprocessing_function_names;
  };

  template <int dim>
  NotchedTestData<dim>::NotchedTestData()
  {
    // displacement_boundary_labels =     {2, 3, 3};
    // displacement_boundary_components = {1, 1, 0};
    // displacement_boundary_velocities = {0, 1, 0};
    this->displacement_boundary_labels =     {2, 3};
    this->displacement_boundary_components = {1, 1};
    this->displacement_boundary_velocities = {0, 1};

    // young_modulus = 1e10;
    // poisson_ratio = 0.3;

    this->lame_constant = 121.15*1e3;
    this->shear_modulus   = 80.77*1e3;

    // phase-field control parameters
    this->penalty_parameter = 10;
    this->regularization_parameter_kappa = 1e-10;
    phi_refinement_value = 0.5;

    fracture_toughness_file_name = {"test.pgm"};
    bitmap_range = {{0, 1}, {0, 1}};
    homogeneous_fracture_toughness = false;
    fracture_toughness_constant = 2.7;
    fracture_toughness_limits = std::pair<double,double>(2.7, 7);

    regularization_epsilon_coefficients = std::pair<double,double>(4, 1);

    // Mesh
    mesh_file_name = "mesh/unit_slit.msh";
    // mesh_file_name = "mesh/notched.msh";
    // mesh_file_name = "mesh/unit_slit.inp";
    initial_refinement_level = 4;

    n_prerefinement_steps = 1;
    local_prerefinement_region.resize(dim);
    local_prerefinement_region[0] = std::pair<double,double>(0.4, 0.55);
    local_prerefinement_region[1] = std::pair<double,double>(0.4, 0.6);

    n_adaptive_steps = 3;
    phi_refinement_value = 0.5;


    t_max = 1e-2;
    initial_time_step = 1e-4;

    // Solver
    max_newton_iter = 20;
    newton_tolerance = 1e-8;

    // postprocessing
    postprocessing_function_names = {"Load"};

    compute_runtime_parameters();
  }  // EOM


  template <int dim>
  void NotchedTestData<dim>::compute_runtime_parameters()
  {
    // double E = young_modulus;
    // double nu = poisson_ratio;
    // lame_constant = E*nu/((1. + nu)*(1. - 2.*nu));
    // shear_modulus = 0.5*E/(1 + nu);

    if (homogeneous_fracture_toughness)
      this->get_fracture_toughness =
        new ConstantFunction<dim>(fracture_toughness_constant, 1);
    else
      this->get_fracture_toughness =
        new BitMap::BitMapFunction<dim>(
          fracture_toughness_file_name,
          bitmap_range[0][0], bitmap_range[0][1],
          bitmap_range[1][0], bitmap_range[1][1],
          fracture_toughness_limits.first,
          fracture_toughness_limits.second);

    //   this->get_fracture_toughness->value(p, 0) << std::endl;
  }  // eom

  template <int dim>
  void NotchedTestData<dim>::
  compute_mesh_dependent_parameters(double mesh_size)
  {
    this->regularization_parameter_epsilon =
      regularization_epsilon_coefficients.first
      *std::pow(mesh_size, regularization_epsilon_coefficients.second);

  }
}  // end of namespace
