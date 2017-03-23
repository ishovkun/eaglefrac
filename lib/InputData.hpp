#pragma once

#include <deal.II/base/parameter_handler.h>
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <cstdlib>
// #include <typeinfo>  // for typeinfo :-)

// custom modules
#include <BitMap.hpp>


namespace input_data {
  using namespace dealii;

  template<typename T>
  std::vector<T> parse_string_list(std::string list_string,
                                   std::string delimiter=",")
  {
    std::vector<T> list;
    T item;
    if (list_string.size() == 0) return list;
    std::vector<std::string> strs;
    boost::split(strs, list_string, boost::is_any_of(delimiter));

    for (const auto &string_item : strs)
    {
      std::stringstream convert(string_item);
      convert >> item;
      list.push_back(item);
    }
    return list;
  }


  template<>
  std::vector<bool> parse_string_list(std::string list_string,
                                      std::string delimiter)
  {
    // std::cout << "Parsing bool list" << std::endl;
    std::vector<bool> list;
    bool item;
    if (list_string.size() == 0) return list;
    std::vector<std::string> strs;
    boost::split(strs, list_string, boost::is_any_of(delimiter));

    for (auto &string_item : strs)
    {
      std::istringstream is(string_item);
      is >> std::boolalpha >> item;
      // std::cout << std::endl << string_item << std::endl;
      // std::cout << item << std::endl;
      list.push_back(item);
    }
    return list;
  }


  template <int dim>
  std::vector< Point<dim> > parse_point_list(const std::string &str)
  {
    // std::cout << str << std::endl;
    std::vector< Point<dim> > points;
    // std::vector<std::string> point_strings;
    // int point_index = 0;
    unsigned int i = 0;
    // loop over symbols and get strings surrounded by ()
    while (i < str.size())
    {
      if (str.compare(i, 1, "(") == 0)  // if str[i] == "(" -> begin point
        {
          std::string tmp;
          while (i < str.size())
          {
            i++;

            if (str.compare(i, 1, ")") != 0)
              tmp.push_back(str[i]);
            else
              break;
          }
          // Add point
          std::vector<double> coords = parse_string_list<double>(tmp);
          Point<dim> point;
          for (int p=0; p<dim; ++p)
            point(p) = coords[p];
          points.push_back(point);
        }
        i++;
    }

    return points;
  }  // eom


  std::vector<std::string> parse_pathentheses_list(const std::string &str)
  {
    std::vector<std::string> result;
    unsigned int i = 0;
    // loop over symbols and get strings surrounded by ()
    while (i < str.size())
    {
      if (str.compare(i, 1, "(") == 0)  // if str[i] == "(" -> begin point
      {
        std::string tmp;
        while (i < str.size())
        {
          i++;

          if (str.compare(i, 1, ")") != 0)
            tmp.push_back(str[i]);
          else
            break;
       }  // end insize parentheses
       // add what's inside parantheses
       result.push_back(tmp);
      }
      i++;
    }
    return result;
  }



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
      *get_poisson_ratio,
      *get_lame_constant,
      *get_shear_modulus,
      *get_fracture_toughness;

  };

  template <int dim>
  class NotchedTestData : public PhaseFieldData<dim>
  {
  public:
    NotchedTestData();
    void compute_mesh_dependent_parameters(double);
    void read_input_file(std::string);
    double get_time_step(const double);

  public:
    // BC's
    std::vector<double> displacement_boundary_velocities;
    std::vector< Point<dim> > displacement_points;
    std::vector<int>          displacement_point_components;
    std::vector<double>       displacement_point_velocities;
    std::vector<bool>         constraint_point_phase_field;

    // Solver
    double newton_tolerance;
    int max_newton_iter;
    double t_max, minimum_time_step;

    // Mesh
    int initial_refinement_level, n_prerefinement_steps, n_adaptive_steps;
    double phi_refinement_value;
    std::vector<std::pair<double,double>> local_prerefinement_region;
    std::string mesh_file_name;
    // postprocessing
    std::vector<std::string> postprocessing_function_names;

    // this is a container for postprocessing function arguments
    // they can be strings, ints, or doubles
    std::vector< std::vector< boost::variant<int, double, std::string> > >
      postprocessing_function_args;


  private:
    void declare_parameters();
    void compute_runtime_parameters();
    void assign_parameters();
    void check_input();

    ParameterHandler prm;
    bool uniform_fracture_toughness, uniform_young_modulus;
    // Properties
    double fracture_toughness_constant;
    std::pair<double,double> fracture_toughness_limits, young_modulus_limits;
    std::pair<double,double> regularization_epsilon_coefficients;
    // Bitmap
    std::string bitmap_file_name;
    std::vector< std::pair<double,double> > bitmap_range;
    std::map<double, double> timestep_table;
  };

  template <int dim>
  NotchedTestData<dim>::NotchedTestData()
  {
    declare_parameters();

  }  // EOM


  template <int dim>
  void NotchedTestData<dim>::compute_runtime_parameters()
  {
    double E = this->young_modulus;
    double nu = this->poisson_ratio;
    this->lame_constant = E*nu/((1. + nu)*(1. - 2.*nu));
    this->shear_modulus = 0.5*E/(1. + nu);

    if (uniform_fracture_toughness)
      this->get_fracture_toughness =
        new ConstantFunction<dim>(fracture_toughness_constant, 1);
    else
      this->get_fracture_toughness =
        new BitMap::BitMapFunction<dim>(
          bitmap_file_name,
          bitmap_range[0].first, bitmap_range[0].second,
          bitmap_range[1].first, bitmap_range[1].second,
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

  }  // eom


  template <int dim>
  void NotchedTestData<dim>::declare_parameters()
  {
    { // Mesh
      prm.enter_subsection("Mesh");
      prm.declare_entry("Mesh file", "", Patterns::Anything());
      prm.declare_entry("Initial global refinement steps", "0", Patterns::Integer(0, 100));
      prm.declare_entry("Local refinement steps", "0", Patterns::Integer(0, 100));
      prm.declare_entry("Adaptive steps", "0", Patterns::Integer(0, 100));
      prm.declare_entry("Adaptive phi value", "0", Patterns::Double(0, 1));
      prm.declare_entry("Local refinement region", "",
                        Patterns::List(Patterns::Double()));
      prm.leave_subsection();
    }
    { // BC's
      prm.enter_subsection("Boundary conditions");
      prm.declare_entry("Displacement boundary labels", "",
                    Patterns::List(Patterns::Integer()));
      prm.declare_entry("Displacement boundary components", "",
                    Patterns::List(Patterns::Integer(0, dim-1)));
      prm.declare_entry("Displacement boundary velocities", "",
                      Patterns::List(Patterns::Double()));
      prm.declare_entry("Displacement points", "", Patterns::Anything());
      prm.declare_entry("Displacement point components", "",
                    Patterns::List(Patterns::Integer(0, dim-1)));
      prm.declare_entry("Displacement point velocities", "",
                    Patterns::List(Patterns::Double()));
      prm.declare_entry("Constraint point phase field", "",
                        Patterns::List(Patterns::Bool()));
      prm.leave_subsection();
    }
    { // equation data
      prm.enter_subsection("Equation data");
      // Constant parameters
      prm.declare_entry("Young modulus", "1", Patterns::Double());
      prm.declare_entry("Poisson ratio", "0.3", Patterns::Double(0, 0.5));
      prm.declare_entry("Fracture toughness", "1e10", Patterns::Double());
      prm.declare_entry("Regularization kappa", "0", Patterns::Double());
      prm.declare_entry("Regularization epsilon", "2, 1",
                        Patterns::List(Patterns::Double()));
      prm.declare_entry("Penalization c", "10", Patterns::Double());
      // Uniformity boolian
      prm.declare_entry("Uniform Young modulus", "true", Patterns::Bool());
      prm.declare_entry("Uniform Poisson ratio", "true", Patterns::Bool());
      prm.declare_entry("Uniform fracture toughness", "true", Patterns::Bool());
      // Heterogeneity limits
      prm.declare_entry("Young modulus range", "",
                        Patterns::List(Patterns::Double()));
      prm.declare_entry("Poisson ratio range", "",
                        Patterns::List(Patterns::Double(1e-3, 0.5)));
      prm.declare_entry("Fracture toughness range", "",
                        Patterns::List(Patterns::Double(0, 1e4)));
      // Files with homogeneous data
      prm.declare_entry("Bitmap file", "", Patterns::Anything());
      prm.declare_entry("Bitmap range", "0, 1, 0, 1",
                        Patterns::List(Patterns::Double()));
      prm.leave_subsection();
    }
    { // Solver
      prm.enter_subsection("Solver");
      prm.declare_entry("T max", "1", Patterns::Double());
      prm.declare_entry("Time stepping table", "(0, 1e-5)", Patterns::Anything());
      prm.declare_entry("Minimum time step", "1e-9", Patterns::Double());
      prm.declare_entry("Newton tolerance", "1e-9", Patterns::Double());
      prm.declare_entry("Max Newton steps", "20", Patterns::Integer());
      prm.leave_subsection();
    }
    {
      prm.enter_subsection("Postprocessing");
      prm.declare_entry("Functions", "", Patterns::Anything());
      prm.declare_entry("Arguments", "", Patterns::Anything());
      prm.leave_subsection();
    }
  }  // eom


  template <int dim>
  void NotchedTestData<dim>::read_input_file(std::string file_name)
  {
    prm.read_input(file_name);
    prm.print_parameters(std::cout, ParameterHandler::Text);
    assign_parameters();
    compute_runtime_parameters();
    check_input();
  }  // eom


template <int dim>
void NotchedTestData<dim>::check_input()
{
  if (!this->uniform_fracture_toughness)
    {
      AssertThrow(fracture_toughness_limits.first > 0,
        ExcMessage("Fracture toughness should be > 0"));
      AssertThrow(fracture_toughness_limits.second > 0,
        ExcMessage("Fracture toughness should be > 0"));
    }
   else
      AssertThrow(fracture_toughness_constant > 0,
        ExcMessage("Fracture toughness should be > 0"));

  if (!this->uniform_young_modulus)
    {
      AssertThrow(this->young_modulus_limits.first > 0,
        ExcMessage("Young's modulus should be > 0"));
      AssertThrow(this->young_modulus_limits.second > 0,
        ExcMessage("Young modulus should be > 0"));
    }
   else
      AssertThrow(this->young_modulus > 0,
        ExcMessage("Young modulus should be > 0"));

  if (!this->uniform_young_modulus
      &&
      !this->uniform_fracture_toughness)
  {
    AssertThrow(this->bitmap_range[0].first < this->bitmap_range[0].second,
                ExcMessage("Bitmap range is incorrect"));
    AssertThrow(this->bitmap_range[1].first < this->bitmap_range[1].second,
                ExcMessage("Bitmap range is incorrect"));
  }

AssertThrow(this->postprocessing_function_names.size() ==
            this->postprocessing_function_args.size(),
            ExcMessage("Postprocessing input incorrect"));

}

template <int dim>
void NotchedTestData<dim>::assign_parameters()
{
  { // Mesh
    prm.enter_subsection("Mesh");
    mesh_file_name = prm.get("Mesh file");
    initial_refinement_level = prm.get_integer("Initial global refinement steps");
    n_prerefinement_steps = prm.get_integer("Local refinement steps");
    n_adaptive_steps = prm.get_integer("Adaptive steps");
    phi_refinement_value = prm.get_double("Adaptive phi value");
    std::vector<double> tmp =
      parse_string_list<double>(prm.get("Local refinement region"));
    local_prerefinement_region.resize(dim);
    AssertThrow(tmp.size() == 2*dim,
                ExcMessage("Wrong entry in Local refinement region"));
    local_prerefinement_region[0].first = tmp[0];
    local_prerefinement_region[0].second = tmp[1];
    local_prerefinement_region[1].first = tmp[2];
    local_prerefinement_region[1].second = tmp[3];
    prm.leave_subsection();
  }
  { // Boundary conditions
    prm.enter_subsection("Boundary conditions");
    this->displacement_boundary_labels =
      parse_string_list<int>(prm.get("Displacement boundary labels"));
    this->displacement_boundary_components =
      parse_string_list<int>(prm.get("Displacement boundary components"));
    this->displacement_boundary_velocities =
      parse_string_list<double>(prm.get("Displacement boundary velocities"));
    this->displacement_points =
      parse_point_list<dim>(prm.get("Displacement points"));
    this->displacement_point_components =
      parse_string_list<int>(prm.get("Displacement point components"));
    this->displacement_point_velocities =
      parse_string_list<double>(prm.get("Displacement point velocities"));
    this->constraint_point_phase_field =
      parse_string_list<bool>(prm.get("Constraint point phase field"));
    prm.leave_subsection();
  }
  {  // Equation data
    prm.enter_subsection("Equation data");
    // Uniformity boolean
    this->uniform_fracture_toughness = prm.get_bool("Uniform fracture toughness");
    this->uniform_young_modulus = prm.get_bool("Uniform Young modulus");

    // coefficients that are either constant or mapped
    if (this->uniform_fracture_toughness)
      this->fracture_toughness_constant = prm.get_double("Fracture toughness");
    else
    {
      std::vector<double> tmp;
      tmp.resize(2);
      tmp = parse_string_list<double>(prm.get("Fracture toughness range"));
      this->fracture_toughness_limits.first = tmp[0];
      this->fracture_toughness_limits.second = tmp[1];
    }

    if (this->uniform_young_modulus)
      this->young_modulus = prm.get_double("Young modulus");
    else
    {
      std::vector<double> tmp;
      tmp.resize(2);
      tmp = parse_string_list<double>(prm.get("Young modulus range"));
      this->young_modulus_limits.first = tmp[0];
      this->young_modulus_limits.second = tmp[1];
    }

    this->poisson_ratio = prm.get_double("Poisson ratio");
    this->regularization_parameter_kappa = prm.get_double("Regularization kappa");
    this->penalty_parameter = prm.get_double("Penalization c");
    std::vector<double> tmp =
      parse_string_list<double>(prm.get("Regularization epsilon"));
    regularization_epsilon_coefficients.first = tmp[0];
    regularization_epsilon_coefficients.second = tmp[1];
    // Ranges
    // tmp.clear();
    // Bitmap file
    bitmap_file_name = prm.get("Bitmap file");
    std::vector<double> tmp1 =
      parse_string_list<double>(prm.get("Bitmap range"));
    bitmap_range.resize(dim);
    // std::cout << tmp1.size() << std::endl;
    if (tmp1.size() > 0)
      for (int i=0; i<dim; ++i)
      {
        bitmap_range[i].first = tmp1[2*i];
        bitmap_range[i].second = tmp1[2*i + 1];
      }
    prm.leave_subsection();
  }
  { // Solver
    prm.enter_subsection("Solver");
    this->t_max = prm.get_double("T max");
    std::vector<Point<2> > tmp =
        parse_point_list<2>(prm.get("Time stepping table"));
    for (const auto &row : tmp)
      this->timestep_table[row[0]] = row[1];

    this->minimum_time_step = prm.get_double("Minimum time step");
    this->newton_tolerance = prm.get_double("Newton tolerance");
    this->max_newton_iter = prm.get_integer("Max Newton steps");
    prm.leave_subsection();
  }
  { // Postprocessing
    prm.enter_subsection("Postprocessing");
    postprocessing_function_names =
        parse_string_list<std::string>(prm.get("Functions"));

    std::vector<std::string> tmp =
        parse_pathentheses_list(prm.get("Arguments"));

    AssertThrow(tmp.size() == postprocessing_function_names.size(),
                ExcMessage("Number of argument groups needs to match number of functions"));

    // loop through function names and assign the appropriate parameters
    for (unsigned int i=0; i<postprocessing_function_names.size(); i++)
    {
      std::vector<std::string> string_vector;
      std::vector< boost::variant<int, double, std::string> > args;
      boost::split(string_vector, tmp[i], boost::is_any_of(","));

      unsigned int l = postprocessing_function_names[i].size();
      if (postprocessing_function_names[i].compare(0, l, "boundary_load") == 0)
      { // this function takes only a list of integers
        for (const auto &arg : string_vector)
        {
          std::stringstream convert(arg);
          int item;
          convert >> item;
          args.push_back(item);
        }
      }
      postprocessing_function_args.push_back(args);
    }

    // this demonstrates how to get the types stored in a boost::variant variable
    // for (unsigned int i=0; i<postprocessing_function_args[0].size(); i++)
    // {
    //   int t = postprocessing_function_args[0][i].which();
    //   // std::cout << postprocessing_function_args[0][i].which() << std::endl;
    //   if (t == 0)
    //     std:: cout << "type: int" << "\t";
    //   else if (t == 1)
    //     std:: cout << "type: double" << "\t";
    //   else
    //     std:: cout << "type: std::string" << "\t";
    //
    //   std::cout << "value: "
    //             << postprocessing_function_args[0][i] << "\t"
    //             << std::endl;
    // }

    prm.leave_subsection();
  }

}  // eom

  template <int dim>
  double NotchedTestData<dim>::get_time_step(const double time)
  /* get value of the time step from the time-stepping table */
  {
    double time_step = timestep_table.rbegin()->second;
    for (const auto &it : timestep_table)
    {
      if (time >= it.first)
        time_step = it.second;
      else
        break;
    }

    return time_step;
  }

}  // end of namespace
