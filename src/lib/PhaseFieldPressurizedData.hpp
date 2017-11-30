#pragma once
#include <deal.II/base/function_parser.h>

// #include <InputData.hpp>
#include <SinglePhaseData.hpp>

namespace InputData
{
	using namespace dealii;


	template <int dim>
	// class PhaseFieldPressurizedData : public PhaseFieldSolidData<dim>
	class PhaseFieldPressurizedData : public SinglePhaseData<dim>
	{
	public:
		PhaseFieldPressurizedData(ConditionalOStream &pcout_);
    void read_input_file(std::string);
	private:
		void assign_parameters();
		void declare_parameters();

	// private:
		// PhaseFieldSolidData<dim>
	public:
		std::vector<std::vector<double> > defect_coordinates;
		std::vector<double> 							displacement_boundary_values;

		// depends only on time and constants
		FunctionParser<1>               pressure_function;
	};


	template <int dim>
	PhaseFieldPressurizedData<dim>::
	PhaseFieldPressurizedData(ConditionalOStream &pcout_)
	:
	// PhaseFieldSolidData<dim>(pcout_),
    SinglePhaseData<dim>(pcout_),
	pressure_function(1)
	{
		declare_parameters();
	}  // eom


	template <int dim> void
	PhaseFieldPressurizedData<dim>::declare_parameters()
  {
		// pcout << "Declaring shit 1" << std::endl;
    { // Mesh
      this->prm.enter_subsection("Mesh");
      this->prm.declare_entry("Mesh file", "", Patterns::Anything());
      this->prm.declare_entry("Initial global refinement steps", "0", Patterns::Integer(0, 100));
      this->prm.declare_entry("Local refinement steps", "0", Patterns::Integer(0, 100));
      this->prm.declare_entry("Adaptive steps", "0", Patterns::Integer(0, 100));
      this->prm.declare_entry("Adaptive phi value", "0", Patterns::Double(0, 1));
      this->prm.declare_entry("Local refinement region", "",
                        Patterns::List(Patterns::Double()));
      this->prm.leave_subsection();
    }
    { // BC's
      this->prm.enter_subsection("Boundary conditions");
      this->prm.declare_entry("Displacement boundary labels", "",
                    Patterns::List(Patterns::Integer()));
      this->prm.declare_entry("Displacement boundary components", "",
                    Patterns::List(Patterns::Integer(0, dim-1)));
      this->prm.declare_entry("Displacement boundary values", "",
                      Patterns::List(Patterns::Double()));
      this->prm.leave_subsection();
    }
		{ // IC's
      this->prm.enter_subsection("Initial conditions");
      this->prm.declare_entry("Defects", "", Patterns::Anything());
      this->prm.leave_subsection();
		}
    { // equation data
      this->prm.enter_subsection("Equation data");
      // Constant parameters
      this->prm.declare_entry("Pressure", "0", Patterns::Anything());
      this->prm.declare_entry("Young modulus", "1", Patterns::Double());
      this->prm.declare_entry("Poisson ratio", "0.3", Patterns::Double(0, 0.5));
			this->prm.declare_entry("Biot coefficient", "0.0",
															Patterns::Double(0.0, 1.0));
      this->prm.declare_entry("Fracture toughness", "1e10", Patterns::Double());
      this->prm.declare_entry("Regularization kappa", "0", Patterns::Double());
      this->prm.declare_entry("Regularization epsilon", "2, 1",
                        Patterns::List(Patterns::Double()));
      this->prm.declare_entry("Penalization c", "10", Patterns::Double());
      // Uniformity boolian
      this->prm.declare_entry("Uniform Young modulus", "true", Patterns::Bool());
      this->prm.declare_entry("Uniform Poisson ratio", "true", Patterns::Bool());
      this->prm.declare_entry("Uniform fracture toughness", "true", Patterns::Bool());
      // Heterogeneity limits
      this->prm.declare_entry("Young modulus range", "",
                        Patterns::List(Patterns::Double()));
      this->prm.declare_entry("Poisson ratio range", "",
                        Patterns::List(Patterns::Double(1e-3, 0.5)));
      this->prm.declare_entry("Fracture toughness range", "",
                        Patterns::List(Patterns::Double(0, 1e4)));
      // Files with homogeneous data
      this->prm.declare_entry("Bitmap file", "", Patterns::Anything());
      this->prm.declare_entry("Bitmap range", "0, 1, 0, 1",
                        Patterns::List(Patterns::Double()));
      this->prm.leave_subsection();
    }
    { // Solver
      this->prm.enter_subsection("Solver");
      this->prm.declare_entry("T max", "1", Patterns::Double());
      this->prm.declare_entry("Time stepping table", "(0, 1e-5)", Patterns::Anything());
      this->prm.declare_entry("Minimum time step", "1e-9", Patterns::Double());
      this->prm.declare_entry("Newton tolerance", "1e-9", Patterns::Double());
      this->prm.declare_entry("Max Newton steps", "20", Patterns::Integer());
      this->prm.declare_entry("Level set constant", "0.1", Patterns::Double());
      this->prm.declare_entry("Penalty theta", "1000", Patterns::Double());
      this->prm.leave_subsection();
    }
    {
      this->prm.enter_subsection("Postprocessing");
      this->prm.declare_entry("Functions", "", Patterns::Anything());
      this->prm.declare_entry("Arguments", "", Patterns::Anything());
      this->prm.leave_subsection();
    }
  }  // eom


	template <int dim>
  void PhaseFieldPressurizedData<dim>::read_input_file(std::string file_name)
  {
    this->prm.read_input(file_name);
    assign_parameters();
    this->compute_runtime_parameters();
    // check_input();
  }  // eom


	template <int dim>
	void PhaseFieldPressurizedData<dim>::assign_parameters()
	{
	  { // Mesh
	    this->prm.enter_subsection("Mesh");
	    this->mesh_file_name = this->prm.get("Mesh file");
	    this->initial_refinement_level = this->prm.get_integer("Initial global refinement steps");
	    this->n_prerefinement_steps = this->prm.get_integer("Local refinement steps");
	    this->n_adaptive_steps = this->prm.get_integer("Adaptive steps");
	    this->phi_refinement_value = this->prm.get_double("Adaptive phi value");
	    std::vector<double> tmp =
	      Parsers::parse_string_list<double>(this->prm.get("Local refinement region"));
	    this->local_prerefinement_region.resize(dim);
	    AssertThrow(tmp.size() == 2*dim,
	                ExcMessage("Wrong entry in Local refinement region"));
	    this->local_prerefinement_region[0].first = tmp[0];
	    this->local_prerefinement_region[0].second = tmp[1];
	    this->local_prerefinement_region[1].first = tmp[2];
	    this->local_prerefinement_region[1].second = tmp[3];
	    this->prm.leave_subsection();
	  }
	  { // Boundary conditions
	    this->prm.enter_subsection("Boundary conditions");
	    this->displacement_boundary_labels =
	      Parsers::parse_string_list<int>(this->prm.get("Displacement boundary labels"));
	    this->displacement_boundary_components =
	      Parsers::parse_string_list<int>(this->prm.get("Displacement boundary components"));
	    this->displacement_boundary_values =
	      Parsers::parse_string_list<double>(this->prm.get("Displacement boundary values"));
	    this->prm.leave_subsection();
	  }
		{ // initial conditions
	    this->prm.enter_subsection("Initial conditions");
			std::vector<std::string> tmp_vector;
			tmp_vector = Parsers::parse_pathentheses_list(this->prm.get("Defects"));
			// for (auto &item: tmp_vector)
			// 	this->pcout << item << std::endl;
			for (unsigned int i = 0; i<tmp_vector.size(); ++i)
			{
				std::vector<double> coords =
					Parsers::parse_string_list<double>(tmp_vector[i]);
				AssertThrow(coords.size() == 2*dim, ExcMessage("Error in Defects"));
				defect_coordinates.push_back(coords);
					// for (auto &item: coords)
					// 	this->pcout << item << std::endl;
			}
	    this->prm.leave_subsection();
		}
	  {  // Equation data
	    this->prm.enter_subsection("Equation data");
			// Pressure expression
			std::string pressure_string = this->prm.get("Pressure");
			std::string variables_for_pressure = "time";
			std::map<std::string, double> constants_for_pressure;
			pressure_function.initialize(variables_for_pressure,
																	 pressure_string,
																	 constants_for_pressure);
      // std::cout << "Pressure string "
			//           << pressure_string
			// 					<< std::endl;
      // std::cout << "Pressure "
			//           << pressure_function.value(Point<1>(0.0))
			// 					<< std::endl;
	    // Uniformity boolean
	    this->uniform_fracture_toughness = this->prm.get_bool("Uniform fracture toughness");
	    this->uniform_young_modulus = this->prm.get_bool("Uniform Young modulus");

	    // coefficients that are either constant or mapped
	    if (this->uniform_fracture_toughness)
	      this->fracture_toughness_constant = this->prm.get_double("Fracture toughness");
	    else
	    {
	      std::vector<double> tmp;
	      tmp.resize(2);
	      tmp = Parsers::parse_string_list<double>(this->prm.get("Fracture toughness range"));
	      this->fracture_toughness_limits.first = tmp[0];
	      this->fracture_toughness_limits.second = tmp[1];
	    }

	    if (this->uniform_young_modulus)
	      this->young_modulus = this->prm.get_double("Young modulus");
	    else
	    {
	      std::vector<double> tmp;
	      tmp.resize(2);
	      tmp = Parsers::parse_string_list<double>(this->prm.get("Young modulus range"));
	      this->young_modulus_limits.first = tmp[0];
	      this->young_modulus_limits.second = tmp[1];
	    }

	    this->poisson_ratio = this->prm.get_double("Poisson ratio");
			this->biot_coef = this->prm.get_double("Biot coefficient");
	    this->regularization_parameter_kappa = this->prm.get_double("Regularization kappa");
	    this->penalty_parameter = this->prm.get_double("Penalization c");
	    std::vector<double> tmp =
	      Parsers::parse_string_list<double>(this->prm.get("Regularization epsilon"));
	    this->regularization_epsilon_coefficients.first = tmp[0];
	    this->regularization_epsilon_coefficients.second = tmp[1];
	    // Ranges
	    // tmp.clear();
	    // Bitmap file
	    this->bitmap_file_name = this->prm.get("Bitmap file");
	    std::vector<double> tmp1 =
	      Parsers::parse_string_list<double>(this->prm.get("Bitmap range"));
	    this->bitmap_range.resize(dim);
	    // std::cout << tmp1.size() << std::endl;
	    if (tmp1.size() > 0)
	      for (int i=0; i<dim; ++i)
	      {
	        this->bitmap_range[i].first = tmp1[2*i];
	        this->bitmap_range[i].second = tmp1[2*i + 1];
	      }
	    this->prm.leave_subsection();
	  }
	  { // Solver
	    this->prm.enter_subsection("Solver");
	    this->t_max = this->prm.get_double("T max");
	    std::vector<Point<2> > tmp =
	        Parsers::parse_point_list<2>(this->prm.get("Time stepping table"));
	    for (const auto &row : tmp)
	      this->timestep_table[row[0]] = row[1];

	    this->minimum_time_step = this->prm.get_double("Minimum time step");
	    this->newton_tolerance = this->prm.get_double("Newton tolerance");
	    this->max_newton_iter = this->prm.get_integer("Max Newton steps");
      this->constant_level_set = this->prm.get_double("Level set constant");
      AssertThrow(this->constant_level_set < this->phi_refinement_value,
                  ExcMessage("Level set constant should be > phi refinement constant"));
      this->penalty_theta = this->prm.get_integer("Penalty theta");
	    this->prm.leave_subsection();
	  }
	  { // Postprocessing
	    this->prm.enter_subsection("Postprocessing");
	    this->postprocessing_function_names =
	        Parsers::parse_string_list<std::string>(this->prm.get("Functions"));

	    std::vector<std::string> tmp =
	        Parsers::parse_pathentheses_list(this->prm.get("Arguments"));

	    AssertThrow(tmp.size() == this->postprocessing_function_names.size(),
	                ExcMessage("Number of argument groups needs to match number of functions"));

	    // loop through function names and assign the appropriate parameters
	    for (unsigned int i=0; i<this->postprocessing_function_names.size(); i++)
	    {
	      std::vector<std::string> string_vector;
	      std::vector< boost::variant<int, double, std::string> > args;
	      boost::split(string_vector, tmp[i], boost::is_any_of(","));
	      unsigned int l = this->postprocessing_function_names[i].size();

				// boundary load arguments
	      if (this->postprocessing_function_names[i].compare(0, l, "boundary_load") == 0)
	      { // this function takes only a list of integers
	        for (const auto &arg : string_vector)
	        {
						int item = Parsers::convert<int>(arg);
	          args.push_back(item);
	        }
	      }  // end boundary load

				// crack opening displacements arguments
	      if (this->postprocessing_function_names[i].compare(0, l, "COD") == 0)
				{ // double coord_start, double coord_end, int n_lines, int direction<dim
					AssertThrow(string_vector.size() == 4,
				              ExcMessage("Number of arguments in COD is wrong"));
					{ // convert first argument to double
	          // std::stringstream convert(string_vector[0]);
	          // double double_item;
	          // convert >> double_item;
						double item = Parsers::convert<double>(string_vector[0]);
	          args.push_back(item);
					}
					{ // convert second argument to double
	          // std::stringstream convert(string_vector[1]);
	          // double double_item;
	          // convert >> double_item;
						double item = Parsers::convert<double>(string_vector[1]);
	          args.push_back(item);
					}
					{ // convert third argument to int
	          // std::stringstream convert(string_vector[2]);
	          // int int_item;
	          // convert >> int_item;
						int item = Parsers::convert<int>(string_vector[2]);
	          args.push_back(item);
					}
					{ // convert fourth argument to int
						int item = Parsers::convert<int>(string_vector[3]);
						AssertThrow(item < dim,
						            ExcMessage("COD direction parameter should be < dim"));
	          args.push_back(item);
						// std::stringstream convert
					}
				}

	      this->postprocessing_function_args.push_back(args);
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

	    this->prm.leave_subsection();
	  }

	}  // eom

}  // end of namespace
