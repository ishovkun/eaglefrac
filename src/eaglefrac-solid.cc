#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_in.h>

#include <limits>       // std::numeric_limits

#include <boost/filesystem.hpp>
#include <boost/variant/get.hpp>

// Custom modules
#include <PhaseFieldSolver.hpp>
#include <Postprocessing.hpp>
#include <InputData.hpp>
#include <Mesher.hpp>


namespace EagleFrac
{
  using namespace dealii;


  template <int dim>
  class PDSSolid
  {
  public:
    PDSSolid(const std::string &input_file_name_);
    ~PDSSolid();

    void run();

  private:
    void create_mesh();
    void read_mesh();
    void setup_dofs();
    void impose_displacement_on_solution(double time);
    void output_results(int time_step_number); //const;
    void refine_mesh();
    void execute_postprocessing(const double time);
    void exectute_adaptive_refinement();
    void prepare_output_directories();

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    ConditionalOStream pcout;
    TimerOutput computing_timer;

    InputData::PhaseFieldSolidData<dim> data;
    PhaseField::PhaseFieldSolver<dim> phase_field_solver;
    std::string input_file_name, case_name;
  };


  template <int dim>
  PDSSolid<dim>::PDSSolid(const std::string &input_file_name_)
    :
    mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing
                  (Triangulation<dim>::smoothing_on_refinement |
                   Triangulation<dim>::smoothing_on_coarsening)),
    pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator)
           == 0)),
    computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times),
		data(pcout),
    phase_field_solver(mpi_communicator,
                       triangulation, data,
                       pcout, computing_timer),
    input_file_name(input_file_name_)
  {}


  template <int dim>
  PDSSolid<dim>::~PDSSolid()
  {}

  template <int dim>
  void PDSSolid<dim>::read_mesh()
  {
    GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f(data.mesh_file_name);
    // typename GridIn<dim>::Format format = GridIn<dim>::ucd;
    // gridin.read(f, format);
	  gridin.read_msh(f);
  }

  template <int dim>
  void PDSSolid<dim>::impose_displacement_on_solution(double time)
  {
    int n_displacement_conditions = data.displacement_boundary_labels.size();
    std::vector<double> displacement_values(n_displacement_conditions);
    for (int i=0; i<n_displacement_conditions; ++i)
      displacement_values[i] = data.displacement_boundary_velocities[i]*time;

    int n_displacement_node_conditions = data.displacement_points.size();
    std::vector<double> displacement_point_values(n_displacement_node_conditions);
    for (int i=0; i<n_displacement_node_conditions; ++i)
    {
      displacement_point_values[i] = data.displacement_point_velocities[i]*time;
    }

    phase_field_solver.impose_displacement(data.displacement_boundary_labels,
                                           data.displacement_boundary_components,
                                           displacement_values,
                                           data.displacement_points,
                                           data.displacement_point_components,
                                           displacement_point_values,
                                           data.constraint_point_phase_field);
  }  // eom


  template <int dim>
  void PDSSolid<dim>::exectute_adaptive_refinement()
  {
    phase_field_solver.relevant_solution = phase_field_solver.solution;
    std::vector<const TrilinosWrappers::MPI::BlockVector *> tmp(3);
    tmp[0] = &phase_field_solver.relevant_solution;
    tmp[1] = &phase_field_solver.old_solution;
    tmp[2] = &phase_field_solver.old_old_solution;

    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::BlockVector>
        solution_transfer(phase_field_solver.dof_handler);

    solution_transfer.prepare_for_coarsening_and_refinement(tmp);
    triangulation.execute_coarsening_and_refinement();

    phase_field_solver.setup_dofs();

    TrilinosWrappers::MPI::BlockVector
      tmp_owned1(phase_field_solver.owned_partitioning, mpi_communicator),
      tmp_owned2(phase_field_solver.owned_partitioning, mpi_communicator);

    std::vector<TrilinosWrappers::MPI::BlockVector *> tmp1(3);
    tmp1[0] = &phase_field_solver.solution;
    tmp1[1] = &tmp_owned1;
    tmp1[2] = &tmp_owned2;

    solution_transfer.interpolate(tmp1);
    phase_field_solver.old_solution = tmp_owned1;
    phase_field_solver.old_old_solution = tmp_owned2;

  }  // eom


  template <int dim>
  void PDSSolid<dim>::prepare_output_directories()
  {
    size_t path_index = input_file_name.find_last_of("/");

    size_t extension_index = input_file_name.substr(path_index).rfind(".");
    // if (extension_index != string::npos)
    //   extension = filename.substr(pos+1);

    case_name = input_file_name.substr(path_index+1, extension_index-1);

    boost::filesystem::path output_directory_path("./" + case_name);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      if (!boost::filesystem::is_directory(output_directory_path))
      {
          pcout << "Output folder not found\n"
                << "Creating directory: ";
          if (boost::filesystem::create_directory(output_directory_path))
            std::cout << "Success" << std::endl;
      }
      else
      { // remove everything from this directory
        pcout << "Folder exists: cleaning folder: ";
        boost::filesystem::remove_all(output_directory_path);
        if (boost::filesystem::create_directory(output_directory_path))
           std::cout << "Success" << std::endl;
      }
    }
  }  // eom

  template <int dim>
  void PDSSolid<dim>::run()
  {
    data.read_input_file(input_file_name);
    read_mesh();

    // debug input
    // Point<dim> p(3e-3, 0.01), p1(3e-3, 0.01905);
    // pcout << "toughness " << data.get_fracture_toughness->value(p, 0) << std::endl;
    // pcout << "toughness1 " << data.get_fracture_toughness->value(p1, 0) << std::endl;
		// pcout << data.lame_constant << std::endl;
		// pcout << data.shear_modulus << std::endl;
		// pcout << data.displacement_boundary_velocities[0]*1 << std::endl;
    // return;

    prepare_output_directories();

    // compute_runtime_parameters
    double minimum_mesh_size = Mesher::compute_minimum_mesh_size(triangulation,
                                                                 mpi_communicator);
    const int max_refinement_level =
        data.n_prerefinement_steps
      + data.initial_refinement_level
      + data.n_adaptive_steps;

    minimum_mesh_size /= std::pow(2, max_refinement_level);
    data.compute_mesh_dependent_parameters(minimum_mesh_size);

    pcout << "min mesh size " << minimum_mesh_size << std::endl;

    // local prerefinement
    triangulation.refine_global(data.initial_refinement_level);
    Mesher::refine_region(triangulation,
                          data.local_prerefinement_region,
                          data.n_prerefinement_steps);

    phase_field_solver.setup_dofs();

    // set initial phase-field to 1
    phase_field_solver.solution.block(1) = 1;
    phase_field_solver.old_solution.block(1) = 1;

    double time = 0;
    double time_step = data.get_time_step(time);
    double old_time_step = time_step;
    int time_step_number = 0;

    while(time < data.t_max)
    {
      time_step = data.get_time_step(time);
      time += time_step;
      time_step_number++;

      phase_field_solver.update_old_solution();
      std::pair<double,double> time_steps = std::make_pair(time_step, old_time_step);

    redo_time_step:
      pcout << std::endl
            << "Time: "
            << std::defaultfloat << time
						<< "\tStep:"
						<< time_step
            << std::endl;

    	impose_displacement_on_solution(time);

      IndexSet old_active_set(phase_field_solver.active_set);

			pcout << "Iter #" << "\t"
			      << "ASet" << "\t"
			      << "error" << "\t\t"
			      << "GMRES" << "\t"
			      << "Search" << "\t"
						<< std::endl;
      int newton_step = 0;
      const double newton_tolerance = data.newton_tolerance;
      while (newton_step < data.max_newton_iter)
      {
        // pcout << "Newton iteration: " << newton_step << "\t";
				pcout << newton_step << "\t";

        double error;
        if (newton_step > 0)
        {
          phase_field_solver.
            compute_nonlinear_residual(phase_field_solver.solution,
                                       time_steps);

          phase_field_solver.compute_active_set(phase_field_solver.solution);
          phase_field_solver.all_constraints.set_zero(phase_field_solver.residual);
          error = phase_field_solver.residual_norm();

      	  pcout << phase_field_solver.active_set_size()
								<< "\t";
					std::cout.precision(3);
          pcout << std::scientific << error << "\t";
          std::cout.unsetf(std::ios_base::scientific);

          // break condition
          if (phase_field_solver.active_set_changed(old_active_set) &&
              error < newton_tolerance)
          {
            pcout << "Converged!" << std::endl;
            break;
          }

          old_active_set = phase_field_solver.active_set;
        }  // end first newton step condition

				std::pair<unsigned int, unsigned int> newton_step_results =
        	phase_field_solver.solve_newton_step(time_steps);

				pcout << newton_step_results.first << "\t";
				pcout << newton_step_results.second << "\t";

        // output_results(newton_step);
				// pcout << "here" << std::endl;
				newton_step++;
        pcout << std::endl;
      }  // End Newton iter

      // cut the time step if no convergence
      if (newton_step == data.max_newton_iter)
      {
        pcout << "Time step didn't converge: reducing to dt = "
              << time_step/10 << std::endl;
        if (time_step/10 < data.minimum_time_step)
        {
          pcout << "Time step too small: aborting" << std::endl;
          std::cout.unsetf(std::ios_base::scientific);
          throw SolverControl::NoConvergence(0, 0);
        }

        time -= time_step;
        time_step /= 10.0;
        time += time_step;
        phase_field_solver.solution = phase_field_solver.old_solution;
        phase_field_solver.use_old_time_step_phi = true;
        goto redo_time_step;
      }

      // do adaptive refinement if needed
      if (data.n_adaptive_steps > 0)
        if (Mesher::prepare_phase_field_refinement(phase_field_solver,
                                                   data.phi_refinement_value,
                                                   max_refinement_level))
        {
          pcout << std::endl
               << "Adapting mesh"
               << std::endl;
          exectute_adaptive_refinement();
          goto redo_time_step;
        }

      phase_field_solver.truncate_phase_field();
      output_results(time_step_number);
      execute_postprocessing(time);
      // return;

      phase_field_solver.use_old_time_step_phi = true;

      old_time_step = time_step;

      if (time >= data.t_max) break;
    }  // end time loop

    // pcout << std::fixed;
    // show timer table in default format
    std::cout.unsetf(std::ios_base::scientific);
  }  // EOM


  template <int dim>
  void PDSSolid<dim>::execute_postprocessing(const double time)
  {
    // just some commands so no compiler warnings
    for (unsigned int i=0; i<data.postprocessing_function_names.size(); i++)
    {
      unsigned int l = data.postprocessing_function_names[i].size();
      if (data.postprocessing_function_names[i].compare(0, l, "boundary_load") == 0)
      {
        int boundary_id =
            boost::get<int>(data.postprocessing_function_args[i][0]);
        Tensor<1,dim> load =
          Postprocessing::compute_boundary_load(phase_field_solver,
                                                data, boundary_id);
        // Sum write output
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream ff;
          ff.open("./" + case_name + "/boundary_load-" +
                  Utilities::int_to_string(boundary_id, 1) +
                  ".txt",
                  std::ios_base::app);
          ff << time << "\t"
             << load[0] << "\t"
             << load[1] << "\t"
             << std::endl;
        }
      }  // end boundary load

    }  // end loop over postprocessing functions
  }  // eom


  template <int dim>
  void PDSSolid<dim>::output_results(int time_step_number) // const
  {
    // Add data vectors to output
    std::vector<std::string> solution_names(dim, "displacement");
    solution_names.push_back("phase_field");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
      .push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    phase_field_solver.relevant_solution = phase_field_solver.solution;
    data_out.attach_dof_handler(phase_field_solver.dof_handler);
    data_out.add_data_vector(phase_field_solver.relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    // add active set
    data_out.add_data_vector(phase_field_solver.active_set, "active_set");
    // data_out.add_data_vector(phase_field_solver.residual, "residual");
    // Add domain ids
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    int n_time_step_digits = 3,
        n_processor_digits = 3;

    // Write output from local processors
    const std::string filename = ("./" + case_name + "/solution-" +
                                  Utilities::int_to_string(time_step_number,
                                                           n_time_step_digits) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(),
                                   n_processor_digits));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    // Write master file
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back("solution-" +
                              Utilities::int_to_string(time_step_number,
                                                       n_time_step_digits) +
                              "." +
                              Utilities::int_to_string (i,
                                                        n_processor_digits) +
                              ".vtu");
        std::ofstream
          master_output(("./" + case_name + "/solution-" +
                         Utilities::int_to_string(time_step_number, n_time_step_digits) +
                         ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
      }  // end master output
  } // EOM
}  // end of namespace


std::string parse_command_line(int argc, char *const *argv) {
  std::string filename;
  if (argc < 2) {
    std::cout << "specify the file name" << std::endl;
    exit(1);
  }

  std::list<std::string> args;
  for (int i=1; i<argc; ++i)
    args.push_back(argv[i]);

  int arg_number = 1;
  while (args.size()){
    std::cout << args.front() << std::endl;
    if (arg_number == 1)
      filename = args.front();
    args.pop_front();
    arg_number++;
  } // EO while args

  return filename;
}  // EOM

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string input_file_name = parse_command_line(argc, argv);
    EagleFrac::PDSSolid<2> problem(input_file_name);
    problem.run();
    return 0;
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
