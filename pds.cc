#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
// will probably remove it later
#include <deal.II/grid/grid_in.h>

// Custom modules
#include <PhaseFieldSolver.cc>


namespace pds_solid
{
  using namespace dealii;


  template <int dim>
  class PDSSolid
  {
  public:
    PDSSolid();
    ~PDSSolid();

    void run();

  private:
    void create_mesh();
    void read_mesh();
    void setup_dofs();
    void impose_displacement_on_solution(double time);
    void output_results(int time_step_number) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    ConditionalOStream pcout;
    TimerOutput computing_timer;

    input_data::NotchedTestData data;
    phase_field::PhaseFieldSolver<dim> phase_field_solver;
  };


  template <int dim>
  PDSSolid<dim>::PDSSolid()
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
    phase_field_solver(mpi_communicator,
                       triangulation, data,
                       pcout, computing_timer)
  {}


  template <int dim>
  PDSSolid<dim>::~PDSSolid()
  {}

  template <int dim>
  void PDSSolid<dim>::read_mesh()
  {
    GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f("notched.msh");
	  gridin.read_msh(f);
  }

  template <int dim>
  void PDSSolid<dim>::create_mesh()
  {
    double half_length = 0.005;
    GridGenerator::hyper_cube_slit(triangulation,
                                   0, half_length,
                                   /*colorize = */ false);
    const Point<dim> top(0, half_length);
    const Point<dim> bottom(0, -half_length);

    {// Assign boundary ids
      typename Triangulation<2>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

      for (; cell != endc; ++cell)
        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->face(face_no)->at_boundary())
            {
              if (std::fabs(cell->face(face_no)->center()[1] - top[1]) < 1e-12)
                cell->face(face_no)->set_boundary_id(2);
              if (std::fabs(cell->face(face_no)->center()[1] - bottom[1]) < 1e-12)
                cell->face(face_no)->set_boundary_id(1);
            }
    }

    int initial_refinement_level = 4;
    triangulation.refine_global(initial_refinement_level);
  }

  template <int dim>
  void PDSSolid<dim>::impose_displacement_on_solution(double time)
  {
    int n_displacement_conditions = data.displacement_boundary_labels.size();
    std::vector<double> displacement_values(n_displacement_conditions);
    for (int i=0; i<n_displacement_conditions; ++i)
      displacement_values[i] = data.displacement_boundary_velocities[i]*time;
    phase_field_solver.impose_displacement(displacement_values);
  }  // eom


  template <int dim>
  void PDSSolid<dim>::run()
  {
    create_mesh();
    // read_mesh();
    phase_field_solver.setup_dofs();

    // set initial phase-field to 1
    phase_field_solver.solution.block(1) = 1;
    phase_field_solver.old_solution.block(1) = 1;

    int time_step_number = 0;
    double time = 1, t_max = 3, time_step = 1;
    while(time < t_max)
      {
        pcout << "Time: " << time << std::endl;
        phase_field_solver.old_old_solution = phase_field_solver.old_solution;
        phase_field_solver.old_solution = phase_field_solver.solution;

        IndexSet old_active_set(phase_field_solver.active_set);
        impose_displacement_on_solution(time);

        int n_iter = 1;
        int max_newton_iter = 50;
        double newton_tolerance = 1e-6;
        while (n_iter < max_newton_iter)
          {
            pcout << "Newton iteration: " << n_iter << "\t";
            if (n_iter > 1)
              {
                phase_field_solver.compute_residual();
                phase_field_solver.compute_active_set();
                pcout << "Size of active set: "
                      << phase_field_solver.active_set.n_elements()
                      << "\t";
                double error = phase_field_solver.residual_norm();
                pcout << "error = " << error << "\t";
                if ((phase_field_solver.active_set == old_active_set) &&
                    (error < newton_tolerance))
                  {
                    pcout << "Cool" << std::endl;
                    break;
                  }
                old_active_set = phase_field_solver.active_set;
              }

            phase_field_solver.assemble_system();
            phase_field_solver.solve();
            phase_field_solver.solution_update *= 0.6;
            phase_field_solver.solution += phase_field_solver.solution_update;

            n_iter++;

          }  // End Newton iter

        output_results(time_step_number);

        time += time_step;
        time_step_number++;
      }  // end time loop
  }  // EOM


  template <int dim>
  void PDSSolid<dim>::output_results(int time_step_number) const
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
    data_out.attach_dof_handler(phase_field_solver.dof_handler);
    data_out.add_data_vector(phase_field_solver.solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    // add active set
    data_out.add_data_vector(phase_field_solver.active_set, "active_set");
    // Add domain ids
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    int n_time_step_digits = 3,
        n_processor_digits = 3;

    // Write output from local processors
    const std::string filename = ("solution/solution-" +
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
        std::ofstream master_output(("solution/solution-" +
                                     Utilities::int_to_string(time_step_number,
                                                              n_time_step_digits) +
                                     ".pvtu").c_str());
        data_out.write_pvtu_record(master_output, filenames);
      }  // end master output
  } // EOM
}  // end of namespace



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    pds_solid::PDSSolid<2> problem;
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
