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
    void refine_mesh();

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
    double length = data.domain_size;
    GridGenerator::hyper_cube_slit(triangulation,
                                   0, length,
                                   /*colorize = */ false);
    {// Assign boundary ids
      // outer domain middle-edge points
      const Point<dim> top(0, length);
      const Point<dim> bottom(0, 0);
      const Point<dim> right(length, 0);
      const Point<dim> left(0, 0);

      // outer domain boundary id's
      const int left_boundary_id = 0;
      const int right_boundary_id = 1;
      const int bottom_boundary_id = 2;
      const int top_boundary_id = 3;

      // slit edge id's
      const int slit_edge1_id = 4;
      const int slit_edge2_id = 5;

      { // Colorize slit boundaries
        typename Triangulation<2>::active_cell_iterator
          cell = triangulation.begin_active();

        cell->face(1)->set_boundary_id(slit_edge1_id);  // left slit edge
        ++cell;
        cell->face(0)->set_boundary_id(slit_edge2_id);  // right slit edge
      }

      typename Triangulation<2>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

      for (; cell != endc; ++cell)
        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell;
             ++face_no)
          if (cell->face(face_no)->at_boundary())
            {
              if (
                (cell->face(face_no)->boundary_id() != slit_edge1_id)
                &&
                (cell->face(face_no)->boundary_id() != slit_edge2_id)
              )
              {
                if (std::fabs(cell->face(face_no)->center()[1] - top[1]) < 1e-12)
                  cell->face(face_no)->set_boundary_id(top_boundary_id);
                if (std::fabs(cell->face(face_no)->center()[1] - bottom[1]) < 1e-12)
                  cell->face(face_no)->set_boundary_id(bottom_boundary_id);
                if (std::fabs(cell->face(face_no)->center()[0] - left[0]) < 1e-12)
                  cell->face(face_no)->set_boundary_id(left_boundary_id);
                if (std::fabs(cell->face(face_no)->center()[0] - right[0]) < 1e-12)
                  cell->face(face_no)->set_boundary_id(right_boundary_id);
              }
            }  // end cell loop
    } // end assigning boundary id's

  }  // eom

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
  void PDSSolid<dim>::refine_mesh()
  {
    typename Triangulation<2>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();

    double refined_portion = 0.2;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        if (
          std::fabs(cell->face(2)->center()[0]) < data.domain_size*(0.5+refined_portion/2)
          &&
          std::fabs(cell->face(2)->center()[0]) > data.domain_size*(0.5-refined_portion/2)
          &&
          cell->face(0)->center()[1] > data.domain_size/3
          )
          cell->set_refine_flag();

    // triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }  // eom


  template <int dim>
  void PDSSolid<dim>::run()
  {
    create_mesh();
    triangulation.refine_global(data.initial_refinement_level);
    for (int refinement_cycle = 0;
         refinement_cycle < data.max_refinement_level - data.initial_refinement_level;
         refinement_cycle++)
      refine_mesh();

    // read_mesh();
    phase_field_solver.setup_dofs();

    // Compute regularization_parameter_epsilon
    double minimum_mesh_size = (data.domain_size/2)/std::pow(2, data.max_refinement_level);
    data.regularization_parameter_epsilon = 2*minimum_mesh_size;

    // set initial phase-field to 1
    phase_field_solver.solution.block(1) = 1;
    phase_field_solver.old_solution.block(1) = 1;

    IndexSet old_active_set(phase_field_solver.active_set);

    int time_step_number = 0;
    double time_step = 1e-4;
    double old_time_step = time_step;
    double time = 0;
    double t_max = 1e-2;
    while(time < t_max)
    {
      time += time_step;
      time_step_number++;

      pcout << std::endl
            << "Time: "
            << std::fixed << time
            << std::endl;

      phase_field_solver.truncate_phase_field();
      phase_field_solver.update_old_solution();

      double tmp_time_step = time_step;

    redo_time_step:
      impose_displacement_on_solution(time);
      std::vector<double> time_steps = {time_step, old_time_step};

      int newton_step = 0;
      const int max_newton_iter = 20;
      const double newton_tolerance = 1e-6;
      while (newton_step < max_newton_iter)
      {
        pcout << "Newton iteration: " << newton_step << "\t";

        double error;
        if (newton_step > 0)
        {
          phase_field_solver.
            compute_nonlinear_residual(phase_field_solver.solution,
                                       time_steps);

          phase_field_solver.compute_active_set(phase_field_solver.solution);
          phase_field_solver.all_constraints.set_zero(phase_field_solver.residual);

          pcout << "Active set: "
                << phase_field_solver.active_set.n_elements()
                << "\t";
          error = phase_field_solver.residual_norm();
          pcout << std::scientific << "error = " << error << "\t";

          // break condition
          if (phase_field_solver.active_set_changed(old_active_set) &&
              error < newton_tolerance)
            {
              pcout << "Converged!" << std::endl;
              break;
            }
          old_active_set = phase_field_solver.active_set;
        }

        phase_field_solver.solve_newton_step(time_steps);

        // output_results(newton_step);
        newton_step++;

        pcout << std::endl;
      }  // End Newton iter

      // cut the time step if no convergence
      if (newton_step == max_newton_iter)
      {
        pcout << "Time step didn't converge: reducing to dt = "
              << time_step/10 << std::endl;
        time -= time_step;
        time_step /= 10.0;
        time += time_step;
        phase_field_solver.solution = phase_field_solver.old_solution;
        phase_field_solver.use_old_time_step_phi = true;
        goto redo_time_step;
      }

      output_results(time_step_number);

      phase_field_solver.use_old_time_step_phi = false;

      old_time_step = time_step;
      time_step = tmp_time_step;
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
