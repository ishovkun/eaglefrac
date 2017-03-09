#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_in.h>

#include <limits>       // std::numeric_limits

// Custom modules
#include <PhaseFieldSolver.hpp>
#include <Postprocessing.hpp>
#include <InputData.hpp>
#include <Mesher.hpp>


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
    void output_results(int time_step_number); //const;
    void refine_mesh();
    void execute_postprocessing(const double time);
    void exectute_adaptive_refinement();

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    ConditionalOStream pcout;
    TimerOutput computing_timer;

    input_data::NotchedTestData data;
    PhaseField::PhaseFieldSolver<dim> phase_field_solver;
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
	  std::ifstream f(data.mesh_file_name);
    typename GridIn<dim>::Format format = GridIn<dim>::ucd;
    gridin.read(f, format);
	  // gridin.read_msh(f);
  }

  template <int dim>
  void PDSSolid<dim>::create_mesh()
  {
    double domain_size = 1;
    double length = domain_size;
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

    data.displacement_boundary_labels =     {0, 1, 1};
    data.displacement_boundary_components = {0 ,0, 1};
    data.displacement_boundary_velocities = {0, 1, 0};

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

    double domain_size = 1;
    double refined_portion = 0.2;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        if (
          std::fabs(cell->face(2)->center()[0]) < domain_size*(0.5+refined_portion/2)
          &&
          std::fabs(cell->face(2)->center()[0]) > domain_size*(0.5-refined_portion/2)
          &&
          cell->face(0)->center()[1] > domain_size/3
          )
          cell->set_refine_flag();

    // triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
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
      tmp_owned1(phase_field_solver.owned_partitioning),
      tmp_owned2(phase_field_solver.owned_partitioning);

    std::vector<TrilinosWrappers::MPI::BlockVector *> tmp1(3);
    tmp1[0] = &phase_field_solver.solution;
    tmp1[1] = &tmp_owned1;
    tmp1[2] = &tmp_owned2;

    solution_transfer.interpolate(tmp1);
    phase_field_solver.old_solution = tmp_owned1;
    phase_field_solver.old_old_solution = tmp_owned2;

  }  // eom


  template <int dim>
  void PDSSolid<dim>::run()
  {
    // create_mesh();
    read_mesh();

    // compute_runtime_parameters
    double minimum_mesh_size = Mesher::compute_minimum_mesh_size(triangulation,
                                                                 mpi_communicator);
    pcout << "max mesh size " << minimum_mesh_size << std::endl;
    const int max_refinement_level =
        data.n_prerefinement_steps
      + data.initial_refinement_level
      + data.n_adaptive_steps;
    minimum_mesh_size /= std::pow(2, max_refinement_level);
    data.regularization_parameter_epsilon = 2*minimum_mesh_size;

    // local prerefinement
    triangulation.refine_global(data.initial_refinement_level);
    for (int refinement_cycle = 0;
         refinement_cycle < data.n_prerefinement_steps;
         refinement_cycle++)
      refine_mesh();

    // read_mesh();
    phase_field_solver.setup_dofs();

    // set initial phase-field to 1
    phase_field_solver.solution.block(1) = 1;
    phase_field_solver.old_solution.block(1) = 1;

    double time_step = data.initial_time_step;
    double old_time_step = time_step;

    double time = 0;
    int time_step_number = 0;
    while(time < data.t_max)
    {
      time += time_step;
      time_step_number++;

      phase_field_solver.truncate_phase_field();
      phase_field_solver.update_old_solution();

      double tmp_time_step = time_step;

    redo_time_step:
      pcout << std::endl
            << "Time: "
            << std::fixed << time
            << std::endl;

      impose_displacement_on_solution(time);
      std::pair<double,double> time_steps = std::make_pair(time_step, old_time_step);

      IndexSet old_active_set(phase_field_solver.active_set);

      int newton_step = 0;
      const double newton_tolerance = data.newton_tolerance;
      while (newton_step < data.max_newton_iter)
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
                << phase_field_solver.active_set_size()
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
        }  // end first newton step condition

        phase_field_solver.solve_newton_step(time_steps);

        // output_results(newton_step);
        newton_step++;

        pcout << std::endl;
      }  // End Newton iter

      // cut the time step if no convergence
      if (newton_step == data.max_newton_iter)
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

      // do adaptive refinement if needed
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

      output_results(time_step_number);
      execute_postprocessing(time);

      phase_field_solver.use_old_time_step_phi = false;

      old_time_step = time_step;
      time_step = tmp_time_step;

      if (time >= data.t_max) break;
    }  // end time loop

    pcout << std::fixed;
  }  // EOM


  // template <int dim>
  // void PDSSolid<dim>::prepare_postprocessing(double time)
  // {
  //
  // }  // eom


  template <int dim>
  void PDSSolid<dim>::execute_postprocessing(const double time)
  {
    int boundary_id = data.displacement_boundary_labels[1];
    Tensor<1,dim> load = Postprocessing::compute_boundary_load(phase_field_solver,
                                                               data,
                                                               boundary_id);
    double d = data.displacement_boundary_velocities[1]*time;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::ofstream ff;
      ff.open("solution/post.txt", std::ios_base::app);
      ff << time << "\t"
         << d << "\t"
         << load[0] << "\t"
         << load[1] << "\t"
         << std::endl;
    }
  }  // eomj


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
