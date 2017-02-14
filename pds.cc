#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>

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
    void setup_dofs();
    void impose_displacement_on_solution(double time);

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
  void PDSSolid<dim>::create_mesh()
  {
    Tensor<1, dim> point_1, point_2;

    double domain_length = 10, domain_width=10;

    for(int i=0; i<dim; ++i){
      point_1[i] += domain_length/2;
      point_2[i] -= domain_width/2;
    }

    Point<dim> p1(point_1), p2(point_2);

    GridGenerator::hyper_rectangle
      (triangulation, p1, p2,
       /*colorize = */ true);
    // GridGenerator::hyper_cube(triangulation);
    // triangulation.set_all_manifold_ids(0);

    int initial_refinement_level = 2;
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
  }

  template <int dim>
  void PDSSolid<dim>::run()
  {
    create_mesh();
    phase_field_solver.setup_dofs();
    double time = 0, t_max = 10, time_step = 1;
    while(time < t_max)
      {
        phase_field_solver.old_old_solution = phase_field_solver.old_solution;
        phase_field_solver.old_solution = phase_field_solver.solution;

        IndexSet old_active_set(phase_field_solver.active_set);
        impose_displacement_on_solution(time);

        int n_iter = 1;
        int max_newton_iter = 10;
        double newton_tolerance = 1e-6;
        while (n_iter < max_newton_iter)
          {
            if (n_iter > 1)
              {
                phase_field_solver.compute_active_set();
                double error = phase_field_solver.compute_residual();
                pcout << "error = " << error << std::endl;
                if ((phase_field_solver.active_set == old_active_set) &&
                    (error < newton_tolerance)
                    // 1
                    )
                  {
                    pcout << "Cool" << std::endl;
                    break;
                  }
                old_active_set = phase_field_solver.active_set;
              }

            phase_field_solver.assemble_system();
            phase_field_solver.solve();
            phase_field_solver.solution += phase_field_solver.solution_update;

            n_iter++;

          }  // End Newton iter

        time += time_step;

      }

  }


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
