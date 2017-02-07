#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/generic_linear_algebra.h>


#include <deal.II/lac/vector.h>
// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/constraint_matrix.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
// #include <deal.II/numerics/data_out.h>
// #include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/distributed/tria.h>

// Custom modules
#include <ConstitutiveModel.cc>

namespace phase_field
{
  using namespace dealii;


  const double mu = 1000, lambda = 1e6;

  template <int dim>
  class PhaseFieldSolver
  {
  public:
    // methods
    PhaseFieldSolver(MPI_Comm &mpi_communicator,
                     parallel::distributed::Triangulation<dim> &triangulation,
                     ConditionalOStream &pcout_,
                     TimerOutput &computing_timer_);
    ~PhaseFieldSolver();

    void setup_dofs();
    void compute_residual();
    void assemble_rhs_vector();
    void assemble_system_matrix();

    // variables
    TrilinosWrappers::MPI::Vector residual, rhs_vector, solution, solution_update,
      old_solution, old_old_phase_field;
    TrilinosWrappers::SparseMatrix system_matrix;
    ConstraintMatrix constraints;

  private:
    void get_stress_decomposition(const SymmetricTensor<2,dim> &strain_tensor,
                                  SymmetricTensor<2,dim>       &stress_tensor_plus,
                                  SymmetricTensor<2,dim>       &stress_tensor_minus);
    // variables
    MPI_Comm &mpi_communicator;
    DoFHandler<dim> dof_handler;
    // parallel::distributed::Triangulation<dim> &triangulation;
    ConditionalOStream &pcout;
    TimerOutput &computing_timer;
    FESystem<dim> fe;
    IndexSet locally_owned_dofs, locally_relevant_dofs;
    IndexSet active_set;
    TrilinosWrappers::SparseMatrix reduced_system_matrix;

  public:
    double time_step;
  };



  template <int dim>
  PhaseFieldSolver<dim>::PhaseFieldSolver
  (MPI_Comm &mpi_communicator_,
   parallel::distributed::Triangulation<dim> &triangulation,
   ConditionalOStream &pcout_,
   TimerOutput &computing_timer_)
    :
    mpi_communicator(mpi_communicator_),
    dof_handler(triangulation),
    pcout(pcout_),
    computing_timer(computing_timer_),
    // displacement components + phase_field var
    fe(FE_Q<dim>(1), dim+1)
  {
    pcout << "Solver class initialization successful" << std::endl;
  }  // EOM


  template <int dim>
  PhaseFieldSolver<dim>::~PhaseFieldSolver()
  {
    dof_handler.clear();
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::setup_dofs()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs(fe);
    active_set.set_size(dof_handler.n_dofs());
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    // TODO: constraints only on the displacement part
    // VectorTools::interpolate_boundary_values(dof_handler,
    //                                          0,
    //                                          ZeroFunction<dim>(),
    //                                          constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    false);
    system_matrix.reinit(dsp);
    reduced_system_matrix.reinit(dsp);
    IndexSet solution_index_set = dof_handler.locally_owned_dofs();
    solution.reinit(solution_index_set, mpi_communicator);
    rhs_vector.reinit(solution_index_set, mpi_communicator);

    // TODO:
    // Add some stuff to make mass matrix B and reinitialize reduced system
    // matrix, residual, and rhs
    TrilinosWrappers::SparseMatrix mass_matrix;
    // mass_matrix.reinit(dsp);
    // assemble_mass_matrix_diagonal(mass_matrix);
    // diagonal_of_mass_matrix.reinit (solution_index_set);
    // for (unsigned int j=0; j<solution.size (); j++)
    //   diagonal_of_mass_matrix (j) = mass_matrix.diag_element (j);
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::assemble_rhs_vector()
  {
    TimerOutput::Scope t(computing_timer, "assemble rhs vector");

    const QGauss<dim> quadrature_formula(3);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_JxW_values);
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();
    FullMatrix<double>   local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector displacement(0);
    const FEValuesExtractors::Scalar phase_field(dim);

    SymmetricTensor<2,dim>	eps_i;

    // store solution displacement gradients
    std::vector< SymmetricTensor<2,dim> >
      strain_tensor(quadrature_formula.size());
    SymmetricTensor<2, dim> stress_tensor_plus, stress_tensor_minus;
    Tensor<1, dim> grad_xi_phi_i;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          local_matrix = 0;
          local_rhs = 0;
          // right_hand_side.value_list(fe_values.get_quadrature_points(),
          //                            rhs_values);

          fe_values[displacement].get_function_symmetric_gradients
            (solution, strain_tensor);

          for (unsigned int q=0; q<n_q_points; ++q) {
            get_stress_decomposition(strain_tensor[q],
                                     stress_tensor_plus,
                                     stress_tensor_minus);

            // for (unsigned int k=0; k<dofs_per_cell; ++k) {
            //   strain_tensor =
            // }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                eps_i = fe_values[displacement].symmetric_gradient(i, q);

                // that's for the pressure term, don't remove it
                // double div_phi_i = fe_values[displacement].divergence(i, q);

                double xi_phi_i = fe_values[phase_field].value(i ,q);
                grad_xi_phi_i = fe_values[phase_field].gradient(i, q);


              }  // end i loop

          }  // end q loop
          break;
        }  // end of cell loop

  } // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::get_stress_decomposition
  (const  SymmetricTensor<2,dim> &strain_tensor,
   SymmetricTensor<2,dim>        &stress_tensor_plus,
   SymmetricTensor<2,dim>        &stress_tensor_minus)
  {
    SymmetricTensor<2,dim> strain_tensor_plus;
    constitutive_model::get_strain_tensor_plus(strain_tensor,
                                               strain_tensor_plus);
    double trace_eps = trace(strain_tensor);
    double trace_eps_pos = std::max(trace_eps, 0.0);
    stress_tensor_plus = 2*mu*strain_tensor_plus;
    stress_tensor_plus[0][0] += lambda*trace_eps_pos;
    stress_tensor_plus[1][1] += lambda*trace_eps_pos;

    stress_tensor_minus = 2*mu*(strain_tensor - strain_tensor_plus);
    stress_tensor_minus[0][0] += lambda*(trace_eps - trace_eps_pos);
    stress_tensor_minus[1][1] += lambda*(trace_eps - trace_eps_pos);
  }


}  // end of namespace
