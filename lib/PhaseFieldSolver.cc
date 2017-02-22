#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>
// #include <deal.II/lac/trilinos_precondition.h>

// #include <deal.II/grid/tria_accessor.h>
// #include <deal.II/grid/tria_iterator.h>

// DOF stuff
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/sparsity_tools.h>

#include <cmath>        // std:: math functions

// Custom modules
#include <ConstitutiveModel.cc>
#include <InputData.cc>
#include <LinearSolver.cc>


namespace phase_field
{
  using namespace dealii;


  template <int dim>
  class PhaseFieldSolver
  {
  public:
    // methods
    PhaseFieldSolver(MPI_Comm &mpi_communicator,
                     parallel::distributed::Triangulation<dim> &triangulation_,
                     input_data::PhaseFieldData &data_,
                     ConditionalOStream &pcout_,
                     TimerOutput &computing_timer_);
    ~PhaseFieldSolver();

    void setup_dofs();
    double compute_residual();
    double residual_norm() const;
    void assemble_system();
    void compute_active_set();
    void impose_displacement(const std::vector<double> &displacement_values);
    void solve();

  private:
    // this function also computes finest mesh size
    void assemble_mass_matrix_diagonal(TrilinosWrappers::
                                       BlockSparseMatrix &mass_matrix);

  public:
    // variables
    MPI_Comm &mpi_communicator;
    parallel::distributed::Triangulation<dim> &triangulation;
    input_data::PhaseFieldData &data;
    DoFHandler<dim> dof_handler;

  private:
    ConditionalOStream &pcout;
    TimerOutput &computing_timer;

    FESystem<dim> fe;

    IndexSet locally_owned_dofs;

    TrilinosWrappers::MPI::BlockVector rhs_vector, reduced_rhs_vector;
    TrilinosWrappers::MPI::BlockVector mass_matrix_diagonal;

    TrilinosWrappers::BlockSparseMatrix system_matrix, reduced_system_matrix;
    TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

    ConstraintMatrix physical_constraints, all_constraints;
    double min_cell_size;

  public:
    double time_step;
    IndexSet active_set;
    TrilinosWrappers::MPI::BlockVector solution, solution_update, residual;
    TrilinosWrappers::MPI::BlockVector old_solution, old_old_solution;
  };


  template <int dim>
  PhaseFieldSolver<dim>::PhaseFieldSolver
  (MPI_Comm &mpi_communicator_,
   parallel::distributed::Triangulation<dim> &triangulation_,
   input_data::PhaseFieldData &data_,
   ConditionalOStream &pcout_,
   TimerOutput &computing_timer_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    data(data_),
    dof_handler(triangulation_),
    pcout(pcout_),
    computing_timer(computing_timer_),
    fe(FE_Q<dim>(1), dim,  // displacement components
       FE_Q<dim>(1), 1)    // phase-field
  {}  // EOM


  template <int dim>
  PhaseFieldSolver<dim>::~PhaseFieldSolver()
  {
    dof_handler.clear();
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::setup_dofs()
  {
    computing_timer.enter_section("Setup system");

    // Displacements in block 0
    // phase-field in block 1
    std::vector<unsigned int> blocks(dim+1, 0);
    blocks[dim] = 1;

    // distribute and renumber dofs
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler, blocks);

    // Count dofs per block
    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(dof_handler, dofs_per_block, blocks);
    const unsigned int n_u = dofs_per_block[0],
                       n_phi = dofs_per_block[1];

    pcout << "Number of active cells: "
          << triangulation.n_global_active_cells()
          << " (on "
          << triangulation.n_levels()
          << " levels)"
          << std::endl
          << "   Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << " (" << n_u << '+' << n_phi << ')'
          << std::endl;

    // Partitioning
    std::vector<IndexSet> partitioning, relevant_partitioning;
    IndexSet locally_relevant_dofs;
    { // compute owned dofs, partitioning, and relevant partitioning
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      active_set.set_size(dof_handler.n_locally_owned_dofs());

      partitioning.push_back(locally_owned_dofs.get_view(0, n_u));
      partitioning.push_back(locally_owned_dofs.get_view(n_u, n_u+n_phi));

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      relevant_partitioning.push_back(locally_relevant_dofs.get_view(0, n_u));
      relevant_partitioning.push_back(locally_relevant_dofs.get_view(n_u, n_u+n_phi));
    }

    { // constraints
      physical_constraints.clear();
      physical_constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              physical_constraints);

      /*
        Impose dirichlet conditions:
        we set the components of those displacements, that are imposed on the
        solution, to zero
      */
      // Extract displacement components
      std::vector<FEValuesExtractors::Scalar> displacement_masks(dim);
      for (int i=0; i<dim; ++i)
        {
          const FEValuesExtractors::Scalar comp(i);
          displacement_masks[i] = comp;
        }

      int n_dirichlet_conditions = data.displacement_boundary_labels.size();

      // Insert values into the constraints matrix
      for (int cond=0; cond<n_dirichlet_conditions; ++cond)
        {
          int component = data.displacement_boundary_components[cond];
          VectorTools::interpolate_boundary_values
            (dof_handler,
             data.displacement_boundary_labels[cond],
             ZeroFunction<dim>(dim+1),
             physical_constraints,
             fe.component_mask(displacement_masks[component]));
        } // end loop over dirichlet conditions

      physical_constraints.close();

      all_constraints.clear();
      all_constraints.reinit(locally_relevant_dofs);
      all_constraints.merge(physical_constraints);
      all_constraints.close();
    }

    { // Setup system matrices and diagonal mass matrix
      system_matrix.clear();
      reduced_system_matrix.clear();

      /*
      Displacements are coupled with one another (upper-left block = true),
      displacements are coupled with phase-field (lower-left block = true),
      phase-field is not coupled with displacements (upper-right block = false),
      phase-field is coupled with itself (lower-right block = true)
      */
      Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);
      for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
          if ( (c<=dim && d<dim) || (c==dim && d==dim) )
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      // TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
      //                                           relevant_partitioning,
      //                                           mpi_communicator);
      // DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
      //                                 physical_constraints,
      //                                 /* 	keep_constrained_dofs = */ false,
      //                                 Utilities::MPI::
      //                                 this_mpi_process(mpi_communicator));
      // sp.compress();

      // system_matrix.reinit(sp);
      // reduced_system_matrix.reinit(sp);

      BlockDynamicSparsityPattern sp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
                                      physical_constraints,
                                      /* 	keep_constrained_dofs = */ false);
      SparsityTools::
        distribute_sparsity_pattern(sp,
                                    dof_handler.locally_owned_dofs_per_processor(),
                                    mpi_communicator,
                                    locally_relevant_dofs);
      system_matrix.reinit(partitioning, sp, mpi_communicator);
      reduced_system_matrix.reinit(partitioning, sp, mpi_communicator);

      IndexSet::ElementIterator
        index = locally_owned_dofs.begin(),
        end_index = locally_owned_dofs.end();

      // Finally assemble the diagonal of the mass matrix
      TrilinosWrappers::BlockSparseMatrix mass_matrix;
      mass_matrix.reinit(sp);
      assemble_mass_matrix_diagonal(mass_matrix);
      mass_matrix_diagonal.reinit(partitioning, relevant_partitioning,
                                  mpi_communicator, /* omit-zeros=*/ true);

      for (; index!=end_index; ++index)
        mass_matrix_diagonal(*index) = mass_matrix.diag_element(*index);

      mass_matrix_diagonal.compress(VectorOperation::insert);
    }

    { // Preconditioner matrix
      // preconditioner_matrix.clear();
      //
      // TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
      //                                           relevant_partitioning,
      //                                           mpi_communicator);
      //
      // // only phi-phi entries
      // Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);
      // for (unsigned int c=0; c<dim+1; ++c)
      //   for (unsigned int d=0; d<dim+1; ++d)
      //     if (c==dim && d==dim)
      //       coupling[c][d] = DoFTools::always;
      //     else
      //       coupling[c][d] = DoFTools::none;
      //
      // DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
      //                                 physical_constraints,
      //                                 /* 	keep_constrained_dofs = */ false,
      //                                 Utilities::MPI::
      //                                 this_mpi_process(mpi_communicator));
      //
      // sp.compress();
      // preconditioner_matrix.reinit(sp);
    }

    { // Setup vectors
      solution.reinit(partitioning, mpi_communicator);
      old_solution.reinit(solution);
      old_old_solution.reinit(solution);
      solution_update.reinit(partitioning, relevant_partitioning, mpi_communicator);
      rhs_vector.reinit(partitioning, mpi_communicator, /* omit-zeros=*/ true);
      reduced_rhs_vector.reinit(partitioning, relevant_partitioning,
                                mpi_communicator, /* omit-zeros=*/ true);
      residual.reinit(rhs_vector);
    }

    computing_timer.exit_section();
  }  // EOM


  template <int dim>
  void convert_to_tensor(const SymmetricTensor<2, dim> &symm_tensor,
                         Tensor<2, dim>                &tensor)
  {
    for (int i=0; i<dim; ++i)
      for (int j=0; j<dim; ++j)
        tensor[i][j] = symm_tensor[i][j];
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::assemble_system()
  {
    computing_timer.enter_section("Assemble system");

    const QGauss<dim> quadrature_formula(3);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points |
                            update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix(dofs_per_cell, dofs_per_cell),
                         prec_local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector displacement(0);
    const FEValuesExtractors::Scalar phase_field(dim);

    // FeValues containers
    std::vector< Tensor<2,dim> > eps_u(dofs_per_cell);
    std::vector< Tensor<1,dim> > grad_xi_phi(dofs_per_cell);
    std::vector< Tensor<1,dim> > grad_phi_values(n_q_points);

    // Solution values containers
    std::vector< SymmetricTensor<2,dim> > strain_tensor_values(n_q_points);
    Tensor<2, dim> strain_tensor_value;
    std::vector<double> phi_values(n_q_points),
                        old_phi_values(n_q_points),
                        old_old_phi_values(n_q_points);
    std::vector<double> xi_phi(dofs_per_cell);

    // Stress decomposition containers
    constitutive_model::EnergySpectralDecomposition<dim> stress_decomposition;
    Tensor<2, dim> stress_tensor_plus, stress_tensor_minus;
    std::vector< Tensor<2, dim> >
      sigma_u_plus(dofs_per_cell),
      sigma_u_minus(dofs_per_cell);

    // Equation data
    double kappa = data.regularization_parameter_kappa;
    double e =
      data.penalty_parameter
      *std::pow(min_cell_size, 0.2);

    double gamma_c = data.energy_release_rate;

    Tensor<4,dim> gassman_tensor =
     constitutive_model::isotropic_gassman_tensor<dim>(data.lame_constant,
                                                       data.shear_modulus);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          local_rhs = 0;
          local_matrix = 0;
          prec_local_matrix = 0;

          fe_values.reinit(cell);

          fe_values[displacement].get_function_symmetric_gradients
            (solution, strain_tensor_values);

          fe_values[phase_field].get_function_values(solution,
                                                     phi_values);
          fe_values[phase_field].get_function_values(old_solution,
                                                     old_phi_values);
          fe_values[phase_field].get_function_values(old_old_solution,
                                                     old_old_phi_values);
          fe_values[phase_field].get_function_gradients(solution,
                                                        grad_phi_values);

          for (unsigned int q=0; q<n_q_points; ++q)
            {
              // convert from non-symmetric tensor
              convert_to_tensor(strain_tensor_values[q], strain_tensor_value);

              // stress_decomposition.get_stress_decomposition(strain_tensor_value,
              //                                               data.lame_constant,
              //                                               data.shear_modulus,
              //                                               stress_tensor_plus,
              //                                               stress_tensor_minus);
              stress_tensor_minus = 0;
              stress_tensor_plus =
                  double_contract<2, 0, 3, 1>(gassman_tensor, strain_tensor_value);

              // TODO: include time into hereouble_contract<2, 0, 3, 1>(gassman_tensor,
              double d_phi = old_phi_values[q] - old_old_phi_values[q];
              double phi_e = old_phi_values[q] + d_phi;  // extrapolated
              double phi_value = phi_values[q];
              double jxw = fe_values.JxW(q);

              for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                  eps_u[k] = fe_values[displacement].symmetric_gradient(k, q);
                  xi_phi[k] = fe_values[phase_field].value(k ,q);
                  grad_xi_phi[k] = fe_values[phase_field].gradient(k, q);

                  // stress_decomposition.get_stress_decomposition_derivatives
                  //   (strain_tensor_value,
                  //    eps_u[k],
                  //    data.lame_constant,
                  //    data.shear_modulus,
                  //    sigma_u_plus[k],
                  //    sigma_u_minus[k]);

                  sigma_u_minus[k] = 0;
                  sigma_u_plus[k] =
                    double_contract<2, 0, 3, 1>(gassman_tensor, eps_u[k]);

                }  // end k loop

              // Assemble local rhs +
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  double rhs_u =
                    ((1.-kappa)*phi_e*phi_e + kappa)
                      *scalar_product(stress_tensor_plus, eps_u[i])
                    +
                    scalar_product(stress_tensor_minus, eps_u[i]);

                  double rhs_phi =
                    (1.-kappa)*phi_value*xi_phi[i]
                      *scalar_product(stress_tensor_plus, strain_tensor_value)
                    -
                    gamma_c/e*(1-phi_value)*xi_phi[i]
                    +
                    gamma_c*e
                      *scalar_product(grad_phi_values[q], grad_xi_phi[i])
                    ;

                  local_rhs[i] -= (rhs_u + rhs_phi)*jxw;

                }  // end i loop

              // Assemble local matrix
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                    double m_u_u =
                      ((1.-kappa)*phi_e*phi_e + kappa)
                        *scalar_product(sigma_u_plus[j], eps_u[i])
                      +
                      scalar_product(sigma_u_minus[j], eps_u[i]);

                    double m_phi_u =
                      2.*(1.-kappa)*phi_value
                        *scalar_product(sigma_u_plus[j], strain_tensor_value)
                        *xi_phi[i];

                    // double m_u_phi = 0;

                    double m_phi_phi =
                      (1.-kappa)*xi_phi[j]*xi_phi[i]
                        *scalar_product(stress_tensor_plus, strain_tensor_value)
                      +
                      gamma_c/e*(xi_phi[j]*xi_phi[i])
                      +
                      gamma_c*e*scalar_product(grad_xi_phi[j], grad_xi_phi[i])
                      ;

                    local_matrix(i, j) += (m_u_u + m_phi_u + m_phi_phi) * jxw;

                    // prec_local_matrix(i, j) +=
                    //   // 1./e/gamma_c*
                    //   (xi_phi[j]*xi_phi[i])*jxw;
                      // scalar_product(grad_xi_phi[j], grad_xi_phi[i])*jxw;

                  }  // end j loop
            }  // end q loop

          // pcout << "\n cell matrix = " << std::endl;
          // for (int i=0; i<dofs_per_cell; ++i)
          //   {
          //     for (int j=0; j<dofs_per_cell; ++j)
          //       pcout << local_matrix(i, j) << "\t";
          //     pcout << std::endl;
          //   }
          // pcout << "\n cell rhs = " << std::endl;
          // for (int i=0; i<dofs_per_cell; ++i)
          //   pcout << local_rhs(i) << "\t";
          // pcout << std::endl;


          cell->get_dof_indices(local_dof_indices);

          physical_constraints.distribute_local_to_global(local_matrix,
                                                          local_rhs,
                                                          local_dof_indices,
                                                          system_matrix,
                                                          rhs_vector);

          all_constraints.distribute_local_to_global(local_matrix,
                                                     local_rhs,
                                                     local_dof_indices,
                                                     reduced_system_matrix,
                                                     reduced_rhs_vector);

          // all_constraints.distribute_local_to_global(prec_local_matrix,
          //                                            local_dof_indices,
          //                                            preconditioner_matrix);
        }  // end of cell loop

    system_matrix.compress(VectorOperation::add);
    reduced_system_matrix.compress(VectorOperation::add);
    preconditioner_matrix.compress(VectorOperation::add);
    rhs_vector.compress(VectorOperation::add);
    reduced_rhs_vector.compress(VectorOperation::add);

    computing_timer.exit_section();

  } // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::
  assemble_mass_matrix_diagonal(TrilinosWrappers::
                                BlockSparseMatrix &mass_matrix)
  {
    Assert (fe.degree == 1, ExcNotImplemented());
    computing_timer.enter_section("Assemble mass matrix diagonal");
    const QTrapez<dim> quadrature_formula;
    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_quadrature_points |
                             update_values |
                             update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // additionally, compute finest cell size
    double min_local_cell_size =
      triangulation.last()->diameter()/std::sqrt(dim);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    mass_matrix = 0;

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell_matrix = 0;
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              cell_matrix(i,i) += (fe_values.shape_value(i, q_point) *
                                   fe_values.shape_value(i, q_point) *
                                   fe_values.JxW(q_point));
          cell->get_dof_indices(local_dof_indices);
          physical_constraints.distribute_local_to_global(cell_matrix,
                                                          local_dof_indices,
                                                          mass_matrix);

          double cell_size = cell->diameter()/std::sqrt(dim);
          if (cell_size < min_local_cell_size)
            min_local_cell_size = cell_size;
          // break;
        }  // end cell loop

    mass_matrix.compress(VectorOperation::add);

    min_cell_size =
      -Utilities::MPI::max(-min_local_cell_size, MPI_COMM_WORLD);
    pcout << "Minimum cell size: " << min_cell_size << std::endl;

    computing_timer.exit_section();

  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::
  compute_active_set()
  {
    computing_timer.enter_section("Computing active set");
    active_set.clear();
    all_constraints.clear();

    // Iterator
    IndexSet::ElementIterator
      index = locally_owned_dofs.begin(),
      end_index = locally_owned_dofs.end();

    for (; index!=end_index; ++index)
      {
        const unsigned int i = *index;
        if (residual[i]/mass_matrix_diagonal[i] +
            data.penalty_parameter*solution_update[i] > 0)
          {
            active_set.add_index(i);
            all_constraints.add_line(i);
            all_constraints.set_inhomogeneity(i, 0);
            solution_update[i] = 0;
            residual[i] = 0;
          }
      }  // end of dof loop

    all_constraints.merge(physical_constraints);
    all_constraints.close();
    computing_timer.exit_section();
  }  // EOM


  template <int dim>
  double PhaseFieldSolver<dim>::compute_residual()
  {
    return system_matrix.residual(residual, solution_update, rhs_vector);
  }  // EOM


  template <int dim>
  double PhaseFieldSolver<dim>::residual_norm() const
  {
    return residual.l2_norm();
  }


  template <int dim>
  void PhaseFieldSolver<dim>::
  impose_displacement(const std::vector<double> &displacement_values)
  {
    computing_timer.enter_section("Imposing displacement values");

    // Extract displacement components
    std::vector<FEValuesExtractors::Scalar> displacement_masks(dim);
    for (int i=0; i<dim; ++i)
      {
        const FEValuesExtractors::Scalar comp(i);
        displacement_masks[i] = comp;
      }

    // container for displacement boundary values
    std::map<types::global_dof_index, double> boundary_values;
    int n_dirichlet_conditions = data.displacement_boundary_labels.size();

    // Store BC's in the container "boundary_values"
    for (int cond=0; cond<n_dirichlet_conditions; ++cond)
      {
        int component = data.displacement_boundary_components[cond];
        double dirichlet_value = displacement_values[cond];
        VectorTools::interpolate_boundary_values
          (dof_handler,
           data.displacement_boundary_labels[cond],
           ConstantFunction<dim>(dirichlet_value, dim+1),
           boundary_values,
           fe.component_mask(displacement_masks[component]));
      }

    // Apply BC values to the solution vector
    for (std::map<types::global_dof_index,double>::const_iterator
           p = boundary_values.begin();
           p != boundary_values.end(); ++p)
      solution(p->first) = p->second;

    solution.compress(VectorOperation::insert);
    computing_timer.exit_section();
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::
  solve()
  {
    /*
      In this method we essentially use 2 block diagonal preconditioners
      for the block (0,0) and the block (1, 1)
    */
    computing_timer.enter_section("Solve phase-field system");

    // Preconditioner for the displacement (0, 0) block
    TrilinosWrappers::PreconditionAMG prec_A;
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      prec_A.initialize(system_matrix.block(0, 0), data);
    }

    // Preconditioner for the phase-field (1, 1) block
    TrilinosWrappers::PreconditionAMG prec_S;
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      prec_S.initialize(system_matrix.block(1, 1), data);
    }

    // Construct block preconditioner (for the whole matrix)
    const LinearSolvers::
      BlockDiagonalPreconditioner<TrilinosWrappers::PreconditionAMG,
                                  TrilinosWrappers::PreconditionAMG>
      preconditioner(prec_A, prec_S);

    // set up the linear solver and solve the system
    unsigned int max_iter = 5*system_matrix.m();
    SolverControl solver_control(max_iter, 1e-10);

    // SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
    SolverGMRES<TrilinosWrappers::MPI::BlockVector>
      solver(solver_control);

    // all_constraints.set_zero(solution_update);
    solver.solve(system_matrix, solution_update, rhs_vector, preconditioner);

    pcout << "Solved in " << solver_control.last_step()
          << " iterations." << std::endl;

    all_constraints.distribute(solution_update);

    computing_timer.exit_section();

  }  // EOM

}  // end of namespace
