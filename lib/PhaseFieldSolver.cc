#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_gmres.h>

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

// Custom modules
#include <ConstitutiveModel.cc>
#include <InputData.cc>
#include <LinearSolver.cc>


namespace phase_field
{
  using namespace dealii;


  const double mu = 1000, lambda = 1e6;
  const double kappa = 1e-12, gamma_c = 1, e = 1e-6;
  const double penalty_parameter = 1;

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
    void compute_residual();
    void assemble_system();
    void compute_active_set();
    void impose_displacement(const std::vector<double> &displacement_values);
    void solve();

  private:
    void assemble_mass_matrix_diagonal(TrilinosWrappers::
                                       BlockSparseMatrix &mass_matrix);

    // variables
    MPI_Comm &mpi_communicator;
    parallel::distributed::Triangulation<dim> &triangulation;
    input_data::PhaseFieldData &data;
    DoFHandler<dim> dof_handler;
    ConditionalOStream &pcout;
    TimerOutput &computing_timer;

    FESystem<dim> fe;

    IndexSet locally_owned_dofs;
    IndexSet active_set;

    TrilinosWrappers::MPI::BlockVector solution, solution_update,
                                       old_solution, old_old_solution;
    TrilinosWrappers::MPI::BlockVector residual, rhs_vector, reduced_rhs_vector;
    TrilinosWrappers::MPI::BlockVector mass_matrix_diagonal;

    TrilinosWrappers::BlockSparseMatrix system_matrix, reduced_system_matrix;
    TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

    ConstraintMatrix physical_constraints, all_constraints;

  public:
    double time_step;
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

      // impose dirichlet conditions
      FEValuesExtractors::Vector displacement(0);
      ComponentMask mask = fe.component_mask(displacement);
      // TODO: constraints the appropriate boundaries with right components
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               ZeroFunction<dim>(dim+1),
                                               physical_constraints,
                                               mask);
      physical_constraints.close();

      all_constraints.clear();
      all_constraints.reinit(locally_relevant_dofs);
      all_constraints.merge(physical_constraints);
      all_constraints.close();
    }

    { // Setup system matrices and diagonal mass matrix
      system_matrix.clear();
      reduced_system_matrix.clear();

      TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                                relevant_partitioning,
                                                mpi_communicator);
      /*
      Displacements are coupled with one another (upper-left block = true),
      displacements are coupled with phase-field (lower-left block = true),
      phase-field is not coupled with displacements (upper-right block = false),
      phase-field is coupled with itself (lower-right block = true)
      */
      Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);
      for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
          if ((c<dim) || ((c==dim) && (d==dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
                                      physical_constraints,
                                      /* 	keep_constrained_dofs = */ false,
                                      Utilities::MPI::
                                      this_mpi_process(mpi_communicator));
      sp.compress();

      system_matrix.reinit(sp);
      reduced_system_matrix.reinit(sp);

      // Finally assemble the diagonal of the mass matrix
      TrilinosWrappers::BlockSparseMatrix mass_matrix;
      mass_matrix.reinit(sp);
      assemble_mass_matrix_diagonal(mass_matrix);
      mass_matrix_diagonal.reinit(partitioning, relevant_partitioning,
                                  mpi_communicator, /* omit-zeros=*/ true);

      IndexSet::ElementIterator
        index = locally_owned_dofs.begin(),
        end_index = locally_owned_dofs.end();
      for (; index!=end_index; ++index)
        mass_matrix_diagonal(*index) = mass_matrix.diag_element(*index);
      mass_matrix_diagonal.compress(VectorOperation::insert);
    }

    { // Preconditioner matrix
      preconditioner_matrix.clear();

      TrilinosWrappers::BlockSparsityPattern sp(partitioning, partitioning,
                                                relevant_partitioning,
                                                mpi_communicator);

      // only phi-phi entries
      Table<2,DoFTools::Coupling> coupling(dim+1, dim+1);
      for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
          if (c==dim && d==dim)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
                                      physical_constraints,
                                      /* 	keep_constrained_dofs = */ false,
                                      Utilities::MPI::
                                      this_mpi_process(mpi_communicator));

      sp.compress();
      preconditioner_matrix.reinit(sp);
    }

    { // Setup vectors
      solution.reinit(partitioning, relevant_partitioning, mpi_communicator);
      old_solution.reinit(solution);
      old_old_solution.reinit(solution);
      solution_update.reinit(solution);
      rhs_vector.reinit(partitioning, relevant_partitioning,
                        mpi_communicator, /* omit-zeros=*/ true);
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
    Tensor<2,dim>	eps_u_i, eps_u_j;
    Tensor<1,dim> grad_xi_phi_i, grad_xi_phi_j;
    std::vector< Tensor<1,dim> > grad_phi_values(n_q_points);
    // Solution values containers
    std::vector< SymmetricTensor<2,dim> > strain_tensor_values(n_q_points);
    Tensor<2, dim> strain_tensor;
    std::vector<double> phi_values(n_q_points),
                        old_phi_values(n_q_points),
                        old_old_phi_values(n_q_points);
    // Stress decomposition containers
    constitutive_model::EnergySpectralDecomposition<dim> stress_decomposition;
    Tensor<2, dim> stress_tensor_plus, stress_tensor_minus;
    Tensor<2, dim> sigma_u_plus_i, sigma_u_minus_i;

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
          // right_hand_side.value_list(fe_values.get_quadrature_points(),
          //                            rhs_values);
          fe_values[displacement].get_function_symmetric_gradients
            (solution, strain_tensor_values);

          // get old phi solutions for extrapolation
          fe_values[phase_field].get_function_values(solution,
                                                     phi_values);
          fe_values[phase_field].get_function_values(old_solution,
                                                     old_phi_values);
          fe_values[phase_field].get_function_values(old_old_solution,
                                                     old_old_phi_values);
          fe_values[phase_field].get_function_gradients(solution,
                                                        grad_phi_values);

          for (unsigned int q=0; q<n_q_points; ++q) {
            convert_to_tensor(strain_tensor_values[q], strain_tensor);
            stress_decomposition.get_stress_decomposition(strain_tensor,
                                                          data.lame_constant,
                                                          data.shear_modulus,
                                                          stress_tensor_plus,
                                                          stress_tensor_minus);
            // TODO: include time into here
            double d_phi = old_phi_values[q] - old_old_phi_values[q];
            double phi_e = old_phi_values[q] + d_phi;  // extrapolated
            double phi = phi_values[q];
            double jxw = fe_values.JxW(q);


            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                eps_u_i = fe_values[displacement].symmetric_gradient(i, q);

                // that's for the pressure term, don't remove it
                // double div_phi_i = fe_values[displacement].divergence(i, q);

                double xi_phi_i = fe_values[phase_field].value(i ,q);
                grad_xi_phi_i = fe_values[phase_field].gradient(i, q);

                local_rhs[i] +=
                  (
                   ((1 - kappa)*phi_e*phi_e + kappa)*
                     scalar_product(stress_tensor_plus, eps_u_i)
                   + scalar_product(stress_tensor_minus, eps_u_i)
                  +
                   (1 - kappa)*phi*xi_phi_i*
                   scalar_product(stress_tensor_plus, strain_tensor)
                  + gamma_c*(-1/e*(1 - phi)*xi_phi_i +
                             e*grad_phi_values[q]*grad_xi_phi_i)
                   ) * jxw;


                // Find sigma_plus_du, and sigma_minus_du
                stress_decomposition.get_stress_decomposition_derivatives
                  (strain_tensor,
                   eps_u_i,
                   data.lame_constant,
                   data.shear_modulus,
                   sigma_u_plus_i,
                   sigma_u_minus_i);

                // pcout << scalar_product(stress_tensor_plus, eps_u_i) << "\t"
                //       << scalar_product(stress_tensor_minus, eps_u_i) << std::endl;
                // pcout << local_rhs[i] << std::endl;
                // if (std::isnan(sigma_u_plus_i[0][0]))
                //   pcout << "Nan: " << i << std::endl;

                // Assemble local matrix
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  {
                    double xi_phi_j = fe_values[phase_field].value(j ,q);
                    grad_xi_phi_j = fe_values[phase_field].gradient(j, q);
                    eps_u_j = fe_values[displacement].symmetric_gradient(j, q);

                    local_matrix(i, j) +=
                      (
                       ((1-kappa)*phi_e*phi_e + kappa)
                       *scalar_product(sigma_u_plus_i, eps_u_j)
                       +
                       scalar_product(sigma_u_minus_i, eps_u_j)
                       +
                       (1-kappa)*xi_phi_j*
                       (xi_phi_i*
                        scalar_product(stress_tensor_plus, strain_tensor) +
                        2*phi*
                        scalar_product(sigma_u_plus_i, strain_tensor))
                       +
                       gamma_c*(1/e*xi_phi_i*xi_phi_j +
                                e*grad_xi_phi_i*grad_xi_phi_j)
                      ) * jxw;

                    prec_local_matrix(i, j) += (xi_phi_i*xi_phi_j*jxw);

                  }  // end j loop
              }  // end i loop
          }  // end q loop

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

          all_constraints.distribute_local_to_global(prec_local_matrix,
                                                     local_dof_indices,
                                                     preconditioner_matrix);
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
          // break;
        }  // end cell loop

    mass_matrix.compress(VectorOperation::add);
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
        unsigned int i = *index;
        if (residual[i]/mass_matrix_diagonal[i] +
            penalty_parameter*solution_update[i] > 0)
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
    // std::cout << "   Reduced Residual: "
    //           << residual.l2_norm()
    //           << std::endl;
    computing_timer.exit_section();
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::
  impose_displacement(const std::vector<double> &displacement_values)
  {
    computing_timer.enter_section("Imposing displacement values");

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const int n_dirichlet_conditions = displacement_values.size();

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
            {
              int face_boundary_id = cell->face(f)->boundary_id();

            // loop through different boundary labels
            for (int l=0; l<n_dirichlet_conditions; ++l)
              {
                int id = data.displacement_boundary_labels[l];
                if(face_boundary_id == id)
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                    const int component_i = fe.system_to_component_index(i).first;
                    if (component_i == data.displacement_boundary_components[l])
                      solution(local_dof_indices[i]) = displacement_values[i];
                    }  // end of component loop
              } // end loop through components
            } // end if cell @ boundary

    solution.compress(VectorOperation::insert);
    computing_timer.exit_section();
  }  // EOM


  template <int dim>
  void PhaseFieldSolver<dim>::
  solve()
  {
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
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    // The InverseMatrix is used to solve for the mass matrix
    typedef LinearSolvers::
      InverseMatrix<TrilinosWrappers::SparseMatrix,
                    TrilinosWrappers::PreconditionAMG> mp_inverse_t;
    const mp_inverse_t
      mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

    // Construct block preconditioner (for the whole matrix)
    const LinearSolvers::
      BlockDiagonalPreconditioner<TrilinosWrappers::PreconditionAMG, mp_inverse_t>
      preconditioner(prec_A, mp_inverse);

    // set up the linear solver and solve the system
    SolverControl solver_control (system_matrix.m(),
                                  1e-10*rhs_vector.l2_norm());

    SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
      solver(solver_control);

    all_constraints.set_zero(solution_update);
    solver.solve(system_matrix, solution_update, rhs_vector,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;
    all_constraints.distribute(solution_update);

    computing_timer.exit_section();

  }  // EOM

}  // end of namespace
