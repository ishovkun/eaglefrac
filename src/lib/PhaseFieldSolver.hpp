#pragma once

// base dealii modules
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

// Trilinos stuff
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>

// DOF stuff
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

// dealii fem modules
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/sparsity_tools.h>

// standard c++ modules
#include <cmath>        // std:: math functions

// Custom modules
#include <ConstitutiveModel.hpp>
#include <InputData.hpp>
#include <LinearSolver.hpp>


namespace PhaseField
{
using namespace dealii;

template <int dim>
class PhaseFieldSolver
{
public:
// methods
PhaseFieldSolver(MPI_Comm &mpi_communicator,
                 parallel::distributed::Triangulation<dim> &triangulation_,
                 InputData::PhaseFieldData<dim> &data_,
                 ConditionalOStream &pcout_,
                 TimerOutput &computing_timer_);
~PhaseFieldSolver();

void setup_dofs();
double residual_norm() const;

void assemble_system(const TrilinosWrappers::MPI::BlockVector &,
                     const std::pair<double,double> &,
                     bool assemble_matrix=true);
// more simple assembly function interface
void assemble_system(const std::pair<double,double> &);

// assmbly method for coupled (with pressure) system
void
assemble_coupled_system(const TrilinosWrappers::MPI::BlockVector &linerarization_point,
												const TrilinosWrappers::MPI::BlockVector &pressure_relevant_solution,
                				const std::pair<double,double> 					 &time_steps,
                				const bool 															 include_pressure,
                				const bool 															 assemble_matrix);

double compute_nonlinear_residual(const TrilinosWrappers::MPI::BlockVector &,
                                  const std::pair<double, double> &);
// simplified interface
double compute_nonlinear_residual(const std::pair<double, double> &);

void compute_active_set(TrilinosWrappers::MPI::BlockVector &);

void impose_displacement(const std::vector<int>          &,
                         const std::vector<int>          &,
                         const std::vector<double>       &,
                         const std::vector< Point<dim> > &,
                         const std::vector<int>          &,
                         const std::vector<double>        &,
                         const std::vector<bool>          &);
unsigned int solve();
std::pair<unsigned int, unsigned int>
	solve_newton_step(const std::pair<double,double> &time_steps);
void update_old_solution();
bool active_set_changed(const IndexSet &) const;
void truncate_phase_field();
unsigned int active_set_size() const;
void set_coupling(const DoFHandler<dim>            &,
	           			const FESystem<dim>   					 &,
						 			const FEValuesExtractors::Scalar &);
double linear_residual(TrilinosWrappers::MPI::BlockVector &);

private:
// this function also computes finest mesh size
void assemble_mass_matrix_diagonal();
void setup_preconditioners();
void impose_boundary_displacement(const std::vector<int>       &,
                                  const std::vector<int>       &,
                                  const std::vector<double>    &);
void impose_node_displacement(const std::vector < Point<dim> > &,
                              const std::vector<int>           &,
                              const std::vector<double>        &,
                              const std::vector<bool>          &);

public:
// variables
MPI_Comm &mpi_communicator;
parallel::distributed::Triangulation<dim> &triangulation;
InputData::PhaseFieldData<dim> &data;
DoFHandler<dim> dof_handler;

private:
ConditionalOStream &pcout;
TimerOutput &computing_timer;

public:
std::vector<IndexSet> owned_partitioning, relevant_partitioning;

private:
IndexSet locally_owned_dofs, locally_relevant_dofs;

std::vector< std::vector<bool> > constant_modes;      // need for solver

TrilinosWrappers::MPI::BlockVector mass_matrix_diagonal_relevant;
TrilinosWrappers::MPI::BlockVector relevant_residual;

TrilinosWrappers::BlockSparseMatrix system_matrix;
TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

TrilinosWrappers::PreconditionAMG prec_displacement, prec_phase_field;

// Pointers to couple with pore pressure
// these are set by method set_coupling
const DoFHandler<dim> *p_pressure_dof_handler;
const FESystem<dim> *p_pressure_fe;
const FEValuesExtractors::Scalar *p_pressure_extractor;


public:
FESystem<dim> fe;
double time_step;
IndexSet active_set;
TrilinosWrappers::MPI::BlockVector solution, solution_update, residual;
TrilinosWrappers::MPI::BlockVector old_solution, old_old_solution,
                                   relevant_solution;
TrilinosWrappers::MPI::BlockVector rhs_vector;
ConstraintMatrix physical_constraints, all_constraints, hanging_nodes_constraints;
bool use_old_time_step_phi;

};


template <int dim>
PhaseFieldSolver<dim>::PhaseFieldSolver
    (MPI_Comm &mpi_communicator_,
    parallel::distributed::Triangulation<dim> &triangulation_,
    InputData::PhaseFieldData<dim> &data_,
    ConditionalOStream &pcout_,
    TimerOutput &computing_timer_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    data(data_),
    dof_handler(triangulation_),
    pcout(pcout_),
    computing_timer(computing_timer_),
    fe(FE_Q<dim>(1), dim, // displacement components
       FE_Q<dim>(1), 1), // phase-field
    use_old_time_step_phi(false)
{}     // EOM


template <int dim>
PhaseFieldSolver<dim>::~PhaseFieldSolver()
{
  dof_handler.clear();
}    // EOM


template <int dim>
void PhaseFieldSolver<dim>::setup_dofs()
{
  computing_timer.enter_section("Setup phase-field system");

  // Displacements in block 0
  // phase-field in block 1
  std::vector<unsigned int> blocks(dim+1, 0);
  blocks[dim] = 1;

  // distribute and renumber dofs
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler, blocks);

  // Need this stuff for solver
  FEValuesExtractors::Vector displacement(0);
  constant_modes.clear();
  DoFTools::extract_constant_modes(dof_handler,
                                   fe.component_mask(displacement),
                                   constant_modes);

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
  { // compute owned dofs, owned partitioning, and relevant partitioning
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    active_set.clear();
    active_set.set_size(dof_handler.n_locally_owned_dofs());

    owned_partitioning.clear();
    owned_partitioning.resize(2);
    owned_partitioning[0] = locally_owned_dofs.get_view(0, n_u);
    owned_partitioning[1] = locally_owned_dofs.get_view(n_u, n_u+n_phi);

    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

    relevant_partitioning.clear();
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u+n_phi);
  }

  { // constraints
    hanging_nodes_constraints.clear();
    hanging_nodes_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_nodes_constraints);
    hanging_nodes_constraints.close();

    physical_constraints.clear();
    physical_constraints.reinit(locally_relevant_dofs);
    physical_constraints.merge(hanging_nodes_constraints);
    physical_constraints.close();

    all_constraints.clear();
    all_constraints.reinit(locally_relevant_dofs);
    all_constraints.merge(physical_constraints);
    all_constraints.close();
  }

  { // Setup system matrices and diagonal mass matrix
    system_matrix.clear();

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

    // BlockDynamicSparsityPattern sp(dofs_per_block, dofs_per_block);
    TrilinosWrappers::BlockSparsityPattern sp(owned_partitioning,
                                              owned_partitioning,
                                              relevant_partitioning,
                                              mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sp,
                                    // physical_constraints,
                                    hanging_nodes_constraints,
                                    /*  keep_constrained_dofs = */ false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    sp.compress();
    system_matrix.reinit(sp);

    mass_matrix_diagonal_relevant.reinit(relevant_partitioning);
    assemble_mass_matrix_diagonal();
  }

  { // Setup vectors
    solution.reinit(owned_partitioning, mpi_communicator);
    // solution.reinit(relevant_partitioning, mpi_communicator);
    relevant_solution.reinit(relevant_partitioning, mpi_communicator);
    old_solution.reinit(relevant_partitioning, mpi_communicator);
    old_old_solution.reinit(relevant_partitioning, mpi_communicator);
    solution_update.reinit(owned_partitioning, mpi_communicator);
    rhs_vector.reinit(owned_partitioning, relevant_partitioning,
                      mpi_communicator, /* omit-zeros=*/ true);
    residual.reinit(owned_partitioning, mpi_communicator, /* omit-zeros=*/ false);
    relevant_residual.reinit(relevant_partitioning, mpi_communicator);

    active_set.clear();
    active_set.set_size(dof_handler.n_dofs());
  }

  computing_timer.exit_section();
}    // EOM


template <int dim>
void convert_to_tensor(const SymmetricTensor<2, dim> &symm_tensor,
                       Tensor<2, dim>                &tensor)
{
  for (int i=0; i<dim; ++i)
    for (int j=0; j<dim; ++j)
      tensor[i][j] = symm_tensor[i][j];
}    // EOM


template <int dim> inline
double sum(const Tensor<1, dim> &t)
{
	double s = 0;
  for (int i=0; i<dim; ++i)
		s += t[i];
	return s;
}    // EOM


template <int dim>
void PhaseFieldSolver<dim>::
assemble_system(const std::pair<double,double> &time_steps)
{
  assemble_system(solution, time_steps, true);
}    // eom


template <int dim>
void PhaseFieldSolver<dim>::
set_coupling(const DoFHandler<dim>            &pressure_dof_handler,
	           const FESystem<dim>   						&pressure_fe,
						 const FEValuesExtractors::Scalar &pressure_extractor)
{
	p_pressure_fe = &pressure_fe;
	p_pressure_dof_handler = &pressure_dof_handler;
	p_pressure_extractor = &pressure_extractor;
}  // eom


template <int dim>
void PhaseFieldSolver<dim>::
assemble_coupled_system(const TrilinosWrappers::MPI::BlockVector &linerarization_point,
												const TrilinosWrappers::MPI::BlockVector &pressure_relevant_solution,
                				const std::pair<double,double> 					 &time_steps,
                				const bool 															 include_pressure,
                				const bool 															 assemble_matrix)
{
	if (include_pressure)
		AssertThrow(p_pressure_dof_handler != NULL,
		            ExcMessage("pointer to pressure dof_handler is null"));

  if (assemble_matrix)
    computing_timer.enter_section("Assemble system");
  else
    computing_timer.enter_section("Assemble nonlinear residual");

  const QGauss<dim> quadrature_formula(fe.degree+2);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points |
                          update_JxW_values);

	// dummy unless include_pressure=true
	FEValues<dim> *p_pressure_fe_values = NULL;
	if (include_pressure)
    p_pressure_fe_values = new FEValues<dim>(*p_pressure_fe, quadrature_formula,
	                     												update_values | update_gradients);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>   local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Scalar phase_field(dim);

  // FeValues containers
  std::vector< Tensor<2,dim> > eps_u(dofs_per_cell);
  std::vector< Tensor<2,dim> > grad_xi_u(dofs_per_cell);
  std::vector< Tensor<1,dim> > grad_xi_phi(dofs_per_cell);
  std::vector< Tensor<1,dim> > grad_phi_values(n_q_points);
  std::vector<double>  				 xi_phi(dofs_per_cell);

  // Solution values containers
  std::vector< Tensor<2, dim> > grad_u_values(n_q_points);
  Tensor<2, dim> 								strain_tensor_value;
  std::vector<double> 					phi_values(n_q_points),
						  									old_phi_values(n_q_points),
						  									old_old_phi_values(n_q_points);

	// std::vector
	std::vector<double> pressure_values(n_q_points);
	// std::vector< std::vector<double> > pressure_gradients(n_q_points,)

  // Stress decomposition containers
  ConstitutiveModel::EnergySpectralDecomposition<dim> stress_decomposition;
  Tensor<2, dim> stress_tensor_plus, stress_tensor_minus;
  std::vector< Tensor<2, dim> >
  	sigma_u_plus(dofs_per_cell),
  	sigma_u_minus(dofs_per_cell);

  // Equation data
  double kappa = data.regularization_parameter_kappa;
  double e = data.regularization_parameter_epsilon;

  relevant_solution = linerarization_point;

  if (assemble_matrix)
  {
    system_matrix = 0;
    rhs_vector = 0;
  }
  else
    residual = 0;

	// Pressure cell is either taken from solid dof_handler (no pressure terms)
	// or from pressure dof handler (include_pressure = true)
  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler.begin_active(),
	  pressure_cell = dof_handler.begin_active(),
	  endc = dof_handler.end();

	if (include_pressure)
  	pressure_cell = p_pressure_dof_handler->begin_active();

  for (; cell!=endc; ++cell, ++pressure_cell)
    if (cell->is_locally_owned())
    {
      local_rhs = 0;
      local_matrix = 0;

      fe_values.reinit(cell);
			if (include_pressure)
				p_pressure_fe_values->reinit(pressure_cell);

      fe_values[phase_field].get_function_values(relevant_solution,
                                                 phi_values);
      fe_values[phase_field].get_function_values(old_solution,
                                                 old_phi_values);
      fe_values[phase_field].get_function_values(old_old_solution,
                                                 old_old_phi_values);
      fe_values[phase_field].get_function_gradients(relevant_solution,
                                                    grad_phi_values);
			fe_values[displacement].get_function_gradients(relevant_solution,
																										 grad_u_values);

			if (include_pressure)
			{
				(*p_pressure_fe_values)[*p_pressure_extractor].get_function_values
					(pressure_relevant_solution, pressure_values);
				// pcout << "Pressure " << pressure_values[2] << std::endl;
			}

      double G_c = data.get_fracture_toughness->value(cell->center(), 0);
      // pcout << "g_c " << G_c << std::endl;

      for (unsigned int q=0; q<n_q_points; ++q)
      {
        // convert from SymmetricTensor to Tensor
        // convert_to_tensor(strain_tensor_values[q], strain_tensor_value);
				strain_tensor_value = 0.5*(grad_u_values[q] + transpose(grad_u_values[q]));

        // Simple splitting
        // stress_decomposition.get_stress_decomposition(strain_tensor_value,
        //                                               data.lame_constant,
        //                                               data.shear_modulus,
        //                                               stress_tensor_plus,
        //                                               stress_tensor_minus);

        // Spectral decomposition
        stress_decomposition.stress_spectral_decomposition(strain_tensor_value,
                                                           data.lame_constant,
                                                           data.shear_modulus,
                                                           stress_tensor_plus,
                                                           stress_tensor_minus);

        // we get nans at the first time step
        if (!numbers::is_finite(trace(stress_tensor_plus)))
          stress_decomposition.get_stress_decomposition(strain_tensor_value,
                                                        data.lame_constant,
                                                        data.shear_modulus,
                                                        stress_tensor_plus,
                                                        stress_tensor_minus);

        double phi_value = phi_values[q];
        double old_phi_value = old_phi_values[q];
        double old_old_phi_value = old_old_phi_values[q];
        const double time_step = time_steps.first;
        const double old_time_step = time_steps.second;
        double dphi_dt_old = (old_phi_value - old_old_phi_value)/old_time_step;

        double phi_tilda = old_phi_value + dphi_dt_old*time_step;
        phi_tilda = std::max(std::min(1.0, phi_tilda), 0.0);

        if (use_old_time_step_phi)
          phi_tilda = old_phi_value;

        double jxw = fe_values.JxW(q);

        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          xi_phi[k]      = fe_values[phase_field].value(k,q);
          grad_xi_phi[k] = fe_values[phase_field].gradient(k, q);
					grad_xi_u[k]   = fe_values[displacement].gradient(k, q);
					eps_u[k] = 0.5*(grad_xi_u[k] + transpose(grad_xi_u[k]));

          if (assemble_matrix)
          {
            // No decomposition
            // sigma_u_plus[k] =
            //         data.lame_constant*trace(eps_u[k])*identity_tensor
            //         + 2*data.shear_modulus*eps_u[k];
            // sigma_u_minus[k] = 0;

            // Spectral decomposition
            stress_decomposition.stress_spectral_decomposition_derivatives
              (strain_tensor_value,
               eps_u[k],
               data.lame_constant,
               data.shear_modulus,
               sigma_u_plus[k],
               sigma_u_minus[k]);

            // we get nans at the first time step
            // simple splitting
            if (!numbers::is_finite(trace(sigma_u_plus[k])))
              stress_decomposition.get_stress_decomposition_derivatives
                (strain_tensor_value,
                 eps_u[k],
                 data.lame_constant,
                 data.shear_modulus,
                 sigma_u_plus[k],
                 sigma_u_minus[k]);
          }
        } // end k loop

              // Assemble local rhs +
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          double rhs_u =
                  ((1.-kappa)*phi_tilda*phi_tilda + kappa)
                    *scalar_product(stress_tensor_plus, eps_u[i])
                  + scalar_product(stress_tensor_minus, eps_u[i]);
				  if (include_pressure)
						rhs_u +=
							-(data.biot_coef-1.0)*phi_tilda*phi_tilda*pressure_values[q]*
							// trace(grad_xi_u[i]);
							(grad_xi_u[i][0][0] + grad_xi_u[i][1][1]);

          double rhs_phi =
                  (1.-kappa)*phi_value*xi_phi[i]
                  *scalar_product(stress_tensor_plus, strain_tensor_value)
                  - G_c/e*(1.-phi_value)*xi_phi[i]
                  + G_c*e
                    *scalar_product(grad_phi_values[q], grad_xi_phi[i]);
				  if (include_pressure)
					{
						// pcout << "Include pressure " << std::endl;
						rhs_phi +=
							-2.0*(data.biot_coef-1.0)*phi_value*pressure_values[q] *
							// trace(grad_u_values[q])*xi_phi[i];
							(grad_u_values[q][0][0] + grad_u_values[q][1][1])*xi_phi[i];
					}

          local_rhs[i] -= (rhs_u + rhs_phi)*jxw;

        } // end i loop

        // Assemble local matrix
        if (assemble_matrix)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              double m_u_u =
                  ((1.-kappa)*phi_tilda*phi_tilda + kappa)
                   *scalar_product(sigma_u_plus[j], eps_u[i])
                  + scalar_product(sigma_u_minus[j], eps_u[i]);

              // old (from paper) - probably wrong
              // double m_phi_u =
              //         2.*(1.-kappa)*phi_value
              //         *scalar_product(sigma_u_plus[j], strain_tensor_value)
              //         *xi_phi[i];

              // new
              double m_phi_u =
                      (1.-kappa)*phi_value*
                        (  scalar_product(sigma_u_plus[j], strain_tensor_value)
                         + scalar_product(stress_tensor_plus, eps_u[j]))
                        *xi_phi[i];
							if (include_pressure)
								{
									// pcout << "INcluding pressure" << std::endl;
									m_phi_u +=
										-2.0*(data.biot_coef-1.0)*pressure_values[q]*
										// phi_value*trace(grad_xi_u[j])*xi_phi[i];
										phi_value*(grad_xi_u[j][0][0] + grad_xi_u[j][1][1])*xi_phi[i];
								}

              // double m_u_phi = 0;

              double m_phi_phi =
                (1.-kappa)*xi_phi[j]*xi_phi[i]
                  *scalar_product(stress_tensor_plus, strain_tensor_value)
                + G_c/e*(xi_phi[j]*xi_phi[i])
                + G_c*e*scalar_product(grad_xi_phi[j], grad_xi_phi[i]);
							if (include_pressure)
								m_phi_phi +=
									-2.0*(data.biot_coef-1.0)*pressure_values[q]*
									// trace(grad_u_values[q])*xi_phi[j]*xi_phi[i];
									(grad_u_values[q][0][0] + grad_u_values[q][1][1])*xi_phi[j]*xi_phi[i];

              local_matrix(i, j) += (m_u_u + m_phi_u + m_phi_phi) * jxw;

            } // end i&j loop
      } // end q loop

      cell->get_dof_indices(local_dof_indices);

      if (assemble_matrix)
        all_constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   rhs_vector);
      else
        hanging_nodes_constraints.distribute_local_to_global(local_rhs,
                                                             local_dof_indices,
                                                             residual);

    } // end of cell loop

  if (assemble_matrix)
  {
    system_matrix.compress(VectorOperation::add);
    rhs_vector.compress(VectorOperation::add);
  }
  else
    residual.compress(VectorOperation::add);

  computing_timer.exit_section();

  if (assemble_matrix)
    setup_preconditioners();

  delete p_pressure_fe_values;
}  // eom

template <int dim>
void PhaseFieldSolver<dim>::
assemble_system(const TrilinosWrappers::MPI::BlockVector &linerarization_point,
                const std::pair<double,double> &time_steps,
                bool assemble_matrix)
{
	/* This funciton is for the assembly of only the solid system (with no fluid)
	It calls a generic function that assembles the coupled system but with dummy
	pressure variables and a false include_pressure flag. */

	// First create dummy variables to pass to the coupled (with pressure) generic
	// funciton
	TrilinosWrappers::MPI::BlockVector dummy_pressure_vector;

	assemble_coupled_system(linerarization_point,
													dummy_pressure_vector,
													time_steps,
													/*include pressure = */ false,
													assemble_matrix);

}   // EOM


template <int dim>
void PhaseFieldSolver<dim>::
assemble_mass_matrix_diagonal()
{
  // Assert (fe.degree == 1, ExcNotImplemented());
  computing_timer.enter_section("Assemble mass matrix diagonal");

  // const QTrapez<dim> quadrature_formula;
  QGaussLobatto<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_quadrature_points |
                          update_values |
                          update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  TrilinosWrappers::MPI::BlockVector mass_matrix_diagonal;
  mass_matrix_diagonal.reinit(owned_partitioning, mpi_communicator);
  mass_matrix_diagonal = 0;

  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      cell_rhs = 0;
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i == dim)
            cell_rhs(i) += (fe_values.shape_value(i, q_point) *
                            fe_values.shape_value(i, q_point) *
                            fe_values.JxW(q_point));
        }

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        mass_matrix_diagonal(local_dof_indices[i]) += cell_rhs(i);
    } // end cell loop

  mass_matrix_diagonal.compress(VectorOperation::add);

  mass_matrix_diagonal_relevant = mass_matrix_diagonal;

  computing_timer.exit_section();

}    // EOM


template <int dim>
void PhaseFieldSolver<dim>::
compute_active_set(TrilinosWrappers::MPI::BlockVector &linerarization_point)
{
  computing_timer.enter_section("Computing active set");
  active_set.clear();
  all_constraints.clear();
  all_constraints.reinit(locally_relevant_dofs);

  std::vector<bool> dof_touched(dof_handler.n_dofs(), false);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  relevant_residual = residual;
  relevant_solution = linerarization_point;

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell != endc; ++cell)
    if (!cell->is_artificial())
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        const int component = fe.system_to_component_index(i).first;
        const unsigned int index = local_dof_indices[i];

        if (component == dim
            && dof_touched[index] == false
            && !hanging_nodes_constraints.is_constrained(index)
            // && locally_owned_dofs.is_element(index)
            )
        {
          dof_touched[index] = true;

          double gap = relevant_solution[index] - old_solution[index];
          double res = relevant_residual[index];
          double mass_diag = mass_matrix_diagonal_relevant[index];

          if (res/mass_diag + data.penalty_parameter*gap > 0 )
          {
            active_set.add_index(index);
            all_constraints.add_line(index);
            all_constraints.set_inhomogeneity(index, 0.0);
            linerarization_point(index) = old_solution(index);
          } // end if in active set
        } // end if touched
      } // end dof loop
    } // end cell loop

  linerarization_point.compress(VectorOperation::insert);
  // we might have changed values of the solution, so fix the
  // hanging nodes (we ignore in the active set):
  hanging_nodes_constraints.distribute(linerarization_point);

  // since physical_constraints may fix the phase_field, because
  // we now may constrain phase field in nodes, we merge with the
  // priority of physical constraints
  // all_constraints.merge(physical_constraints);
  all_constraints.merge(physical_constraints,
                        ConstraintMatrix::right_object_wins);
  all_constraints.close();

  computing_timer.exit_section();
}    // EOM


template <int dim>
void PhaseFieldSolver<dim>::update_old_solution()
{
        old_old_solution = old_solution;
        old_solution = solution;
}    // eom

template <int dim>
std::pair<unsigned int, unsigned int>
PhaseFieldSolver<dim>::
solve_newton_step(const std::pair<double,double> &time_steps)
{
    assemble_system(solution, time_steps, true);
    // abort();
    unsigned int n_gmres = solve();

    // line search
    TrilinosWrappers::MPI::BlockVector tmp_vector = solution;
    double old_error = compute_nonlinear_residual(solution, time_steps);
    const int max_steps = 10;
    double damping = 0.6;
		unsigned int n_steps = 0;
    for (int step = 0; step < max_steps; ++step)
    {
			n_steps++;
      solution += solution_update;
      compute_nonlinear_residual(solution, time_steps);
      all_constraints.set_zero(residual);
      double error = residual.l2_norm();

      if (error < old_error)
        break;

      if (step < max_steps)
      {
        // solution = relevant_solution;
        solution = tmp_vector;
        solution_update *= damping;
      }
    } // end line search

		std::pair<unsigned int, unsigned int> solver_results =
			std::make_pair(n_gmres, n_steps);
		return solver_results;
}    // eom


template <int dim>
bool PhaseFieldSolver<dim>::
active_set_changed(const IndexSet &old_active_set) const
{
  return Utilities::MPI::sum((active_set == old_active_set) ? 0 : 1,
                             mpi_communicator) == 0;
}    // eom


template <int dim>
unsigned int PhaseFieldSolver<dim>::active_set_size() const
{
  return Utilities::MPI::sum((active_set & locally_owned_dofs).n_elements(),
                             mpi_communicator);
}    // eom


template <int dim>
double PhaseFieldSolver<dim>::
compute_nonlinear_residual(const TrilinosWrappers::MPI::BlockVector &linerarization_point,
                           const std::pair<double,double> &time_steps)
{
  assemble_system(linerarization_point, time_steps, false);
  return residual.l2_norm();
}    // EOM


template <int dim>
double PhaseFieldSolver<dim>::
compute_nonlinear_residual(const std::pair<double,double> &time_steps)
{
  return compute_nonlinear_residual(solution, time_steps);
}    // EOM


template <int dim>
double PhaseFieldSolver<dim>::residual_norm() const
{
  return residual.l2_norm();
}    // eom


template <int dim>
void PhaseFieldSolver<dim>::truncate_phase_field()
{
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
        const unsigned int comp_i = fe.system_to_component_index(i).first;
        if (comp_i == dim)
        {
          const unsigned int idx = local_dof_indices[i];
          if (dof_handler.locally_owned_dofs().is_element(idx))
            solution(idx) =
                std::max(0.0, std::min(static_cast<double>(solution(idx)), 1.0));
        }
      }
    }

  solution.compress(VectorOperation::insert);
}    // eom


template <int dim>
void PhaseFieldSolver<dim>::
impose_node_displacement(const std::vector < Point<dim> > &displacement_points,
                         const std::vector<int>           &displacement_point_components,
                         const std::vector<double>        &displacement_point_values,
                         const std::vector<bool>          &constraint_point_phase_field)
{
  // loop throught vertices and find nodes that are closest to points
  // store cells that are closest
  const int n_displacement_points = displacement_points.size();
  std::vector< Point<dim> > closest_vertex_coordinates;

  // set unrealistic cell number and distance to compare with later on
  std::vector<double> min_distances(n_displacement_points);
  std::vector<types::global_dof_index> closest_vertex_idx(n_displacement_points);
  // if we want to constrain phase-field in a displacement node
  std::vector<types::global_dof_index> constrained_phi_nodes(n_displacement_points);
  for (int i=0; i<n_displacement_points; ++i)
  {
    min_distances[i] = std::numeric_limits<double>::max();
    closest_vertex_idx[i] = std::numeric_limits<unsigned int>::max();
  }

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell != endc; ++cell)
  {
    if (cell->is_artificial())
      continue;

    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      for (int p=0; p<n_displacement_points; ++p)
      {
        double distance = 0;
        for (int c=0; c<dim; ++c)
        {
          double tmp = cell->vertex(v)[c] - displacement_points[p](c);
          distance += tmp*tmp;
        }  // end component loop

        // check if point is the closest so far
        if (distance < min_distances[p])
        {
          types::global_dof_index idx =
              cell->vertex_dof_index(v, displacement_point_components[p]);
          closest_vertex_idx[p] = idx;
          min_distances[p] = distance;
          if (constraint_point_phase_field[p])
          {
            idx = cell->vertex_dof_index(v, dim);  // phase-field
            constrained_phi_nodes[p] = idx;
          }
        }
      }  // end target point loop
    } // end vertex loop
  }  // end cell loop

  // now we need to make sure that only the process with the closest cells
  // gets to impose the BC's
  for (int p=0; p<n_displacement_points; ++p)
  {
    // std::cout << constraint_point_phase_field[p] << std::endl;
    double global_min_distance = Utilities::MPI::min(min_distances[p],
                                                     mpi_communicator);
    if (global_min_distance == min_distances[p])
      {
        solution[closest_vertex_idx[p]] = displacement_point_values[p];
        physical_constraints.add_line(closest_vertex_idx[p]);
        if (constraint_point_phase_field[p])
        {
          solution[constrained_phi_nodes[p]] = 1;
          // std::cout << "constraining dof" << std::endl;
          physical_constraints.add_line(constrained_phi_nodes[p]);
        }
      }
    solution.compress(VectorOperation::insert);
    // Don't close constraints here: done in impose_displacement
  }
}  // eom


template <int dim>
void PhaseFieldSolver<dim>::
impose_boundary_displacement(const std::vector<int>        &boundary_ids,
                             const std::vector<int>        &displacement_components,
                             const std::vector<double>     &displacement_values)
{

    // Extract displacement components
    std::vector<FEValuesExtractors::Scalar> displacement_masks(dim);
    for (int i=0; i<dim; ++i)
    {
      const FEValuesExtractors::Scalar comp(i);
      displacement_masks[i] = comp;
    }

    // container for displacement boundary values
    std::map<types::global_dof_index, double> boundary_values;
    int n_dirichlet_conditions = boundary_ids.size();

    // Store BC's in the container "boundary_values"
    for (int cond=0; cond<n_dirichlet_conditions; ++cond)
    {
      const int component = displacement_components[cond];
      double dirichlet_value = displacement_values[cond];
      // impose into boundary_values (for solution vector)
      VectorTools::interpolate_boundary_values
              (dof_handler,
              boundary_ids[cond],
              ConstantFunction<dim>(dirichlet_value, dim+1),
              boundary_values,
              fe.component_mask(displacement_masks[component]));
       // impose into constraint (for solution_update)
       VectorTools::interpolate_boundary_values
               (dof_handler,
               data.displacement_boundary_labels[cond],
               ZeroFunction<dim>(dim+1),
               physical_constraints,
               fe.component_mask(displacement_masks[component]));
    }

    // Apply BC values to the solution vector
    for (std::map<types::global_dof_index,double>::const_iterator
         p = boundary_values.begin();
         p != boundary_values.end(); ++p)
            solution(p->first) = p->second;

    solution.compress(VectorOperation::insert);
}    // EOM


template <int dim>
void PhaseFieldSolver<dim>::
impose_displacement(const std::vector<int>          &boundary_ids,
                    const std::vector<int>          &boundary_components,
                    const std::vector<double>       &boundary_values,
                    const std::vector< Point<dim> > &points,
                    const std::vector<int>          &point_components,
                    const std::vector<double>       &point_values,
                    const std::vector<bool>         &constraint_point_phase_field)
{
  computing_timer.enter_section("Imposing displacement values");
  physical_constraints.clear();
  physical_constraints.reinit(locally_relevant_dofs);
  physical_constraints.merge(hanging_nodes_constraints);
  impose_boundary_displacement(boundary_ids, boundary_components,
                               boundary_values);
  impose_node_displacement(points, point_components, point_values,
                           constraint_point_phase_field);
  physical_constraints.close();
  all_constraints.clear();
  all_constraints.reinit(locally_relevant_dofs);
  all_constraints.merge(physical_constraints);
  all_constraints.close();
  computing_timer.exit_section();
}  // eom


template <int dim>
void PhaseFieldSolver<dim>::setup_preconditioners()
{
        // Preconditioner for the displacement (0, 0) block
        // TrilinosWrappers::PreconditionAMG prec_displacement;
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData data;
          data.constant_modes = constant_modes;
          data.elliptic = true;
          data.higher_order_elements = true;
          data.smoother_sweeps = 2;
          data.aggregation_threshold = 0.02;
          prec_displacement.initialize(system_matrix.block(0, 0), data);
        }

        // Preconditioner for the phase-field (1, 1) block
        // TrilinosWrappers::PreconditionAMG prec_phase_field;
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData data;
          data.elliptic = true;
          data.higher_order_elements = true;
          data.smoother_sweeps = 2;
          data.aggregation_threshold = 0.02;
          prec_phase_field.initialize(system_matrix.block(1, 1), data);
        }
}    // eom


template <int dim>
unsigned int PhaseFieldSolver<dim>::solve()
{
  /*
     In this method we essentially use 2 block diagonal preconditioners
     for the block (0,0) and the block (1, 1)
   */
  computing_timer.enter_section("Solve phase-field system");

  // Construct block preconditioner (for the whole matrix)
  const LinearSolvers::
  BlockDiagonalPreconditioner<TrilinosWrappers::PreconditionAMG,
                              TrilinosWrappers::PreconditionAMG>
  preconditioner(prec_displacement, prec_phase_field);

  // set up the linear solver and solve the system
  unsigned int max_iter = system_matrix.m();
  // pcout << "rhs norm" << rhs_vector.l2_norm() << "\t";
  // double tol = std::max(1e-10*rhs_vector.l2_norm(), 1e-14);
  double tol = 1e-10*rhs_vector.l2_norm();
  SolverControl solver_control(max_iter, tol);

  // SolverFGMRES<TrilinosWrappers::MPI::BlockVector>
  SolverGMRES<TrilinosWrappers::MPI::BlockVector>
  solver(solver_control);

  solver.solve(system_matrix, solution_update,
               rhs_vector, preconditioner);

  all_constraints.distribute(solution_update);

  computing_timer.exit_section();

	return solver_control.last_step();
}    // EOM


template <int dim>
double
PhaseFieldSolver<dim>::
linear_residual(TrilinosWrappers::MPI::BlockVector &dst)
{
	return system_matrix.residual(dst, solution_update, rhs_vector);
}

}  // end of namespace
