// #pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
// #include <deal.II/base/function.h>
// #include <deal.II/base/tensor.h>

// Trilinos stuff
#include <deal.II/lac/generic_linear_algebra.h>
// #include <deal.II/lac/solver_gmres.h>
// #include <deal.II/lac/trilinos_solver.h>
// #include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_vector.h>

// DOF stuff
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
// #include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

// dealii fem modules
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/sparsity_tools.h>

#include <SinglePhaseData.hpp>


namespace FluidSolvers
{
	using namespace dealii;

	template <int dim>
	class PressureSolver
	{
	public:
		PressureSolver(MPI_Comm                                  &mpi_communicator_,
                 	 parallel::distributed::Triangulation<dim> &triangulation_,
                 	 const InputData::SinglePhaseData<dim>     &data_,
									 const DoFHandler<dim>                     &dof_handler_solid_,
									 const FESystem<dim>                       &fe_solid_,
                   ConditionalOStream                        &pcout_,
                 	 TimerOutput                               &computing_timer_);
		~PressureSolver();
		void setup_dofs();
		void assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_solid,
												 const TrilinosWrappers::MPI::BlockVector &old_solution_solid,
												 const double time_step);
		const DoFHandler<dim> &get_dof_handler();
		const FESystem<dim>   &get_fe();
		const ConstraintMatrix &get_constraint_matrix();
		double solve();

	private:
		// these guys are passed at initialization
		MPI_Comm 																	&mpi_communicator;
		parallel::distributed::Triangulation<dim> &triangulation;
		const InputData::SinglePhaseData<dim> 		&data;
		DoFHandler<dim> 													dof_handler;
		// Pointers to couple with phase_field_solver
		// these are set by method set_coupling
		const DoFHandler<dim>            					&dof_handler_solid;
		const FESystem<dim> 						 					&fe_solid;
		// auxilary objects
		ConditionalOStream 												&pcout;
		TimerOutput 			 												&computing_timer;

		FESystem<dim> fe;
		double init_pressure;
		ConstraintMatrix constraints;

		TrilinosWrappers::MPI::BlockVector  rhs_vector;
		TrilinosWrappers::BlockSparseMatrix system_matrix;

	public:
		TrilinosWrappers::MPI::BlockVector solution, relevant_solution;
		TrilinosWrappers::MPI::BlockVector old_solution;
		std::vector<IndexSet> owned_partitioning, relevant_partitioning;
	};


	template <int dim>
	PressureSolver<dim>::PressureSolver
  (MPI_Comm                                  &mpi_communicator_,
   parallel::distributed::Triangulation<dim> &triangulation_,
   const InputData::SinglePhaseData<dim>     &data_,
	 const DoFHandler<dim>                     &dof_handler_solid_,
	 const FESystem<dim>                       &fe_solid_,
   ConditionalOStream                        &pcout_,
   TimerOutput                               &computing_timer_)
	:
  mpi_communicator(mpi_communicator_),
  triangulation(triangulation_),
  data(data_),
  dof_handler(triangulation_),
	dof_handler_solid(dof_handler_solid_),
	fe_solid(fe_solid_),
  pcout(pcout_),
  computing_timer(computing_timer_),
  fe(FE_Q<dim>(1), 1), // one linear pressure component
	init_pressure(0.0)
	{}  // eom


	template <int dim>
	PressureSolver<dim>::~PressureSolver()
	{
  	dof_handler.clear();
	}

	template <int dim> void
	PressureSolver<dim>::setup_dofs()
	{
		dof_handler.distribute_dofs(fe);

		IndexSet locally_owned_dofs, locally_relevant_dofs;
		locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);
    // pressure_owned_solution.reinit(locally_owned_dofs, mpi_communicator);

		owned_partitioning.clear();
		relevant_partitioning.clear();
		owned_partitioning.push_back(locally_owned_dofs);
		relevant_partitioning.push_back(locally_relevant_dofs);

		{ // Constraints
			// only hanging nodes for now
			constraints.clear();
			constraints.reinit(locally_relevant_dofs);
	    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    	constraints.close();
		}

		// Vectors and matrices
		relevant_solution.reinit(relevant_partitioning, mpi_communicator);
		solution.reinit(owned_partitioning, mpi_communicator);
		old_solution.reinit(relevant_partitioning, mpi_communicator);
    rhs_vector.reinit(owned_partitioning, relevant_partitioning,
                      mpi_communicator, /* omit-zeros=*/ true);

    TrilinosWrappers::BlockSparsityPattern sp(owned_partitioning,
                                              owned_partitioning,
                                              relevant_partitioning,
                                              mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler, sp, constraints,
                                    /*  keep_constrained_dofs = */ false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    sp.compress();
    system_matrix.reinit(sp);
	}  // eom

	template <int dim> void
	PressureSolver<dim>::
	assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_solid,
									const TrilinosWrappers::MPI::BlockVector &old_solution_solid,
									const double                       time_step)
	{
    computing_timer.enter_section("Assmeble pressure system");

  	const QGauss<dim> quadrature_formula(fe.degree+2);
  	FEValues<dim> fe_values(fe, quadrature_formula,
                          	update_values | update_gradients |
                          	update_quadrature_points |
                          	update_JxW_values);
  	FEValues<dim> fe_values_solid(fe_solid, quadrature_formula,
                          				update_values | update_gradients);

		const FEValuesExtractors::Vector displacement(0);
		const FEValuesExtractors::Scalar phase_field(dim);
		const FEValuesExtractors::Scalar pressure(0);

	  const unsigned int dofs_per_cell = fe.dofs_per_cell;
	  const unsigned int n_q_points    = quadrature_formula.size();

	  FullMatrix<double>   local_matrix(dofs_per_cell, dofs_per_cell);
	  Vector<double>       local_rhs(dofs_per_cell);

  	std::vector<double>  				 phi_values(n_q_points);
  	std::vector<double>  				 p_values(n_q_points);
  	std::vector<double>  				 old_p_values(n_q_points);
		std::vector< Tensor<1,dim> > u_values(n_q_points);
		std::vector<double>  				 div_u_values(n_q_points);
		std::vector<double>  				 div_old_u_values(n_q_points);
		// std::vector< Tensor<1,dim> > old_u_values(n_q_points);

		std::vector<double>  				 xi_p(dofs_per_cell);
  	std::vector< Tensor<1,dim> > grad_xi_p(dofs_per_cell);

  	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		system_matrix = 0;
		rhs_vector = 0;

	  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  cell_solid = dof_handler_solid.begin_active(),
		  endc = dof_handler.end();

	  for (; cell!=endc; ++cell, ++cell_solid)
	    if (cell->is_locally_owned())
			{
				local_matrix = 0;
				local_rhs = 0;

				fe_values.reinit(cell);
				fe_values_solid.reinit(cell_solid);

	      fe_values_solid[phase_field].get_function_values(solution_solid,
                                         								 phi_values);
	      fe_values_solid[displacement].get_function_values(solution_solid,
                                         								  u_values);
	      fe_values_solid[displacement].get_function_divergences(solution_solid,
                                         								  	 	 div_u_values);
	      fe_values_solid[displacement].get_function_divergences(old_solution_solid,
                                         								  		 div_old_u_values);
	      fe_values[pressure].get_function_values(relevant_solution,
                               								  p_values);
	      fe_values[pressure].get_function_values(old_solution,
                               								  old_p_values);


	      double E = data.get_young_modulus->value(cell->center(), 0);
				double nu = data.get_poisson_ratio->value(cell->center(), 0);
				double bulk_modulus = E/3./(1.-2.*nu);
				double grain_bulk_modulus = bulk_modulus/(1.-data.biot_coef);
				double modulus_N = grain_bulk_modulus*(data.biot_coef - data.porosity);
				double modulus_M = (modulus_N/data.fluid_compressibility) /
													 (modulus_N*data.porosity + 1./data.fluid_compressibility);

				// auto  & quadrature_points = fe_values.get_quadrature_points();
				double source_term = 0;
				for (unsigned int k=0; k<data.wells.size(); ++k)
					source_term += data.wells[k]->value(cell->center(), 0);

				// pcout << "poro " << data.porosity << std::endl;
				// pcout << "c_fluid " << data.fluid_compressibility << std::endl;
				// pcout << "perm_res " << data.perm_res << std::endl;
				// pcout << "viscosity " << data.fluid_viscosity << std::endl;
				// pcout << "c_f " << data.fracture_compressibility << std::endl;
				// pcout << "density " << data.fluid_density << std::endl;
				// pcout << "M modulus " << modulus_M << std::endl;

				for (unsigned int q=0; q<n_q_points; ++q)
				{
					double cx = 0.1;
					double c1 = 0.5 - cx;
					double c2 = 0.5 + cx;
					// this is a simplistic way to compute frac width but is good for now
					double w = u_values[q].norm();  // absolute value
					// if (w<1e-15) w = 1e-5;
					double perm_f = 1.0/12.0*w*w;   // fracture perm from lubrication theory
					perm_f = 50*data.perm_res;   // fracture perm from lubrication theory

					// Indicator function
					double xi_f = (c2 - phi_values[q])/(c2 - c1);
					double xi_r = (phi_values[q] - c1)/(c2 - c1);

					if (phi_values[q] <= c1)
					{
						xi_f = 1.0;
						xi_r = 0.0;
					}
					if (phi_values[q] >= c2)
					{
						xi_f = 0.0;
						xi_r = 1.0;
					}

					double div_delta_u = div_u_values[q] - div_old_u_values[q];

					// double source_term = 0;
					// for (unsigned int k=0; k<data.wells.size(); ++k)
					// 	source_term += data.wells[k]->value(quadrature_points[q], 0);
					// pcout << "source " << source_term << std::endl;

					// if (source_term > 0)
					// {
					// 	pcout << "source " << source_term << std::endl;
					// 	pcout << "phi " << phi_values[q] << std::endl;
					// 	pcout << "xi_f " << xi_f << std::endl;
					// 	pcout << "xi_r " << xi_r << std::endl;
					// 	pcout << "point " << cell->center() << std::endl;
					// }
					// interpolate pereability
					// double K_eff = (perm_f*xi_f + data.perm_res)/data.fluid_viscosity;


					// pcout << "xi_f " << xi_f << std::endl;
					// pcout << "xi_r " << xi_r << std::endl;
					// pcout << "phi " << phi_values[q] << std::endl;

					// compute some fe-dependent props
	        for (unsigned int k=0; k<dofs_per_cell; ++k)
					{
						xi_p[k] = fe_values[pressure].value(k, q);
						grad_xi_p[k] = fe_values[pressure].gradient(k, q);
						// pcout << "xi_p " << xi_p[q] << std::endl;
					}  // end k loop

	        for (unsigned int i=0; i<dofs_per_cell; ++i)
					{
	        	for (unsigned int j=0; j<dofs_per_cell; ++j)
						{
							double m_r =
								data.fluid_density/time_step *
								(1./modulus_M + data.biot_coef*data.biot_coef/bulk_modulus) *
								xi_p[j]*xi_p[i]
								+
								data.perm_res*data.fluid_density/data.fluid_viscosity *
								grad_xi_p[j]*grad_xi_p[i];

							double m_f =
								data.fluid_density*data.fracture_compressibility/time_step *
								xi_p[j]*xi_p[i]
								+
								perm_f*data.fluid_density/data.fluid_viscosity *
								grad_xi_p[j]*grad_xi_p[i];

							local_matrix(i, j) += (xi_r*m_r + xi_f*m_f)*fe_values.JxW(q);

							// local_matrix(i, jj)
						}  // end j loop

						double rhs_r =
							data.fluid_density/time_step *
							(1./modulus_M + data.biot_coef*data.biot_coef/bulk_modulus) *
							old_p_values[q]*xi_p[i]
							-
							data.biot_coef/time_step*div_delta_u*xi_p[i]
							// +
							// data.perm_res*data.fluid_density/data.fluid_viscosity *
							// data.fluid_density*g_vector*grad_xi_p[i]
							+
							data.biot_coef*data.biot_coef/bulk_modulus *
							(p_values[q] - old_p_values[q])/time_step*xi_p[i]
							+
							source_term*xi_p[i]
							;

						double rhs_f =
							data.fluid_density*data.fracture_compressibility/time_step *
							old_p_values[q]*xi_p[i]
							// +
							// perm_f*data.fluid_density/data.fluid_viscosity *
							// data.fluid_density*g_vector*grad_xi_p[i]
							+
							source_term*xi_p[i]
							// +
							// leakoff term
							;

							local_rhs[i] += (xi_r*rhs_r + xi_f*rhs_f)*fe_values.JxW(q);
					}  // end i loop

				}  // end q-point loop

	      cell->get_dof_indices(local_dof_indices);
	      constraints.distribute_local_to_global(local_matrix,
	                                             local_rhs,
	                                             local_dof_indices,
	                                             system_matrix,
	                                             rhs_vector);
			}  // end cell loop

    system_matrix.compress(VectorOperation::add);
    rhs_vector.compress(VectorOperation::add);

		computing_timer.exit_section();
	}  // eom


	template <int dim>
	const DoFHandler<dim> &
	PressureSolver<dim>::get_dof_handler()
	{
		return dof_handler;
	}  // eom

	template <int dim>
	const FESystem<dim> &
	PressureSolver<dim>::get_fe()
	{
		return fe;
	}

	template <int dim>
	const ConstraintMatrix &
	PressureSolver<dim>::get_constraint_matrix()
	{
		return constraints;
	}

	template <int dim> double
	PressureSolver<dim>::solve()
	{
  	computing_timer.enter_section("Solve pressure system");
		SolverControl solver_control (dof_handler.n_dofs(), 1e-12);
		TrilinosWrappers::SolverCG solver(solver_control);

		TrilinosWrappers::PreconditionAMG preconditioner;
		TrilinosWrappers::PreconditionAMG::AdditionalData data;
    // data.constant_modes = constant_modes;
    data.elliptic = true;
    data.smoother_sweeps = 2;
    data.aggregation_threshold = 0.02;
		preconditioner.initialize(system_matrix.block(0, 0), data);

		solver.solve(system_matrix.block(0, 0), solution.block(0),
								 rhs_vector.block(0), preconditioner);

		constraints.distribute(solution);
		// relevant_solution = solution;

  	computing_timer.exit_section();

		return solver_control.last_step();
	}  // eom

}  // end of namespace
