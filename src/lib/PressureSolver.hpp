#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
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
                   const DoFHandler<dim>                     &dof_handler_width_,
                   ConditionalOStream                        &pcout_,
                   TimerOutput                               &computing_timer_);
		~PressureSolver();
		void setup_dofs();
		void assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_solid,
												 const TrilinosWrappers::MPI::BlockVector &old_solution_solid,
                         const TrilinosWrappers::MPI::BlockVector &solution_width,
												 const double time_step);
		const DoFHandler<dim> &get_dof_handler();
		const FESystem<dim>   &get_fe();
		const ConstraintMatrix &get_constraint_matrix();
		unsigned int solve();
		double 	solution_increment_norm(
			const TrilinosWrappers::MPI::BlockVector &linearization_point_relevant,
			const TrilinosWrappers::MPI::BlockVector &old_iter_solution_relevant);


	private:
		// these guys are passed at initialization
		MPI_Comm 																	&mpi_communicator;
		parallel::distributed::Triangulation<dim> &triangulation;
		const InputData::SinglePhaseData<dim> 		&data;
		DoFHandler<dim> 													dof_handler;
		// Pointers to couple with phase_field_solver
		// these are set by method set_coupling
		const DoFHandler<dim>            					&dof_handler_solid;
		const DoFHandler<dim>            					&dof_handler_width;
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
	 const DoFHandler<dim>                     &dof_handler_width_,
   ConditionalOStream                        &pcout_,
   TimerOutput                               &computing_timer_)
	:
  mpi_communicator(mpi_communicator_),
  triangulation(triangulation_),
  data(data_),
  dof_handler(triangulation_),
	dof_handler_solid(dof_handler_solid_),
	dof_handler_width(dof_handler_width_),
  pcout(pcout_),
  computing_timer(computing_timer_),
  fe(FE_Q<dim>(1), 1), // one linear pressure component
	init_pressure(0.0)
	{}  // eom


	template <int dim>
	PressureSolver<dim>::~PressureSolver()
	{
  	dof_handler.clear();
	}  // eom


	template <int dim> void
	PressureSolver<dim>::setup_dofs()
	{
		dof_handler.distribute_dofs(fe);

		IndexSet locally_owned_dofs, locally_relevant_dofs;
		locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler,
                                            locally_relevant_dofs);

		owned_partitioning.clear();
		relevant_partitioning.clear();
		owned_partitioning.push_back(locally_owned_dofs);
		relevant_partitioning.push_back(locally_relevant_dofs);

		{ // Constraints
			// only hanging nodes
      ConstraintMatrix hanging_node_constraints;
      hanging_node_constraints.clear();
      hanging_node_constraints.reinit(locally_relevant_dofs);
	    DoFTools::make_hanging_node_constraints(dof_handler,
                                              hanging_node_constraints);
      hanging_node_constraints.close();
      // make constant pressure boundary conditions
			constraints.clear();
			constraints.reinit(locally_relevant_dofs);
      constraints.merge(hanging_node_constraints);
      const FEValuesExtractors::Scalar pressure_mask(0);
      const auto & boundary_ids = triangulation.get_boundary_ids();
      for (unsigned int i=0; i<boundary_ids.size(); i++)
        VectorTools::interpolate_boundary_values
          (dof_handler, boundary_ids[i],
           ConstantFunction<dim>(data.init_pressure, 1),
           constraints, fe.component_mask(pressure_mask));
    	constraints.close();
		}

		{ // system matrix
	    system_matrix.clear();
	    TrilinosWrappers::BlockSparsityPattern sp(owned_partitioning,
	                                              owned_partitioning,
	                                              relevant_partitioning,
	                                              mpi_communicator);
	    DoFTools::make_sparsity_pattern(dof_handler, sp, constraints,
	                                    /*  keep_constrained_dofs = */ false,
	                                    Utilities::MPI::this_mpi_process(mpi_communicator));
	    sp.compress();
	    system_matrix.reinit(sp);
		}
		{ // vectors
			solution.reinit(owned_partitioning, mpi_communicator);
			relevant_solution.reinit(relevant_partitioning, mpi_communicator);
			old_solution.reinit(relevant_partitioning, mpi_communicator);
	    rhs_vector.reinit(owned_partitioning, relevant_partitioning,
	                      mpi_communicator, /* omit-zeros=*/ true);
		}
	}  // eom

	template <int dim> void
	PressureSolver<dim>::
	assemble_system(const TrilinosWrappers::MPI::BlockVector &solution_solid,
									const TrilinosWrappers::MPI::BlockVector &old_solution_solid,
									const TrilinosWrappers::MPI::BlockVector &solution_width,
									const double                       time_step)
	{
    computing_timer.enter_section("Assemble pressure system");

    const auto & fe_solid = dof_handler_solid.get_fe();
    const auto & fe_width = dof_handler_width.get_fe();

  	const QGauss<dim> quadrature_formula(fe.degree+2);
  	FEValues<dim> fe_values(fe, quadrature_formula,
                          	update_values | update_gradients |
                          	update_quadrature_points |
                          	update_JxW_values);
  	FEValues<dim> fe_values_solid(fe_solid, quadrature_formula,
                          				update_values | update_gradients |
																	update_quadrature_points);
  	FEValues<dim> fe_values_width(fe_width, quadrature_formula,
                          				update_values);

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
  	std::vector<double>  				 width_values(n_q_points);
		// std::vector< Tensor<1,dim> > old_u_values(n_q_points);

		// shape functions
		std::vector<double>  				 xi_p(dofs_per_cell);
  	std::vector< Tensor<1,dim> > grad_xi_p(dofs_per_cell);

  	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Just checking how the source term function performs
    // double total_flux = 0;

		system_matrix = 0;
		rhs_vector = 0;

    double test_flow_rate = 0.0;

	  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  cell_solid = dof_handler_solid.begin_active(),
		  cell_width = dof_handler_width.begin_active(),
		  endc = dof_handler.end();

	  for (; cell!=endc; ++cell, ++cell_solid, ++cell_width)
	    if (cell->is_locally_owned())
			{
				local_matrix = 0;
				local_rhs = 0;

				fe_values.reinit(cell);
				fe_values_solid.reinit(cell_solid);
				fe_values_width.reinit(cell_width);

				// extract solution values
	      fe_values_solid[phase_field].get_function_values(solution_solid, phi_values);
	      fe_values_solid[displacement].get_function_values(solution_solid, u_values);
	      fe_values_solid[displacement].get_function_divergences(solution_solid,
                                         								  	 	 div_u_values);
	      fe_values_solid[displacement].get_function_divergences(old_solution_solid,
                                         								  		 div_old_u_values);
	      fe_values[pressure].get_function_values(relevant_solution, p_values);
	      fe_values[pressure].get_function_values(old_solution, old_p_values);
        fe_values_width.get_function_values(solution_width, width_values);


				// compute poroelastic coefficients
	      double E = data.get_young_modulus->value(cell_solid->center(), 0);
				double nu = data.get_poisson_ratio->value(cell_solid->center(), 0);
				double bulk_modulus = E/3.0/(1.0-2.0*nu);

				// reciprocal poroelastic modulus M
				double recM =
					(data.biot_coef - data.porosity)*
					(1.0-data.biot_coef)/bulk_modulus
					+
					data.porosity*data.fluid_compressibility;
				// pcout << "recM " << recM << std::endl;

				/* optimal FSS comvergence coefficient
				ref: Convergence of iterative coupling for coupledflow and geomechanics
				Mikelic, Wheeler, 2012 */
				// this is for 3D
        // double beta = 0.5*data.biot_coef*data.biot_coef/bulk_modulus;
				// this works good for 2D
        double beta = 0.25*data.biot_coef*data.biot_coef/bulk_modulus;

        // double source_term = 0;
        // for (unsigned int k=0; k<data.wells.size(); ++k)
        //   source_term += data.wells[k]->value(cell->center(), 0) /
        //     cell->measure(); // / dofs_per_cell;
        const auto & q_points = fe_values.get_quadrature_points();

				for (unsigned int q=0; q<n_q_points; ++q)
				{
          // Wellbore
          double source_term = 0;
          for (unsigned int k=0; k<data.wells.size(); ++k)
            source_term += data.wells[k]->value(q_points[q], 0);
          //
          test_flow_rate += source_term*fe_values.JxW(q);
					// values that separate fracture, reservoir, and cake zone
					double cx = 0.1;
					double c1 = 0.5 - cx;
					double c2 = 0.5 + cx;
					// Indicator functions
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

					// compute perm
					// this is a simplistic way to compute frac width but is good for now
					// double w = 2*u_values[q].norm();  // absolute value
					double w = std::max(width_values[q], 0.0);  // absolute value
					double perm_f = 1.0/12.0*w*w;   // fracture perm from lubrication theory
					// perm_f = 10*data.perm_res;
					// perm_f = 1e-11;
					// perm_f = std::max(1e3*data.perm_res, perm_f);
					perm_f = std::max(1e-11, perm_f);
					// perm_f = std::max(data.perm_res, perm_f);

					// interpolate pereability
          const double perm_eff = data.perm_res + xi_f*(perm_f - data.perm_res);
					const double K_eff = perm_eff/data.fluid_viscosity;

					// K_eff = std::max(data.perm_res/data.fluid_viscosity, K_eff);

					// if (source_term > 0)
					// 	pcout << "source " << source_term << std::endl;

					// if (phi_values[q] > 0.1 && phi_values[q] < 0.5)
					// {
					// 	pcout << "phi " << phi_values[q] << std::endl;
					// pcout << "xi_f " << xi_f << std::endl;
					// pcout << "xi_r " << xi_r << std::endl;
					// 	pcout << "K_r " << data.perm_res << std::endl;
					// 	pcout << "K_f " << perm_f << std::endl;
					// 	pcout << "K_eff " << K_eff*data.fluid_viscosity << std::endl;
					// }

					// compute shape functions
	        for (unsigned int k=0; k<dofs_per_cell; ++k)
					{
						xi_p[k] = fe_values[pressure].value(k, q);
						grad_xi_p[k] = fe_values[pressure].gradient(k, q);
						// pcout << "xi_p " << xi_p[k] << std::endl;
					}  // end k loop

	        for (unsigned int i=0; i<dofs_per_cell; ++i)
					{
	        	for (unsigned int j=0; j<dofs_per_cell; ++j)
						{
							double m_r =
								(recM + beta) *
								xi_p[j]/time_step*xi_p[i]
								+
								K_eff*grad_xi_p[j]*grad_xi_p[i]
								;

							double m_f =
								data.fluid_compressibility *
								xi_p[j]/time_step*xi_p[i]
								+
								K_eff*grad_xi_p[j]*grad_xi_p[i]
								;

							local_matrix(i, j) += (xi_r*m_r + xi_f*m_f)*fe_values.JxW(q);
						}  // end j loop

						// pcout <<
						// 	(1./modulus_M + data.biot_coef*data.biot_coef/bulk_modulus)
						// 	// * old_p_values[q]/time_step
						// 	<< std::endl;

						double rhs_r =
							(recM + beta) *
							old_p_values[q]/time_step*xi_p[i]
							-
							data.biot_coef *
							(div_u_values[q] - div_old_u_values[q])/time_step*xi_p[i]
							// +
							// K_eff*data.fluid_density*g_vector*grad_xi_p[i]
							+
							beta*(p_values[q] - old_p_values[q])/time_step*xi_p[i]
							+
							source_term*xi_p[i]
							;

						double rhs_f =
							data.fluid_compressibility *
							old_p_values[q]/time_step*xi_p[i]
							// +
							// Keff *
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
    // Just checking
    test_flow_rate = Utilities::MPI::sum(test_flow_rate, mpi_communicator);
    // pcout << "Total flux " << test_flow_rate << std::endl;
		// pcout << "norm " << rhs_vector.l2_norm() << std::endl;

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


	template <int dim> unsigned int
	PressureSolver<dim>::solve()
	{
  	computing_timer.enter_section("Solve pressure system");
  	const unsigned int max_iter = system_matrix.m();
		const double tol = 1e-10 + 1e-10*rhs_vector.l2_norm();
		SolverControl solver_control(max_iter, tol);
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


	template <int dim> double
	PressureSolver<dim>::
	solution_increment_norm(
		const TrilinosWrappers::MPI::BlockVector &linearization_point_relevant,
		const TrilinosWrappers::MPI::BlockVector &old_iter_solution_relevant)
	{
  	const QGauss<dim> quadrature_formula(fe.degree+2);
  	FEValues<dim> fe_values(fe, quadrature_formula,
                          	update_values |
                          	update_quadrature_points |
                          	update_JxW_values);

		const FEValuesExtractors::Scalar pressure(0);

	  const unsigned int dofs_per_cell = fe.dofs_per_cell;
	  const unsigned int n_q_points    = quadrature_formula.size();

  	std::vector<double>  				 p_values(n_q_points);
  	std::vector<double>  				 old_p_values(n_q_points);

  	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		double error = 0;
		double area = 0;

	  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

	  for (; cell!=endc; ++cell)
	    if (cell->is_locally_owned())
			{
				fe_values.reinit(cell);
				double d = cell->diameter();
				area += d*d;

				fe_values[pressure].get_function_values(
					linearization_point_relevant, p_values);
				fe_values[pressure].get_function_values(
					old_iter_solution_relevant, old_p_values);

					for (unsigned int q=0; q<n_q_points; ++q)
					{
						double dp = p_values[q] - old_p_values[q];
						error += dp*dp*fe_values.JxW(q);
					}  // end q_point loop
			} // end cell loop

			error = Utilities::MPI::sum(error, mpi_communicator);
			area = Utilities::MPI::sum(area, mpi_communicator);
			error = std::sqrt(error)/area;
			// pcout << "area " << area << std::endl;
			return error;

	} // eom
}  // end of namespace
