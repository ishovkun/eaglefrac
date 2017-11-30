#pragma once

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/constraint_matrix.h>
// custom modules
#include <SinglePhaseData.hpp>
#include <PhaseFieldSolver.hpp>

/*
  This class computes the fracture using the strategy explained in:
  Iterative coupling of flow, geomechanics and adaptive phase-field fracture
  including level-set crack width approaches
  Lee, Sanghyun, Mary F. Wheeler, and Thomas Wick.
  Iterative coupling of flow, geomechanics and adaptive phase-field fracture
  including level-set crack width approaches.
  Journal of Computational and Applied Mathematics 314 (2017): 40-60.

  For the level set calculation we use the Formulation 4.

  A similar code for material ids for the fracture and reservoir domains +
  crack boundary integral are explained in step-46 of the deal.ii tutorial.
 */

namespace PhaseField
{
  using namespace dealii;


  template <int dim>
  class WidthSolver
  {
    // Methods
  public:
    WidthSolver
    (MPI_Comm                                  &mpi_communicator_,
     parallel::distributed::Triangulation<dim> &triangulation_,
     const InputData::SinglePhaseData<dim>     &data_,
     const DoFHandler<dim>                     &dof_handler_solid_,
     ConditionalOStream                        &pcout_,
     TimerOutput                               &computing_timer_);
    ~WidthSolver();
    void setup_dofs();
    void compute_level_set(const TrilinosWrappers::MPI::BlockVector
                           &relevant_solution_solid);
    void assemble_system(const TrilinosWrappers::MPI::BlockVector
                         &relevant_solution_solid);
    unsigned int solve_system();
    const DoFHandler<dim> & get_dof_handler();

  // private:
  private:
    static bool
    cell_in_fracture (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_in_reservoir (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool cell_in_fracture(const std::vector<double> &phi_values,
                                 const double level_set_value);

    // Fields
  public:
    TrilinosWrappers::BlockSparseMatrix system_matrix;
		TrilinosWrappers::MPI::BlockVector solution, relevant_solution, rhs_vector;
		std::vector<IndexSet> owned_partitioning, relevant_partitioning;
    Vector<double>  material_ids;

  private:
    MPI_Comm                                  &mpi_communicator;
    parallel::distributed::Triangulation<dim> &triangulation;
    DoFHandler<dim>                           dof_handler;
    const DoFHandler<dim>                     &dof_handler_solid;
    const InputData::SinglePhaseData<dim>     &data;
		FE_Q<dim>                                 fe;
		ConditionalOStream 										    &pcout;
		TimerOutput 			 										    &computing_timer;

		ConstraintMatrix                      constraints;
    enum {reservoir_domain_id, fracture_domain_id};
  }; // eod

  template <int dim>
  WidthSolver<dim>::WidthSolver
  (MPI_Comm                                  &mpi_communicator_,
   parallel::distributed::Triangulation<dim> &triangulation_,
   const InputData::SinglePhaseData<dim>     &data_,
	 const DoFHandler<dim>                     &dof_handler_solid_,
   ConditionalOStream                        &pcout_,
   TimerOutput                               &computing_timer_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    dof_handler(triangulation_),
    dof_handler_solid(dof_handler_solid_),
    data(data_),
    // fe(FE_Q<dim>(1), 1), // one linear width component
    fe(1),
    pcout(pcout_),
    computing_timer(computing_timer_)
  {}  // eom


	template <int dim>
	WidthSolver<dim>::~WidthSolver()
	{
  	dof_handler.clear();
	}  // eom


	template <int dim> void
	WidthSolver<dim>::setup_dofs()
	{
  	computing_timer.enter_section("Setup width system");
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
      // make constant pressure boundary conditions
			constraints.clear();
			constraints.reinit(locally_relevant_dofs);
	    DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints);
      // const FEValuesExtractors::Scalar pressure_mask(0);
      const auto & boundary_ids = triangulation.get_boundary_ids();
      for (unsigned int i=0; i<boundary_ids.size(); i++)
        VectorTools::interpolate_boundary_values
          (dof_handler, boundary_ids[i],
           ConstantFunction<dim>(0.0, 1),
           constraints);
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
	    rhs_vector.reinit(owned_partitioning, relevant_partitioning,
	                      mpi_communicator, /* omit-zeros=*/ true);
      material_ids.reinit(triangulation.n_active_cells());
		}
  	computing_timer.exit_section();
	}  // eom


  template <int dim>
  inline bool
  WidthSolver<dim>::
  cell_in_fracture(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fracture_domain_id);
  }  // eom


  template <int dim>
  inline bool
  WidthSolver<dim>::
  cell_in_reservoir(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == reservoir_domain_id);
  }  // eom


	template <int dim> void
	WidthSolver<dim>::
  compute_level_set(const TrilinosWrappers::MPI::BlockVector &relevant_solution_solid)
  {
  	computing_timer.enter_section("Compute level set");

    const auto & fe_solid = dof_handler_solid.get_fe();
    const unsigned int dofs_per_cell = fe_solid.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar phase_field(dim);

    unsigned int idx = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_solid.begin_active(),
      endc = dof_handler_solid.end();

    for (; cell!=endc; ++cell)
      if (!cell->is_artificial())
      {
        cell->get_dof_indices(local_dof_indices);
        bool in_fracture = false;
        // pcout << "cell " << cell_solid->index() << std::endl;

        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const int component = fe_solid.system_to_component_index(i).first;
          const unsigned int index = local_dof_indices[i];
          // pcout << "comp " << component << std::endl;
          if (component == dim)  // component = phi
          {
            double phi_ls =
              relevant_solution_solid[index] - data.constant_level_set;
            // pcout << phi_ls << std::endl;
            if (phi_ls < 0.0)
              in_fracture = true;
          }  // end if component == phi
        }  // end i loop

        if (in_fracture)
        {
          cell->set_material_id(fracture_domain_id);
          material_ids[idx] = 0;
        }
        else
        {
          cell->set_material_id(reservoir_domain_id);
          material_ids[idx] = 1;
        }
        // if (in_fracture)
        //   pcout << "In fracture " << in_fracture << std::endl;
        idx++;
      }  // end cell loop

  	computing_timer.exit_section();
  }  // eom


	template <int dim> void
	WidthSolver<dim>::
  assemble_system(const TrilinosWrappers::MPI::BlockVector &relevant_solution_solid)
  {
  	computing_timer.enter_section("Assemble width system");

    const auto & fe_solid = dof_handler_solid.get_fe();

    const QGauss<dim> quadrature_formula(fe.degree + 2);
    const QGauss<dim-1> face_quadrature_formula(fe.degree + 1);
    const QGauss<dim> quadrature_neighbor(1);

    FEValues<dim>     fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                update_JxW_values);
    FEValues<dim>     fe_values_solid(fe_solid, quadrature_formula,
                                      update_values);
    FEValues<dim>     fe_values_neighbor_solid(fe_solid, quadrature_neighbor,
                                               update_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_JxW_values);
  	FEFaceValues<dim> fe_face_values_solid(fe_solid, face_quadrature_formula,
                                           update_values | update_gradients);

    const FEValuesExtractors::Vector displacement(0);
    const FEValuesExtractors::Scalar phase_field(dim);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = fe_values.n_quadrature_points;
    const unsigned int n_q_face_points = fe_face_values.n_quadrature_points;
    const unsigned int n_q_neighbor_points =
      fe_values_neighbor_solid.n_quadrature_points;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double>            local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>                local_rhs(dofs_per_cell);
		std::vector<double>  				  xi(dofs_per_cell);
  	std::vector< Tensor<1,dim> >  grad_xi(dofs_per_cell);
    // fe values containers
    std::vector<double>           phi_values(n_q_points);
    std::vector< Tensor<1, dim> > u_values(n_q_face_points);
    std::vector< Tensor<1,dim> >  grad_phi_values(n_q_face_points);
    std::vector<double>           phi_values_neighbor(n_q_neighbor_points);

    system_matrix = 0;
    rhs_vector = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      cell_solid = dof_handler_solid.begin_active(),
      endc = dof_handler.end();

    for (; cell!=endc; ++cell, ++cell_solid)
      if (cell->is_locally_owned())
      {
        local_rhs = 0;
        local_matrix = 0;
        fe_values.reinit(cell);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            xi[k] = fe_values.shape_value(k, q);
            grad_xi[k] = fe_values.shape_grad(k ,q);
          }

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              local_matrix(i, j) += grad_xi[j]*grad_xi[i]*fe_values.JxW(q);
        }  // end q_point loop

        fe_values_solid.reinit(cell_solid);
        fe_values_solid[phase_field].get_function_values(relevant_solution_solid,
                                                         phi_values);

        // if (cell_in_fracture(cell_solid))
        if (!cell_in_fracture(phi_values, data.constant_level_set))
        {
          // pcout << "cell in fracture" << std::endl;
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell_solid->at_boundary(f) == false)
            {
              // pcout << "cell not at boundary" << std::endl;
              fe_values_neighbor_solid.reinit(cell_solid->neighbor(f));
              fe_values_neighbor_solid[phase_field].
                get_function_values(relevant_solution_solid,
                                    phi_values_neighbor);
              // pcout << "got neighbor fe values" << std::endl;

              // if (cell_in_reservoir(cell->neighbor(f)))
              if (cell_in_fracture(phi_values_neighbor, data.constant_level_set))
              {
                pcout << "on interface" << std::endl;
                fe_face_values.reinit(cell, f);
                fe_face_values_solid.reinit(cell_solid, f);

                fe_face_values_solid[displacement].
                  get_function_values(relevant_solution_solid, u_values);
                fe_face_values_solid[phase_field].
                  get_function_gradients(relevant_solution_solid,
                                          grad_phi_values);
                // pcout << "got face fe values" << std::endl;

                for (unsigned int q=0; q<n_q_face_points; ++q)
                  {
                    // width at the fracture boundary
                    double w_d =
                      +2.0*scalar_product(u_values[q], grad_phi_values[q]);
                    // -2.0*scalar_product(u_values[q], grad_phi_values[q]);
                    const double grad_phi_norm = grad_phi_values[q].norm();
                    if (grad_phi_norm > 0)
                      w_d /= grad_phi_norm;
                    // if (w_d != 0.0)
                    //   {
                    //     pcout << "cell " << cell->center() << std::endl;
                    //     pcout << "w_d " << w_d << std::endl;
                    //     pcout << "u " << u_values[q] << std::endl;
                    //     pcout << "grad_phi " << grad_phi_values[q] << std::endl;
                    //     pcout << std::endl;
                    //   }

                    const double JxW = fe_face_values.JxW(q);

                    for (unsigned int k=0; k<dofs_per_cell; ++k)
                      xi[k] = fe_values.shape_value(k, q);

                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                      {
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                          local_matrix(i, j) +=
                            data.penalty_theta*xi[j]*xi[i]*JxW;

                        local_rhs(i) +=
                          data.penalty_theta*w_d*xi[i]*JxW;
                      }  // end i loop
                  }
                }  // end if in fracture
            }  // end face loop
        }  // end cell in fracture

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix, local_rhs,
                                               local_dof_indices,
                                               system_matrix, rhs_vector);
      }  // end cell loop

    system_matrix.compress(VectorOperation::add);
    rhs_vector.compress(VectorOperation::add);


  	computing_timer.exit_section();
  }  // eom


  template <int dim> bool
	WidthSolver<dim>::cell_in_fracture(const std::vector<double> &phi_values,
                                     const double level_set_value)
  {
    AssertThrow(phi_values.size() > 0,
                ExcMessage("Size of phi_values should be > 0"));

    double phi_mean = 0;
    for (const auto & phi_value : phi_values)
      phi_mean += phi_value;
    phi_mean /= phi_values.size();
    // std::cout << phi_mean << ", "<<level_set_value << std::endl;

    if (phi_mean < level_set_value)
      return true;
    else
      return false;
  }  // eom


	template <int dim> unsigned int
	WidthSolver<dim>::solve_system()
  {
  	computing_timer.enter_section("Solve width system");

  	const unsigned int max_iter = system_matrix.m();
		double tol = 1e-10*rhs_vector.l2_norm();
    if (tol == 0.0)
      tol = 1e-10;
    pcout << "Width tolerance = " << tol << std::endl;
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


  template <int dim>
  const DoFHandler<dim> &
  WidthSolver<dim>::get_dof_handler()
  {
    return dof_handler;
  }  // eom

}  // end of namespace
