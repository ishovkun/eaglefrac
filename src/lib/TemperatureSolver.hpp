#include <SinglePhaseData.hpp>

namespace FluidSolvers
{
  using namespace dealii;

  template <int dim>
  class TemperatureSolver
  {
  public:
    TemperatureSolver(MPI_Comm                           &mpi_communicator,
               parallel::distributed::Triangulation<dim> &triangulation_,
               InputData::SinglePhaseData<dim>           &data_,
               ConditionalOStream                        &pcout_,
               TimerOutput                               &computing_timer_);
    ~TemperatureSolver();

    void setup_dofs();
    void assemble_system(const double time_step);

		std::vector<IndexSet> owned_partitioning, relevant_partitioning;
		TrilinosWrappers::MPI::BlockVector solution, relevant_solution;
		TrilinosWrappers::MPI::BlockVector old_solution;

  private:
    // these guys are passed at initialization
    MPI_Comm 																	&mpi_communicator;
    parallel::distributed::Triangulation<dim> &triangulation;
    const InputData::SinglePhaseData<dim> 		&data;
    DoFHandler<dim> 													dof_handler;
    // Pointers to couple with phase_field_solver
    // these are set by method set_coupling
    // auxilary objects
    ConditionalOStream 												&pcout;
    TimerOutput 			 												&computing_timer;

		FESystem<dim> fe;
		double init_temperature;
		ConstraintMatrix constraints;

		TrilinosWrappers::MPI::BlockVector  rhs_vector;
		TrilinosWrappers::BlockSparseMatrix system_matrix;
  };

  template <int dim>
  TemperatureSolver<dim>::TemperatureSolver
  (MPI_Comm                                  &mpi_communicator_,
   parallel::distributed::Triangulation<dim> &triangulation_,
   InputData::SinglePhaseData<dim>           &data_,
   ConditionalOStream                        &pcout_,
   TimerOutput                               &computing_timer_)
    :
    mpi_communicator(mpi_communicator_),
    triangulation(triangulation_),
    data(data_),
    dof_handler(triangulation_),
    pcout(pcout_),
    computing_timer(computing_timer_),
    fe(FE_Q<dim>(1), 1),
    init_temperature(0.0)
  {}

  template<int dim>
  TemperatureSolver<dim>::~TemperatureSolver()
  {
  	dof_handler.clear();
  }  // eom


  template <int dim> void
	TemperatureSolver<dim>::setup_dofs()
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
			constraints.clear();
			constraints.reinit(locally_relevant_dofs);
	    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
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
	TemperatureSolver<dim>::assemble_system(const double time_step)
  {

  	const QGauss<dim> quadrature_formula(fe.degree+2);
  	FEValues<dim> fe_values(fe, quadrature_formula,
                          	update_values | update_gradients |
                          	update_JxW_values);
		const FEValuesExtractors::Scalar temperature(0);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
	  const unsigned int n_q_points    = quadrature_formula.size();

	  FullMatrix<double>           local_matrix(dofs_per_cell, dofs_per_cell);
	  Vector<double>               local_rhs(dofs_per_cell);
		std::vector<double>  				 xi(dofs_per_cell);
  	std::vector< Tensor<1,dim> > grad_xi(dofs_per_cell);
  	std::vector<double>  				 temp_values(n_q_points);

  	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		system_matrix = 0;
		rhs_vector = 0;

    const double diffusivity = 1e-6;

	  typename DoFHandler<dim>::active_cell_iterator
		  cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
				local_matrix = 0;
				local_rhs = 0;

				fe_values.reinit(cell);

        fe_values[temperature].get_function_values(solution, temp_values);


				for (unsigned int q=0; q<n_q_points; ++q)
        {
          // compute shape functions
          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              xi[k] = fe_values[temperature].value(k, q);
              grad_xi[k] = fe_values[temperature].gradient(k, q);
              // pcout << "xi_p " << xi_p[k] << std::endl;
            }  // end k loop

	        for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                local_matrix(i, j) +=
                  (xi[j]/time_step*xi[i]
                   + diffusivity*grad_xi[j]*grad_xi[i])*fe_values.JxW(q);

              local_rhs[i] += temp_values[q]/time_step*xi[i]*fe_values.JxW(q);
            }
        }  // end q point loop
      }  // end cell loop
  }  // eom

}  // end of namespace
