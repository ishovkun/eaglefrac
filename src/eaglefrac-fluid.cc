#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/numerics/vector_tools.h>

#include <limits>       // std::numeric_limits

#include <boost/filesystem.hpp>
#include <boost/variant/get.hpp>

// Custom modules
#include <SinglePhaseData.hpp>
#include <PhaseFieldSolver.hpp>
#include <PressureSolver.hpp>
#include <Postprocessing.hpp>
#include <InitialValues.hpp>
#include <Mesher.hpp>
#include <Well.hpp>


namespace EagleFrac
{
  using namespace dealii;

  template <int dim>
  class SinglePhaseModel
  {
  public:
    SinglePhaseModel(const std::string &input_file_name_);
    ~SinglePhaseModel();

    void run();

  private:
    void create_mesh();
    void read_mesh();
    void setup_dofs();
    void impose_displacement_on_solution();
    void output_results(int time_step_number, double time); //const;
    void refine_mesh();
    void execute_postprocessing(const double time);
    void exectute_adaptive_refinement();
    void prepare_output_directories();
		void print_header();
		double 	compute_fss_error(
			const TrilinosWrappers::MPI::BlockVector &solid_solution,
			const TrilinosWrappers::MPI::BlockVector &pressure_solution,
			const TrilinosWrappers::MPI::BlockVector &old_iter_solid_solution,
			const TrilinosWrappers::MPI::BlockVector &old_iter_pressure_solution);
		double 	compute_fss_error(
			const TrilinosWrappers::MPI::BlockVector &pressure_solution,
			const TrilinosWrappers::MPI::BlockVector &old_iter_pressure_solution);


    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    ConditionalOStream pcout;
    TimerOutput computing_timer;

    InputData::SinglePhaseData<dim> data;
    PhaseField::PhaseFieldSolver<dim> phase_field_solver;
		FluidSolvers::PressureSolver<dim> pressure_solver;

    std::string input_file_name, case_name;

		// this object contains time records for output
		// this allows having a real time value in Paraview
		std::vector< std::pair<double,std::string> > times_and_names;

		// FESystem<dim> pressure_fe;
		// TrilinosWrappers::MPI::BlockVector pressure_owned_solution;
		// TrilinosWrappers::MPI::BlockVector pressure_relevant_solution;
		// TrilinosWrappers::MPI::BlockVector fracture_toughness_owned;
		// TrilinosWrappers::MPI::BlockVector fracture_toughness_relevant;
  };


  template <int dim>
  SinglePhaseModel<dim>::SinglePhaseModel(const std::string &input_file_name_)
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
		data(pcout),
    phase_field_solver(mpi_communicator,
                       triangulation, data,
                       pcout, computing_timer),
    pressure_solver(mpi_communicator,
										triangulation, data,
										phase_field_solver.dof_handler, phase_field_solver.fe,
										pcout, computing_timer),
    input_file_name(input_file_name_)
  {}


  template <int dim>
  SinglePhaseModel<dim>::~SinglePhaseModel()
  {}

  template <int dim>
  void SinglePhaseModel<dim>::read_mesh()
  {
    GridIn<dim> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f(data.mesh_file_name);
    // typename GridIn<dim>::Format format = GridIn<dim>::ucd;
    // gridin.read(f, format);
	  gridin.read_msh(f);
  }

  template <int dim>
  void SinglePhaseModel<dim>::impose_displacement_on_solution()
  {
		// pcout << data.displacement_boundary_labels.size() << std::endl;
		// pcout << data.displacement_boundary_values.size() << std::endl;
    int n_displacement_conditions = data.displacement_boundary_labels.size();
    std::vector<double> displacement_values(n_displacement_conditions);
    for (int i=0; i<n_displacement_conditions; ++i)
      displacement_values[i] = data.displacement_boundary_values[i];

    std::vector<double> displacement_point_values(0);
    phase_field_solver.impose_displacement(data.displacement_boundary_labels,
                                           data.displacement_boundary_components,
                                           displacement_values,
                                           data.displacement_points,
                                           data.displacement_point_components,
                                           displacement_point_values,
                                           data.constraint_point_phase_field);
  }  // eom


  template <int dim>
  void SinglePhaseModel<dim>::exectute_adaptive_refinement()
  {
    phase_field_solver.relevant_solution = phase_field_solver.solution;
    std::vector<const TrilinosWrappers::MPI::BlockVector *> tmp(3);
    tmp[0] = &phase_field_solver.relevant_solution;
    tmp[1] = &phase_field_solver.old_solution;
    tmp[2] = &phase_field_solver.old_old_solution;

    std::vector<const TrilinosWrappers::MPI::BlockVector *> tmp_pressure(2);
    tmp_pressure[0] = &pressure_solver.relevant_solution;
    tmp_pressure[1] = &pressure_solver.old_solution;

    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::BlockVector>
        solution_transfer(phase_field_solver.dof_handler);

    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::BlockVector>
        solution_transfer_pressure(pressure_solver.get_dof_handler());

    solution_transfer.prepare_for_coarsening_and_refinement(tmp);
    solution_transfer_pressure.prepare_for_coarsening_and_refinement(tmp_pressure);
    triangulation.execute_coarsening_and_refinement();

		setup_dofs();

    TrilinosWrappers::MPI::BlockVector
      tmp_owned1(phase_field_solver.owned_partitioning, mpi_communicator),
      tmp_owned2(phase_field_solver.owned_partitioning, mpi_communicator);

    TrilinosWrappers::MPI::BlockVector
      tmp_pressure_owned(pressure_solver.owned_partitioning, mpi_communicator);
      // tmp_pressure_owned2(pressure_solver.owned_partitioning, mpi_communicator);

    std::vector<TrilinosWrappers::MPI::BlockVector *> tmp1(3);
    tmp1[0] = &phase_field_solver.solution;
    tmp1[1] = &tmp_owned1;
    tmp1[2] = &tmp_owned2;

    std::vector<TrilinosWrappers::MPI::BlockVector *> tmp1_pressure(2);
    tmp1_pressure[0] = &pressure_solver.solution;
    tmp1_pressure[1] = &tmp_pressure_owned;

    solution_transfer.interpolate(tmp1);
    solution_transfer_pressure.interpolate(tmp1_pressure);

    phase_field_solver.old_solution = tmp_owned1;
    phase_field_solver.old_old_solution = tmp_owned2;

		pressure_solver.old_solution = tmp_pressure_owned;
		pressure_solver.relevant_solution = pressure_solver.solution;

  }  // eom


  template <int dim>
  void SinglePhaseModel<dim>::prepare_output_directories()
  {
    size_t path_index = input_file_name.find_last_of("/");

    size_t extension_index = input_file_name.substr(path_index).rfind(".");
    // if (extension_index != string::npos)
    //   extension = filename.substr(pos+1);

    case_name = input_file_name.substr(path_index+1, extension_index-1);

    boost::filesystem::path output_directory_path("./" + case_name);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      if (!boost::filesystem::is_directory(output_directory_path))
      {
          pcout << "Output folder not found\n"
                << "Creating directory: ";
          if (boost::filesystem::create_directory(output_directory_path))
            std::cout << "Success" << std::endl;
      }
      else
      { // remove everything from this directory
        pcout << "Folder exists: cleaning folder: ";
        boost::filesystem::remove_all(output_directory_path);
        if (boost::filesystem::create_directory(output_directory_path))
           std::cout << "Success" << std::endl;
      }

			// create directory for vtu's
			boost::filesystem::path vtu_path("./" + case_name + "/vtu");
			boost::filesystem::create_directory(vtu_path);
    }  // end mpi==0
  }  // eom


	template <int dim>
  void SinglePhaseModel<dim>::setup_dofs()
	{
		computing_timer.enter_section("Setup full system");

		phase_field_solver.setup_dofs();
		pressure_solver.setup_dofs();

		// auto & pressure_dof_handler = pressure_solver.get_dof_handler();
		// for (unsigned int w=0; w<data.wells.size(); ++w)
		// 	data.wells[w]->locate(pressure_dof_handler, mpi_communicator);

  	computing_timer.exit_section();
	} // eom


  template <int dim>
  void SinglePhaseModel<dim>::print_header()
	{
		pcout << "PDS" << "\t"
		      << "ASet" << "\t"
		      << "error" << "\t\t"
		      << "GMRES" << "\t"
		      << "Search" << "\t"
					<< std::endl;
	}


  template <int dim>
  void SinglePhaseModel<dim>::run()
  {
    data.read_input_file(input_file_name);
    read_mesh();

		// data.update_well_controlls(0.0);
		// pcout << "rate " << data.wells[0]->value(Point<dim>(2,2), 0) << std::endl;
		// return;

		auto & pressure_dof_handler = pressure_solver.get_dof_handler();
		auto & pressure_fe = pressure_solver.get_fe();
		// point phase_field_solver to pressure objects
  	const FEValuesExtractors::Scalar pressure_extractor(0);
		phase_field_solver.set_coupling(pressure_dof_handler,
																		pressure_fe,
																		pressure_extractor);

    prepare_output_directories();

    // compute_runtime_parameters
    double minimum_mesh_size = Mesher::compute_minimum_mesh_size(triangulation,
                                                                 mpi_communicator);
    const int max_refinement_level =
      + data.initial_refinement_level
      + data.n_adaptive_steps;

    minimum_mesh_size /= std::pow(2, max_refinement_level);
    data.compute_mesh_dependent_parameters(minimum_mesh_size);

		for (unsigned int w=0; w<data.wells.size(); ++w)
			data.wells[w]->set_location_radius(minimum_mesh_size);

    pcout << "min mesh size " << minimum_mesh_size << std::endl;

    // local prerefinement
    triangulation.refine_global(data.initial_refinement_level);
		setup_dofs();

		for (int ref_step=0; ref_step<data.n_adaptive_steps; ++ref_step)
		{
			pcout << "Local_prerefinement" << std::endl;
	    Mesher::refine_region(triangulation,
	                          data.local_prerefinement_region,
	                          1);
	    setup_dofs();
		}

    // Initial values
    phase_field_solver.solution.block(0) = 0;
    phase_field_solver.solution.block(1) = 1;
		VectorTools::interpolate(
			phase_field_solver.dof_handler,
			 InitialValues::Defects<dim>(data.defect_coordinates,
																	  // idk why e/2, it just works better
																	 //  data.regularization_parameter_epsilon/2),
																	//  2*minimum_mesh_size),
																	 minimum_mesh_size),
		   phase_field_solver.solution
		 );

		phase_field_solver.old_solution = phase_field_solver.solution;
		pressure_solver.solution = data.init_pressure;
    // phase_field_solver.old_solution.block(1) = phase_field_solver.solution.block(1);

    double time = 0;
    double time_step = data.get_time_step(time);
    double old_time_step = time_step;
    int time_step_number = 0;
		//
    while(time < data.t_max)
    {
      time_step = data.get_time_step(time);
      time += time_step;
      time_step_number++;

      phase_field_solver.update_old_solution();
			pressure_solver.old_solution = pressure_solver.solution;
      phase_field_solver.use_old_time_step_phi = true;

    redo_time_step:
			// std::cout.precision(1.0/std::static_cast<double>(time_step));
			pcout  << std::endl
						<< "_______________________________________________" << std::endl
						<< "===============================================" << std::endl
            << "Time: "
            << std::defaultfloat << time
						<< "\tStep:"
						<< time_step
            << std::endl;

			// return;

      impose_displacement_on_solution();
			data.update_well_controlls(time);

			// pcout << "Flow rate "
			// 			<< data.wells[0]->value(Point<dim>(2,2), 0)
			// 			<< std::endl;

      std::pair<double,double> time_steps = std::make_pair(time_step, old_time_step);

      IndexSet old_active_set(phase_field_solver.active_set);

			TrilinosWrappers::MPI::BlockVector pressure_old_iter =
				pressure_solver.relevant_solution;
			// TrilinosWrappers::MPI::BlockVector solid_tmp =
			// 	phase_field_solver.relevant_solution;

			double fss_error = std::numeric_limits<double>::max();
			unsigned int fss_step = 0;
		  while (fss_step < 30)
			{
				pcout << "-----------------------------------------------" << std::endl;
				pcout << "FSS iteration " << fss_step << std::endl;
				// if (time_step_number > 1 || fss_step > 0)

				print_header();
	      int pds_step = 0;  // solid system iteration number
	      const double newton_tolerance = data.newton_tolerance;
	      while (pds_step < data.max_newton_iter)
	      {
					pcout << pds_step << "\t";

	        double error = std::numeric_limits<double>::max();
	        if (pds_step > 0)
	        {
						// compute residual
				    phase_field_solver.assemble_coupled_system(phase_field_solver.solution,
																											 pressure_solver.relevant_solution,
																											 time_steps,
																											 /*include_pressure = */ true,
																											 /*assemble_matrix = */ false);
	          phase_field_solver.compute_active_set(phase_field_solver.solution);
	          phase_field_solver.all_constraints.set_zero(phase_field_solver.residual);
	          error = phase_field_solver.residual_norm();

						// print active set and error
	          pcout << phase_field_solver.active_set_size() << "\t";
						std::cout.precision(3);
	          pcout << std::scientific << error << "\t";
	          std::cout.unsetf(std::ios_base::scientific);

	          // break condition
	          if (phase_field_solver.active_set_changed(old_active_set) &&
	              error < newton_tolerance)
	          {
	            pcout << "PDS Converged!" << std::endl;
      				// phase_field_solver.truncate_phase_field();
	            break;
	          }

	          old_active_set = phase_field_solver.active_set;
	        }  // end first newton step condition

					std::pair<unsigned int, unsigned int> newton_step_results;
					try
					{
						newton_step_results =
							phase_field_solver.solve_coupled_newton_step
								(pressure_solver.relevant_solution, time_steps);
					}
					catch (SolverControl::NoConvergence e)
					{
					  computing_timer.exit_section();
						pcout << "linear solver didn't converge!"
						      << "Adjusting time step to " << time_step/10
									<< std::endl;
		        time -= time_step;
		        time_step /= 10;
		        time += time_step;
		        phase_field_solver.solution = phase_field_solver.old_solution;
		        phase_field_solver.use_old_time_step_phi = true;
						pressure_solver.solution = pressure_solver.old_solution;
		        goto redo_time_step;
					}
					phase_field_solver.relevant_solution = phase_field_solver.solution;

					pcout << newton_step_results.first << "\t";
					pcout << newton_step_results.second << "\t";

	        pds_step++;

	        pcout << std::endl;
	      }  // End pds iter

	      // cut the time step if no convergence
	      if (pds_step == data.max_newton_iter)
	      {
	        pcout << "Time step didn't converge: reducing to dt = "
	              << time_step/10 << std::endl;
	        if (time_step/10 < data.minimum_time_step)
	        {
	          pcout << "Time step too small: aborting" << std::endl;
	          std::cout.unsetf(std::ios_base::scientific);
	          throw SolverControl::NoConvergence(-1, -1);
	        }

	        time -= time_step;
	        time_step /= 10;
	        time += time_step;
	        phase_field_solver.solution = phase_field_solver.old_solution;
					pressure_solver.solution = pressure_solver.old_solution;
	        phase_field_solver.use_old_time_step_phi = true;
	        goto redo_time_step;
	      }  // end cut time step

	      // do adaptive refinement if needed
	      if (data.n_adaptive_steps > 0)
	        if (Mesher::prepare_phase_field_refinement(phase_field_solver,
	                                                   data.phi_refinement_value,
	                                                   max_refinement_level))
	        {
	          pcout << std::endl
	               << "Adapting mesh"
	               << std::endl
								 << "Redo time step"
	               << std::endl;
	          exectute_adaptive_refinement();
	          goto redo_time_step;
	        } // end adaptive refinement

				// if (time_step_number > 1)
				{ // Solve for pressure
					pcout << "Pressure solver: ";
					phase_field_solver.relevant_solution = phase_field_solver.solution;
					pressure_solver.assemble_system(phase_field_solver.relevant_solution,
																					phase_field_solver.old_solution,
																					time_step);
					const unsigned int n_pressure_iter = pressure_solver.solve();
					pressure_solver.relevant_solution = pressure_solver.solution;
					pcout << n_pressure_iter << std::endl;
				}

				fss_error = pressure_solver.solution_increment_norm
					(pressure_solver.relevant_solution, pressure_old_iter);
				// alternative error that adds error in displacement and phase-field
				// but only computes vector norm (not the FEM L2 norm)
				// fss_error = compute_fss_error(
				// 	phase_field_solver.relevant_solution, pressure_solver.relevant_solution,
				// 	solid_tmp, pressure_old_iter);
				// solid_tmp = phase_field_solver.solution;
				pressure_old_iter = pressure_solver.solution;
	      // output_results(fss_step);

				pcout << "FSS error: " << fss_error << std::endl;

				if (fss_error < 1e-3)  // value from the paper
				{
					pcout << "FSS converged " << std::endl;
					break;
				}

				fss_step++;

	      phase_field_solver.use_old_time_step_phi = false;
		    // //   phase_field_solver.use_old_time_step_phi = true;
			}  // end fss iteration

      // phase_field_solver.truncate_phase_field();
      output_results(time_step_number, time);
			execute_postprocessing(time);

      old_time_step = time_step;

      if (time >= data.t_max) break;
    }  // end time loop
		//
    // pcout << std::fixed;
    // show timer table in default format
    std::cout.unsetf(std::ios_base::scientific);
  }  // EOM


	template <int dim>
  double SinglePhaseModel<dim>::
	compute_fss_error(
		const TrilinosWrappers::MPI::BlockVector &solid_solution,
		const TrilinosWrappers::MPI::BlockVector &pressure_solution,
		const TrilinosWrappers::MPI::BlockVector &old_iter_solid_solution,
		const TrilinosWrappers::MPI::BlockVector &old_iter_pressure_solution)
	{

		auto & solid_df = phase_field_solver.dof_handler;
		auto & pressure_df = pressure_solver.get_dof_handler();

		auto & pressure_hanging_nodes = pressure_solver.get_constraint_matrix();
		auto & solid_hanging_nodes = phase_field_solver.hanging_nodes_constraints;

		auto & pressure_fe = pressure_solver.get_fe();
		auto & solid_fe    = phase_field_solver.fe;
  	const unsigned int solid_dofs_per_cell    = solid_fe.dofs_per_cell;
  	const unsigned int pressure_dofs_per_cell = pressure_fe.dofs_per_cell;

  	std::vector<types::global_dof_index>
			local_dof_indices_solid(solid_dofs_per_cell),
			local_dof_indices_pressure(pressure_dofs_per_cell);

  	std::vector<bool> solid_dof_touched(solid_df.n_dofs(), false);
  	std::vector<bool> pressure_dof_touched(pressure_df.n_dofs(), false);

	  typename DoFHandler<dim>::active_cell_iterator
	    solid_cell = solid_df.begin_active(),
	    pressure_cell = pressure_df.begin_active(),
	    endc = solid_df.end();

		// double error = 0;
		double u_error = 0;
		double phi_error = 0;
		double p_error = 0;

  	for (; solid_cell != endc; ++solid_cell, ++pressure_cell)
    	if (!solid_cell->is_artificial())
			{
      	solid_cell->get_dof_indices(local_dof_indices_solid);
      	pressure_cell->get_dof_indices(local_dof_indices_pressure);

				for (unsigned int i=0; i<solid_dofs_per_cell; ++i)
				{
						const unsigned int index = local_dof_indices_solid[i];
        		const int component = solid_fe.system_to_component_index(i).first;
						// pcout << "Constrained "
						// 			<< solid_hanging_nodes.is_constrained(index)
						// 			<< " touched "
						// 			<< solid_dof_touched[index]
						// 			<< std::endl;
						if(solid_dof_touched[index] == false &&
							 !solid_hanging_nodes.is_constrained(index))
				  	{
							solid_dof_touched[index] = true;
							// error +=
							double e =
								std::pow(solid_solution[index] - old_iter_solid_solution[index], 2);
							if (component == dim)
								phi_error += e;
							else
								u_error += e;
				 		}
				}  // end dof loop

				for (unsigned int i=0; i<pressure_dofs_per_cell; ++i)
				{
					const unsigned int index = 	local_dof_indices_pressure[i];
					if(pressure_dof_touched[index] == false &&
						 !pressure_hanging_nodes.is_constrained(index))
					{
						pressure_dof_touched[index] = true;
						p_error +=
							std::pow(pressure_solution[index] - old_iter_pressure_solution[index], 2);
					}
				}  // end dof loop

			}  // end cell loop

		phi_error = Utilities::MPI::sum(phi_error, mpi_communicator);
		u_error = Utilities::MPI::sum(u_error, mpi_communicator);
		p_error = Utilities::MPI::sum(p_error, mpi_communicator);
		double error = std::max(phi_error, u_error);
		error = std::max(error, p_error);
		error = std::sqrt(error);
		return error;
	}  // eom


	template <int dim>
  double SinglePhaseModel<dim>::
	compute_fss_error(
		const TrilinosWrappers::MPI::BlockVector &pressure_solution,
		const TrilinosWrappers::MPI::BlockVector &old_iter_pressure_solution)
	{

		auto & pressure_df = pressure_solver.get_dof_handler();
		auto & pressure_hanging_nodes = pressure_solver.get_constraint_matrix();
		auto & pressure_fe = pressure_solver.get_fe();
  	const unsigned int pressure_dofs_per_cell = pressure_fe.dofs_per_cell;

  	std::vector<types::global_dof_index>
			local_dof_indices_pressure(pressure_dofs_per_cell);

  	std::vector<bool> pressure_dof_touched(pressure_df.n_dofs(), false);

	  typename DoFHandler<dim>::active_cell_iterator
	    pressure_cell = pressure_df.begin_active(),
	    endc = pressure_df.end();

		double error = 0;
		// double mean_pressure = 0;
		// unsigned int n_avg_points = 0;

  	for (; pressure_cell!=endc; ++pressure_cell)
    	if (!pressure_cell->is_artificial())
			{
      	pressure_cell->get_dof_indices(local_dof_indices_pressure);

				for (unsigned int i=0; i<pressure_dofs_per_cell; ++i)
				{
					const unsigned int index = local_dof_indices_pressure[i];
					if(pressure_dof_touched[index] == false &&
						 !pressure_hanging_nodes.is_constrained(index))
					{
						pressure_dof_touched[index] = true;
						error +=
							std::pow(pressure_solution(index) - old_iter_pressure_solution(index), 2);
					}
				}  // end dof loop
			}  // end cell loop

		error = Utilities::MPI::sum(error, mpi_communicator);
		error = std::sqrt(error);
		return error;
	}  // eom

  template <int dim>
  void SinglePhaseModel<dim>::execute_postprocessing(const double time)
  {
    // just some commands so no compiler warnings
    for (unsigned int i=0; i<data.postprocessing_function_names.size(); i++)
    {
      unsigned int l = data.postprocessing_function_names[i].size();
      if (data.postprocessing_function_names[i].compare(0, l, "well_pressure") == 0)
			{
				auto & pressure_dof_handler = pressure_solver.get_dof_handler();
				// get well points
				const unsigned int n_wells = data.wells.size();
				std::vector< Point<dim> > points(n_wells);
				for (unsigned int w=0; w<n_wells; ++w)
					points[w] = data.wells[w]->true_location;

				Vector<double> pressure_values =
					Postprocessing::get_point_values(
						pressure_dof_handler, pressure_solver.relevant_solution,
						/* comp = */ 0, points, mpi_communicator);

	      // Sum write output
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream ff;
          ff.open("./" + case_name + "/well_pressure.txt",
                  std::ios_base::app);
          ff << time << "\t";
					for (unsigned int w=0; w<n_wells; ++w)
						ff << pressure_values[w] << "\t";
          ff << std::endl;
        }  // end write file
			}  // end well pressure
      if (data.postprocessing_function_names[i].compare(0, l, "boundary_load") == 0)
      {
        int boundary_id =
            boost::get<int>(data.postprocessing_function_args[i][0]);
        Tensor<1,dim> load =
          Postprocessing::compute_boundary_load(phase_field_solver,
                                                data, boundary_id);
        // Sum write output
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream ff;
          ff.open("./" + case_name + "/boundary_load-" +
                  Utilities::int_to_string(boundary_id, 1) +
                  ".txt",
                  std::ios_base::app);
          ff << time << "\t"
             << load[0] << "\t"
             << load[1] << "\t"
             << std::endl;
        }
      }  // end boundary load

    }  // end loop over postprocessing functions
  }  // eom


  template <int dim>
  void SinglePhaseModel<dim>::output_results(int time_step_number, double time)
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

		// Add pressure
		auto & pressure_dof_handler = pressure_solver.get_dof_handler();
		data_out.add_data_vector(pressure_dof_handler,
														 pressure_solver.relevant_solution,
														 "pressure");
    data_out.build_patches();

    int n_time_step_digits = 3,
        n_processor_digits = 3;

    // Write output from local processors
    const std::string filename = ("./" + case_name + "/vtu/solution-" +
                                  Utilities::int_to_string(time_step_number,
                                                           n_time_step_digits) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(),
                                   n_processor_digits));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu(output);

    // Write master pbtu and pvd files
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
                              Utilities::int_to_string (i, n_processor_digits) +
                              ".vtu");
        std::string pvtu_filename =
				  "solution-" +
          Utilities::int_to_string(time_step_number, n_time_step_digits) +
          ".pvtu";
        std::ofstream
          master_output(("./" + case_name + "/vtu/" + pvtu_filename).c_str());
        data_out.write_pvtu_record(master_output, filenames);

				// write pvd file
				const std::string pvd_filename = "solution.pvd";
				times_and_names.push_back
					(std::pair<double,std::string> (time, "./vtu/" + pvtu_filename) );
				std::ofstream pvd_master(("./" + case_name + "/" + pvd_filename).c_str());
				data_out.write_pvd_record(pvd_master, times_and_names);
      }  // end master output
  } // EOM

}  // end of namespace


std::string parse_command_line(int argc, char *const *argv) {
  std::string filename;
  if (argc < 2) {
    std::cout << "specify the file name" << std::endl;
    exit(1);
  }

  std::list<std::string> args;
  for (int i=1; i<argc; ++i)
    args.push_back(argv[i]);

  int arg_number = 1;
  while (args.size()){
    if (arg_number == 1)
      filename = args.front();
    args.pop_front();
    arg_number++;
  } // EO while args

  return filename;
}  // EOM

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string input_file_name = parse_command_line(argc, argv);
    EagleFrac::SinglePhaseModel<2> problem(input_file_name);
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
