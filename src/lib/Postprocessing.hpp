#include <deal.II/base/tensor.h>

#include <PhaseFieldSolver.hpp>
#include <ConstitutiveModel.hpp>

namespace Postprocessing {
  using namespace dealii;

  template <int dim> Tensor<1,dim>
  compute_boundary_load(PhaseField::PhaseFieldSolver<dim> &pf,
                        InputData::PhaseFieldData<dim> &data,
                        const int boundary_id)
  {
    const QGauss<dim-1> face_quadrature_formula(pf.fe.degree+1);
    FEFaceValues<dim> fe_face_values(pf.fe, face_quadrature_formula,
                                     update_gradients | update_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);

    const int dofs_per_cell = pf.fe.dofs_per_cell;
    const int n_q_points = face_quadrature_formula.size();

    std::vector<unsigned int>    local_dof_indices(dofs_per_cell);
    std::vector<  SymmetricTensor<2,dim> > strain_values(n_q_points);
    Tensor<2,dim> strain_value, stress_value;
    Tensor<2,dim> identity_tensor = ConstitutiveModel::get_identity_tensor<dim>();
    Tensor<1,dim> boundary_load, local_load;

    const FEValuesExtractors::Vector displacement(0);

    pf.relevant_solution = pf.solution;

    typename DoFHandler<dim>::active_cell_iterator
      cell = pf.dof_handler.begin_active(),
      endc = pf.dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary()
              && cell->face(f)->boundary_id() == boundary_id)
          {
            fe_face_values.reinit(cell, f);

            fe_face_values[displacement].get_function_symmetric_gradients
              (pf.relevant_solution, strain_values);

            for (int q=0; q<n_q_points; ++q)
            {
              const double lame_constant = data.lame_constant;
              const double shear_modulus = data.shear_modulus;

              PhaseField::convert_to_tensor(strain_values[q], strain_value);

              stress_value =
                (lame_constant*trace(strain_value)*identity_tensor +
                 2*shear_modulus*strain_value);

              local_load += stress_value*fe_face_values.normal_vector(q)*
													  fe_face_values.JxW(q);
            }  // end q-point loop
          }  // end face loop

    for (int c=0; c<dim; ++c)
      boundary_load[c] = Utilities::MPI::sum(local_load[c], pf.mpi_communicator);

    return boundary_load;
  }  // eom


  template <int dim>
	Vector<double>
  compute_cod(PhaseField::PhaseFieldSolver<dim> &pf,
		          std::vector<double>               &lines,
							MPI_Comm                          &mpi_communicator,
							const unsigned int                direction,
						  const double                      space_tol=1e-7)
	{
		AssertThrow(direction < dim, ExcMessage("Direction argument is wrong"));

		const QGauss<dim-1>  face_quadrature_formula(3);
  	const unsigned int n_face_q_points = face_quadrature_formula.size();

	  FEFaceValues<dim> fe_face_values(pf.fe, face_quadrature_formula,
	                          		 		 update_values | update_quadrature_points |
																 	 	 update_JxW_values | update_gradients);

	  const FEValuesExtractors::Vector displacement(0);
	  const FEValuesExtractors::Scalar phase_field(dim);

	  std::vector< Tensor<1,dim> > u_values(n_face_q_points);
	  std::vector< Tensor<1,dim> > grad_phi_values(n_face_q_points);

		const unsigned int n_lines = lines.size();
		Vector<double> cod_values(n_lines);
	  pf.relevant_solution = pf.solution;

    typename DoFHandler<dim>::active_cell_iterator
      cell = pf.dof_handler.begin_active(),
      endc = pf.dof_handler.end();

		for (; cell != endc; ++cell)
	    if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		    {
					fe_face_values.reinit(cell, f);
					fe_face_values[displacement].get_function_values(pf.relevant_solution,
					                                            		 u_values);
	        fe_face_values[phase_field].get_function_gradients(pf.relevant_solution,
					                                              		 grad_phi_values);

	        for (unsigned int q = 0; q < n_face_q_points; ++q)
					{
						auto & q_point = fe_face_values.quadrature_point(q);
						for (unsigned int k=0; k<n_lines; ++k)
						{
							if (
								(q_point[1-direction] > lines[k] - space_tol) &&
								(q_point[1-direction] < lines[k] + space_tol)
							)
							{
								cod_values[k] += 0.5*u_values[q]*grad_phi_values[q]*
																 fe_face_values.JxW(q);
							}
						}  // end line loop
					} // end q_point loop
				}  // end face loop

		for (unsigned int k=0; k<n_lines; ++k)
			cod_values[k] = Utilities::MPI::sum(cod_values[k], mpi_communicator);

		return cod_values;
	}  // eom
}  // end of namespace
