#include <deal.II/base/tensor.h>

#include <PhaseFieldSolver.hpp>
#include <ConstitutiveModel.hpp>

namespace Postprocessing {
  using namespace dealii;

  template <int dim> Tensor<1,dim>
  compute_boundary_load(PhaseField::PhaseFieldSolver<dim> &pf,
                        input_data::PhaseFieldData<dim> &data,
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
              double lame_constant = data.lame_constant;
              double shear_modulus = data.shear_modulus;

              PhaseField::convert_to_tensor(strain_values[q], strain_value);

              stress_value =
                (lame_constant*trace(strain_value)*identity_tensor +
                 2*shear_modulus*strain_value)*fe_face_values.JxW(q);

              local_load += stress_value*fe_face_values.normal_vector(q);
            }  // end q-point loop

          }  // end face loop

    for (int c=0; c<dim; ++c)
      boundary_load[c] = Utilities::MPI::sum(local_load[c], pf.mpi_communicator);

    return boundary_load;
  }  // eof

}  // end of namespace
