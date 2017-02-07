#include <deal.II/fe/fe_values.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>        // std::sqrt(double)


namespace constitutive_model {
  using namespace dealii;


  template <int dim>
  inline SymmetricTensor<2,dim> get_strain_tensor(FEValues<dim> &fe_values,
                                                  const unsigned int shape_func,
                                                  const unsigned int q_point) {
    SymmetricTensor<2,dim> tmp;
    tmp = 0;
    for (unsigned int i=0; i<dim; ++i){
      tmp[i][i] += fe_values.shape_grad_component(shape_func, q_point, i)[i];
      for(unsigned int j=0; j<dim; ++j){
        tmp[i][j] =
          (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
           fe_values.shape_grad_component(shape_func, q_point, j)[i])/2;
      }
    }
    return tmp;
  }


  template <int dim>
  inline SymmetricTensor<2,dim>
  get_strain_tensor (const std::vector<Tensor<1,dim> > &grad)
    /*
      Compute local strain tensor from solution gradients
     */
    {
      Assert (grad.size() == dim, ExcInternalError());
      SymmetricTensor<2,dim> strain;
      for (unsigned int i=0; i<dim; ++i)
        strain[i][i] = grad[i][i];
      for (unsigned int i=0; i<dim; ++i)
        for (unsigned int j=i+1; j<dim; ++j)
          strain[i][j] = (grad[i][j] + grad[j][i]) / 2;
      return strain;
    }


  template <int dim> inline
  SymmetricTensor<4, dim> isotropic_gassman_tensor(double lambda, double mu)
  {
	  SymmetricTensor<4, dim> tmp;
	  for (unsigned int i=0; i<dim; ++i)
		  for (unsigned int j=0; j<dim; ++j)
			  for (unsigned int k=0; k<dim; ++k)
				  for (unsigned int l=0; l<dim; ++l)
					  tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
					  	  	  	  	  	 ((i==l) && (j==k) ? mu : 0.0) +
                               ((i==j) && (k==l) ? lambda : 0.0));
	  return tmp;
  }

  template <int dim> void
  get_strain_tensor_plus(const SymmetricTensor<2,dim> &strain_tensor,
                         SymmetricTensor<2,dim>       &strain_tensor_plus)
  {
    // Assert that dimensions match

    Tensor<2,dim> p_matrix, lambda_matrix;
    strain_tensor_plus = 0;
    double trace_eps = trace(strain_tensor);
    double det_eps = determinant(strain_tensor);
    double lambda_1 = trace_eps/2 + sqrt(trace_eps*trace_eps/4 - det_eps);
    double lambda_2 = trace_eps/2 - sqrt(trace_eps*trace_eps/4 - det_eps);
    lambda_matrix[0][0] = std::max(lambda_1, 0.0);
    lambda_matrix[1][1] = std::max(lambda_2, 0.0);
    lambda_matrix[0][1] = 0;
    lambda_matrix[1][0] = 0;
    double tmp;
    double eps_12 = strain_tensor[0][1];
    // we need to make sure the denominator is nonzero, otherwise we get nans
    if (eps_12 == 0) tmp = 0;
    else tmp = (lambda_1 - strain_tensor[0][0])/strain_tensor[0][1];
    p_matrix[0][0] = 1./sqrt(1 + tmp*tmp);
    p_matrix[1][0] = tmp/sqrt(1 + tmp*tmp);
    if (eps_12 == 0) tmp = 0;
    else tmp = (lambda_2 - strain_tensor[0][0])/strain_tensor[0][1];
    p_matrix[0][1] = 1./sqrt(1 + tmp*tmp);
    p_matrix[1][1] = tmp/sqrt(1 + tmp*tmp);
    strain_tensor_plus = p_matrix*lambda_matrix*transpose(p_matrix);

    // std::cout << (tmp*tmp) << std::endl;
    // std::cout << strain_tensor_plus[0][0] << "\t" << strain_tensor_plus[0][1] << std::endl;
    // std::cout << strain_tensor_plus[1][0] << "\t" << strain_tensor_plus[1][1] << std::endl;
    // std::cout << std::endl;
    // lambda_matrix[0][0] = 1;
    // lambda_matrix[1][1] = 1;
    // p_matrix[0][0] = 1;
    // p_matrix[0][1] = 0;
    // p_matrix[1][0] = 0;
    // p_matrix[1][1] = 1;
    // Tensor<2, dim> result;
    // result = p_matrix*lambda_matrix*transpose(p_matrix);
    // std::cout << result[0][0] << "\t" << result[0][1] << std::endl;
    // std::cout << result[1][0] << "\t" << result[1][1] << std::endl;
    // std::cout << std::endl;
  }  // EOM

}  // end of namespace
