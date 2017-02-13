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

  // template <int dim> void
  // get_strain_tensor_plus(const SymmetricTensor<2,dim> &strain_tensor,
  //                        SymmetricTensor<2,dim>       &strain_tensor_plus)
  // {
  //   // Assert that dimensions match

  //   Tensor<2,dim> p_matrix, lambda_matrix;
  //   strain_tensor_plus = 0;
  //   double trace_eps = trace(strain_tensor);
  //   double det_eps = determinant(strain_tensor);
  //   double lambda_1 = trace_eps/2 + sqrt(trace_eps*trace_eps/4 - det_eps);
  //   double lambda_2 = trace_eps/2 - sqrt(trace_eps*trace_eps/4 - det_eps);
  //   lambda_matrix[0][0] = std::max(lambda_1, 0.0);
  //   lambda_matrix[1][1] = std::max(lambda_2, 0.0);
  //   lambda_matrix[0][1] = 0;
  //   lambda_matrix[1][0] = 0;
  //   double tmp;
  //   double eps_12 = strain_tensor[0][1];
  //   // we need to make sure the denominator is nonzero, otherwise we get nans
  //   if (eps_12 == 0) tmp = 0;
  //   else tmp = (lambda_1 - strain_tensor[0][0])/strain_tensor[0][1];
  //   p_matrix[0][0] = 1./sqrt(1 + tmp*tmp);
  //   p_matrix[1][0] = tmp/sqrt(1 + tmp*tmp);
  //   if (eps_12 == 0) tmp = 0;
  //   else tmp = (lambda_2 - strain_tensor[0][0])/strain_tensor[0][1];
  //   p_matrix[0][1] = 1./sqrt(1 + tmp*tmp);
  //   p_matrix[1][1] = tmp/sqrt(1 + tmp*tmp);
  //   strain_tensor_plus = p_matrix*lambda_matrix*transpose(p_matrix);

  //   // std::cout << strain_tensor_plus[0][0] << "\t" << strain_tensor_plus[0][1] << std::endl;
  //   // std::cout << strain_tensor_plus[1][0] << "\t" << strain_tensor_plus[1][1] << std::endl;
  // }  // EOM

  template <int dim> inline
  void assemble_eigenvalue_matrix(const Tensor<2,dim> &strain_tensor,
                                  Tensor<2,dim> &lambda_matrix)
  {
    double trace_eps = trace(strain_tensor);
    double det_eps = determinant(strain_tensor);
    double lambda_1 = trace_eps/2 + sqrt(trace_eps*trace_eps/4 - det_eps);
    double lambda_2 = trace_eps/2 - sqrt(trace_eps*trace_eps/4 - det_eps);
    lambda_matrix[0][0] = std::max(lambda_1, 0.0);
    lambda_matrix[1][1] = std::max(lambda_2, 0.0);
    lambda_matrix[0][1] = 0;
    lambda_matrix[1][0] = 0;
  }  // EOM


  template <int dim> inline
  void assemble_eigenvalue_matrix_derivative(const Tensor<2,dim> &eps,
                                             const Tensor<2,dim> &eps_u_i,
                                             const Tensor<2,dim> &lambda_matrix,
                                             Tensor<2,dim>       &lambda_matrix_du)
  {
    const double trace_eps = trace(eps);
    const double det_eps = determinant(eps);
    const double trace_eps_u_i = trace(eps_u_i);
    const double lambda_1 = lambda_matrix[0][0],
                 lambda_2 = lambda_matrix[1][1];

    // compute lambda_1_du and lambda_2_du
    double lambda_1_du = trace_eps_u_i/2 +
      0.5/sqrt(trace_eps*trace_eps/4 - det_eps) *
      (eps_u_i[0][1]*eps[1][0] + eps[0][1]*eps_u_i[1][0] +
       0.5*(eps[0][0] - eps[1][1])*(eps[0][0] - eps_u_i[1][1]));
    double lambda_2_du = trace_eps_u_i/2 -
      0.5/sqrt(trace_eps*trace_eps/4 - det_eps) *
      (eps_u_i[0][1]*eps[1][0] + eps[0][1]*eps_u_i[1][0] +
       0.5*(eps[0][0] - eps[1][1])*(eps_u_i[0][0] - eps_u_i[1][1]));

    if (lambda_1 > 0)
      lambda_matrix_du[0][0] = lambda_1_du;
    else
      lambda_matrix_du[0][0] = 0;
    if (lambda_2 > 0)
      lambda_matrix_du[1][1] = lambda_2_du;
    else
      lambda_matrix_du[1][1] = 0;

    lambda_matrix_du[0][1] = 0;
    lambda_matrix_du[1][0] = 0;


  }  // EOM


  template <int dim> inline
  void assemble_eigenvector_matrix_derivative(const Tensor<2,dim> &eps,
                                              const Tensor<2,dim> &eps_u_i,
                                              const Tensor<2,dim> &lambda_matrix,
                                              const Tensor<2,dim> &lambda_matrix_du,
                                              Tensor<2,dim>       &p_matrix_du)
  {
    const double eps_12 = eps[0][1];
    const double lambda_1 = lambda_matrix[0][0],
                 lambda_2 = lambda_matrix[1][1];
    const double lambda_1_du = lambda_matrix_du[0][0],
                 lambda_2_du = lambda_matrix_du[1][1];

    double
      tmp10 = 0, tmp11 = 0, tmp12 = 0,
      tmp20 = 0, tmp21 = 0, tmp22 = 0;
    if (abs(eps_12) > 1e-16)
      {
        // For the first eigenvector
        tmp10 = (lambda_1 - eps[0][0])/eps_12;
        tmp12 =
          ((lambda_1_du - eps_u_i[0][0])*eps_12 -
           (lambda_1 - eps[0][0])*eps_12)/(eps_12*eps_12);
        tmp11 = -1/(1+tmp10*tmp10) * 1/(2*sqrt(tmp10*tmp10)) * 2*tmp10 * tmp12;

        // For the second eigenvector
        tmp20 = (lambda_2 - eps[0][0])/eps[0][1];
        tmp22 =
          ((lambda_2_du - eps_u_i[0][0])*eps_12 -
           (lambda_2 - eps[0][0])*eps_12)/(eps_12*eps_12);
        tmp22 = -1/(1+tmp20*tmp20) * 1/(2*sqrt(tmp20*tmp20)) * 2*tmp20 * tmp22;
      }

    // Compute entries
    p_matrix_du[0][0] = tmp11;
    p_matrix_du[1][0] = tmp10*tmp11 + tmp12;
    p_matrix_du[0][1] = tmp21;
    p_matrix_du[1][1] = tmp20*tmp21 + tmp22;

  }  // EOM


  template <int dim> inline
  void assemble_eigenvector_matrix(const Tensor<2,dim> &strain_tensor,
                                   const Tensor<2,dim> &lambda_matrix,
                                   Tensor<2,dim>       &p_matrix)
  {
    double lambda_1 = lambda_matrix[0][0],
           lambda_2 = lambda_matrix[1][1];
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

  }


  template <int dim>
  class EnergySpectralDecomposition
  {
  public:
    void get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
                                  const double        mu,
                                  const double        lambda,
                                  Tensor<2,dim>       &stress_tensor_plus,
                                  Tensor<2,dim>       &stress_tensor_minus);

    void get_stress_decomposition_derivatives(const Tensor<2,dim> &strain_tensor,
                                              const Tensor<2,dim> &eps_u_i,
                                              const double        mu,
                                              const double        lambda,
                                              Tensor<2,dim>       &sigma_u_plus_i,
                                              Tensor<2,dim>       &sigma_u_minus_i);
  private:
    // void get_strain_tensor_plus(const SymmetricTensor<2,dim> &strain_tensor,
    //                             SymmetricTensor<2,dim>       &strain_tensor_plus);

    Tensor<2,dim> p_matrix, p_matrix_du;
    Tensor<2,dim> lambda_matrix, lambda_matrix_du;
    Tensor<2,dim> strain_tensor_plus, eps_u_plus_i;
    // Tensor<1,dim> eigenvector;
  };


  template <int dim>
  void EnergySpectralDecomposition<dim>::
  get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
                           const double        mu,
                           const double        lambda,
                           Tensor<2,dim>       &stress_tensor_plus,
                           Tensor<2,dim>       &stress_tensor_minus)
  {
    // First get strain tensor decomposition
    assemble_eigenvalue_matrix(strain_tensor, lambda_matrix);
    assemble_eigenvector_matrix(strain_tensor, lambda_matrix,
                                p_matrix);
    strain_tensor_plus = p_matrix*lambda_matrix*transpose(p_matrix);;
    // Finally, get stress tensor decomposition
    double trace_eps = trace(strain_tensor);
    double trace_eps_pos = std::max(trace_eps, 0.0);
    stress_tensor_plus = 2*mu*strain_tensor_plus;
    stress_tensor_plus[0][0] += lambda*trace_eps_pos;
    stress_tensor_plus[1][1] += lambda*trace_eps_pos;

    stress_tensor_minus = 2*mu*(strain_tensor - strain_tensor_plus);
    stress_tensor_minus[0][0] += lambda*(trace_eps - trace_eps_pos);
    stress_tensor_minus[1][1] += lambda*(trace_eps - trace_eps_pos);

  }  // EOM


  template <int dim>
  void EnergySpectralDecomposition<dim>::
  get_stress_decomposition_derivatives(const Tensor<2,dim> &strain_tensor,
                                       const Tensor<2,dim> &eps_u_i,
                                       const double        mu,
                                       const double        lambda,
                                       Tensor<2,dim>       &sigma_u_plus_i,
                                       Tensor<2,dim>       &sigma_u_minus_i)
  {
    // Already done for q points
    // assemble_eigenvalue_matrix(strain_tensor, lambda_matrix);
    // assemble_eigenvector_matrix(strain_tensor, lambda_matrix, p_matrix);

    assemble_eigenvalue_matrix_derivative(strain_tensor, eps_u_i,
                                          lambda_matrix, lambda_matrix_du);
    assemble_eigenvector_matrix_derivative(strain_tensor, eps_u_i,
                                           lambda_matrix, lambda_matrix_du,
                                           p_matrix_du);
    // if (std::isnan(p_matrix_du[0][0]))
    //   std::cout << "Nan: " << std::endl;

    eps_u_plus_i =
      (p_matrix_du*(lambda_matrix*transpose(p_matrix))) +
      (p_matrix*(lambda_matrix_du*transpose(p_matrix))) +
      (p_matrix*(lambda_matrix*transpose(p_matrix_du)));
    double trace_eps_u_i = std::max(0.0, trace(eps_u_i));
    double trace_eps = trace(strain_tensor);

    sigma_u_plus_i = 2*mu*eps_u_plus_i;
    sigma_u_plus_i[0][0] += lambda*trace_eps_u_i;
    sigma_u_plus_i[1][1] += lambda*trace_eps_u_i;

    sigma_u_minus_i = 2*mu*(eps_u_i - eps_u_plus_i);
    sigma_u_minus_i[0][0] += lambda*(trace_eps - trace_eps_u_i);
    sigma_u_minus_i[1][1] += lambda*(trace_eps - trace_eps_u_i);

  }  // EOM

}  // end of namespace
