#pragma once

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/config.h> // for numbers::is_nan
#include <cmath>        // std::sqrt(double)


namespace ConstitutiveModel {
  using namespace dealii;


  void print_tensor(const Tensor<2,2> &t)
  {
    for (int i=0; i<2; ++i)
      {
        for (int j=0; j<2; ++j)
          // if (strain_tensor[i][j]!=0)
          std::cout << t[i][j] << "\t";
        std::cout << std::endl;
      }
    std::cout << std::endl;
  }

  template <int dim>
  inline SymmetricTensor<2,dim> get_strain_tensor(FEValues<dim> &fe_values,
                                                  const unsigned int shape_func,
                                                  const unsigned int q_point) {
    SymmetricTensor<2,dim> tmp;
    tmp.clear();
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
  get_strain_tensor(const std::vector<Tensor<1,dim> > &grad)
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

  template <int dim>
  inline Tensor<2,dim>
  get_identity_tensor()
  {
    Tensor<2,dim> identity_tensor;
    identity_tensor.clear();
    for (int i=0; i<dim; ++i)
      identity_tensor[i][i] = 1;
    return identity_tensor;
  }


  template <int dim> inline
  Tensor<4,dim> isotropic_gassman_tensor(double lambda, double mu)
  {
	  Tensor<4, dim> tmp;
    // double zero = static_case<double>(0);
	  for (unsigned int i=0; i<dim; ++i)
		  for (unsigned int j=0; j<dim; ++j)
			  for (unsigned int k=0; k<dim; ++k)
				  for (unsigned int l=0; l<dim; ++l)
					  tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
					  	  	  	  	  	 ((i==l) && (j==k) ? mu : 0.0) +
                               ((i==j) && (k==l) ? lambda : 0.0));
	  return tmp;
  }


  template <int dim> inline
  void compute_eigenvalues(const Tensor<2,dim> &eps,
                           Tensor<1,dim>       &lambda)
  {
    double trace_eps = trace(eps);

    if ( std::abs(eps[0][1]) > 1e-10*trace_eps )
    {
      double det_eps = determinant(eps);
      double tmp = std::sqrt(trace_eps*trace_eps/4 - det_eps);
      lambda[0] = trace_eps/2 + tmp;
      lambda[1] = trace_eps/2 - tmp;
    }
    else
    {
      lambda[0] = eps[0][0];
      lambda[1] = eps[1][1];
    }
  }  // eom


  template <int dim> inline
  void assemble_eigenvalue_matrix(const Tensor<1,dim> &lambda,
                                  Tensor<2,dim>       &lambda_matrix)
  {
    lambda_matrix.clear();
    for (int i=0; i<dim; ++i)
      lambda_matrix[i][i] = std::max(lambda[i], 0.0);
  }  // EOM


  template <int dim> inline
  void assemble_eigenvector_matrix(const Tensor<2,dim>  &eps,
                                   const Tensor<1,dim>  &lambda,
                                   Tensor<2,dim>        &p_matrix)
  {
    p_matrix.clear(); // clear values
    double trace_eps = trace(eps);

    if (numbers::is_finite(1./eps[0][1])
        &&
        std::abs(eps[0][1]) > 1e-10*trace_eps )
    { // if not diagonal or zero
      double tmp, tmp1;
      tmp = (lambda[0] - eps[0][0])/eps[0][1];
      tmp1 = 1./std::sqrt(1. + tmp*tmp);
      p_matrix[0][0] = tmp1;
      p_matrix[1][0] = tmp*tmp1;

      tmp = (lambda[1] - eps[0][0])/eps[0][1];
      tmp1 = 1./std::sqrt(1. + tmp*tmp);
      p_matrix[0][1] = tmp1;
      p_matrix[1][1] = tmp*tmp1;
    }
  else  // Tensor already diagonal
    {
      p_matrix[0][0] = 1;
      p_matrix[1][1] = 1;
    }
  }  // EOM


  template <int dim> inline
  void compute_eigenvalue_derivative(const Tensor<2,dim> &eps,
                                     const Tensor<2,dim> &eps_u,
                                     Tensor<1,dim>       &lambda_du)
  {
    const double trace_eps = trace(eps);
    const double det_eps = determinant(eps);
    const double trace_eps_u = trace(eps_u);

    double denom = 0.5/(std::sqrt(trace_eps*trace_eps/4 - det_eps));

    double tmp = denom *
        (eps_u[0][1]*eps[1][0] + eps[0][1]*eps_u[1][0] +
         0.5*(eps[0][0] - eps[1][1])*(eps_u[0][0] - eps_u[1][1]));

    lambda_du[0] = trace_eps_u/2 + tmp;
    lambda_du[1] = trace_eps_u/2 - tmp;
  }  // eof


  template <int dim> inline
  void
  assemble_eigenvalue_derivative_matrix(const Tensor<1,dim> &lambda,
                                        const Tensor<1,dim> &lambda_du,
                                        Tensor<2,dim>       &lambda_matrix_du)
  {
    lambda_matrix_du.clear();
    for (int i=0; i<dim; ++i)
      lambda_matrix_du[i][i] = (lambda[i] < 0.0) ? 0.0 : lambda_du[i];
  }  // EOM


  template <int dim> inline
  void
  assemble_eigenvector_matrix_derivative(const Tensor<2,dim> &eps,
                                         const Tensor<2,dim> &eps_u,
                                         const Tensor<1,dim> &lambda,
                                         const Tensor<1,dim> &lambda_du,
                                         Tensor<2,dim>       &p_matrix_du)
  {
    p_matrix_du.clear(); // clear values
    // Compute entries
    for (int i=0; i<dim; ++i)
      {
        double tmp0 = (lambda[i] - eps[0][0])/eps[0][1];
        double tmp2 =
          ((lambda_du[i] - eps_u[0][0])*eps  [0][1] -
           (lambda[i]    - eps  [0][0])*eps_u[0][1])
          /(eps[0][1]*eps[0][1]);

        double tmp1 = -1./(1.+tmp0*tmp0) / (std::sqrt(1.+tmp0*tmp0))
          * tmp0 * tmp2;

        p_matrix_du[0][i] = tmp1;
        p_matrix_du[1][i] = tmp0*tmp1 + tmp2/std::sqrt(1. + tmp0*tmp0);

      }
  }  // EOM



  template <int dim>
  class EnergySpectralDecomposition
  {
  public:
    EnergySpectralDecomposition();

    void get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
                                  const double        lame_constant,
                                  const double        shear_modulus,
                                  Tensor<2,dim>       &stress_tensor_plus,
                                  Tensor<2,dim>       &stress_tensor_minus);

    void stress_spectral_decomposition(const Tensor<2,dim> &strain_tensor,
                                       const double        lame_constant,
                                       const double        shear_modulus,
                                       Tensor<2,dim>       &stress_tensor_plus,
                                       Tensor<2,dim>       &stress_tensor_minus);

    void get_stress_decomposition_derivatives(const Tensor<2,dim> &strain_tensor,
                                              const Tensor<2,dim> &eps_u,
                                              const double        lame_constant,
                                              const double        shear_modulus,
                                              Tensor<2,dim>       &sigma_u_plus_i,
                                              Tensor<2,dim>       &sigma_u_minus_i);

    void stress_spectral_decomposition_derivatives(const Tensor<2,dim> &strain_tensor,
                                                   const Tensor<2,dim> &eps_u,
                                                   const double        lame_constant,
                                                   const double        shear_modulus,
                                                   Tensor<2,dim>       &sigma_u_plus_i,
                                                   Tensor<2,dim>       &sigma_u_minus_i);

  private:
    Tensor<2,dim> p_matrix, p_matrix_du;
    Tensor<1,dim> lambda, lambda_du;
    Tensor<2,dim> lambda_matrix, lambda_matrix_du;
    Tensor<2,dim> eps_plus, eps_u_plus;
    Tensor<2,dim> identity_tensor;
  };


  template <int dim>
  EnergySpectralDecomposition<dim>::EnergySpectralDecomposition()
  :
  identity_tensor(get_identity_tensor<dim>())
  {
    // identity_tensor = get_identity_tensor<dim>();
  }


  template <int dim>
  void EnergySpectralDecomposition<dim>::
  get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
                           const double        lame_constant,
                           const double        shear_modulus,
                           Tensor<2,dim>       &stress_tensor_plus,
                           Tensor<2,dim>       &stress_tensor_minus)
 {
  //  Tensor<2,dim> identity_tensor = get_identity_tensor<dim>();
   double d = static_cast<double>(dim);

   double trace_eps = trace(strain_tensor);
   double trace_eps_plus = std::max(trace_eps, 0.);
   double trace_eps_minus = trace_eps - trace_eps_plus;

   stress_tensor_plus =
      (2*shear_modulus/d + lame_constant)*trace_eps_plus*identity_tensor
      +
      2*shear_modulus*(strain_tensor - 1./d*trace_eps*identity_tensor);

   stress_tensor_minus =
      (2*shear_modulus/d + lame_constant)*trace_eps_minus*identity_tensor;
 }  // eom


template <int dim>
void EnergySpectralDecomposition<dim>::
get_stress_decomposition_derivatives(const Tensor<2,dim> &eps,
                                     const Tensor<2,dim> &eps_u,
                                     const double        lame_constant,
                                     const double        shear_modulus,
                                     Tensor<2,dim>       &sigma_u_plus,
                                     Tensor<2,dim>       &sigma_u_minus)
{
  double d = static_cast<double>(dim);
  double trace_eps = trace(eps);
  double trace_eps_u = trace(eps_u);
  double trace_eps_u_plus = (trace_eps>0.0) ? trace_eps_u : 0.0;
  double trace_eps_u_minus = trace_eps_u - trace_eps_u_plus;

  sigma_u_plus =
      (2*shear_modulus/d  + lame_constant)*trace_eps_u_plus*identity_tensor
      +
      2*shear_modulus*(eps_u - 1./d*trace_eps_u*identity_tensor);

  sigma_u_minus =
      (2*shear_modulus/d + lame_constant)*trace_eps_u_minus*identity_tensor;
}  // eom


template <int dim>
void EnergySpectralDecomposition<dim>::
stress_spectral_decomposition(const Tensor<2,dim> &eps,
                              const double        lame_constant,
                              const double        shear_modulus,
                              Tensor<2,dim>       &stress_tensor_plus,
                              Tensor<2,dim>       &stress_tensor_minus)
{
  compute_eigenvalues(eps, lambda);
  assemble_eigenvalue_matrix(lambda, lambda_matrix);
  assemble_eigenvector_matrix(eps, lambda, p_matrix);

  eps_plus = p_matrix*lambda_matrix*transpose(p_matrix);

  // Finally, get stress tensor decomposition
  double trace_eps = trace(eps);
  double trace_eps_plus = std::max(trace_eps, 0.0);

  stress_tensor_plus =
      lame_constant*trace_eps_plus*identity_tensor +
      2*shear_modulus*eps_plus;

  stress_tensor_minus =
      lame_constant*(trace_eps - trace_eps_plus)*identity_tensor +
      2*shear_modulus*(eps - eps_plus);

}  // EOM


template <int dim>
void EnergySpectralDecomposition<dim>::
stress_spectral_decomposition_derivatives(const Tensor<2,dim> &eps,
                                          const Tensor<2,dim> &eps_u,
                                          const double        lame_constant,
                                          const double        shear_modulus,
                                          Tensor<2,dim>       &sigma_u_plus,
                                          Tensor<2,dim>       &sigma_u_minus)
{
  // already found for the current q-point
  double trace_eps = trace(eps);
  compute_eigenvalues(eps, lambda);
  assemble_eigenvalue_matrix(lambda, lambda_matrix);
  assemble_eigenvector_matrix(eps, lambda, p_matrix);

  compute_eigenvalue_derivative(eps, eps_u, lambda_du);
  assemble_eigenvalue_derivative_matrix(lambda, lambda_du, lambda_matrix_du);
  assemble_eigenvector_matrix_derivative(eps, eps_u, lambda, lambda_du,
                                         p_matrix_du);

  eps_u_plus =
    p_matrix_du*lambda_matrix*transpose(p_matrix) +
    p_matrix*lambda_matrix_du*transpose(p_matrix) +
    p_matrix*lambda_matrix*transpose(p_matrix_du);

  double trace_eps_u = trace(eps_u);
  double trace_eps_u_plus = trace_eps > 0 ? trace_eps_u : 0;

  sigma_u_plus =
      lame_constant*trace_eps_u_plus*identity_tensor +
      2*shear_modulus*eps_u_plus;

  sigma_u_minus =
      lame_constant*(trace_eps_u - trace_eps_u_plus)*identity_tensor +
      2*shear_modulus*(eps_u - eps_u_plus);

}  // EOM


}  // end of namespace
