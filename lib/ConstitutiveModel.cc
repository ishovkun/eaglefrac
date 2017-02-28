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
  void compute_eigenvalues(const Tensor<2,dim> &strain_tensor,
                           std::vector<double> &lambda)
  {
    double trace_eps = trace(strain_tensor);
    double det_eps = determinant(strain_tensor);
    double tmp = std::sqrt(trace_eps*trace_eps/4 - det_eps);
    lambda[0] = trace_eps/2 + tmp;
    lambda[1] = trace_eps/2 - tmp;
  }  // eom


  template <int dim> inline
  void assemble_eigenvalue_matrix(const std::vector<double> &lambda,
                                  Tensor<2,dim>             &lambda_matrix)
  {
    lambda_matrix = 0;
    for (int i=0; i<dim; ++i)
      lambda_matrix[i][i] = std::max(lambda[i], 0.0);
  }  // EOM


  template <int dim> inline
  void assemble_eigenvector_matrix(const Tensor<2,dim>       &eps,
                                   const std::vector<double> &lambda,
                                   Tensor<2,dim>             &p_matrix)
  {
    p_matrix = 0; // clear values

    if (numbers::is_finite(1./eps[0][1]))  // Tensor not diagonal
      {
        double tmp, tmp1;
        tmp = (lambda[0] - eps[0][0])/eps[0][1];
        tmp1 = 1./std::sqrt(1. + tmp*tmp);
        p_matrix[0][0] = tmp1;
        p_matrix[1][0] = tmp/tmp1;

        tmp = (lambda[1] - eps[0][0])/eps[0][1];
        tmp1 = 1./std::sqrt(1. + tmp*tmp);
        p_matrix[0][1] = tmp1;
        p_matrix[1][1] = tmp/tmp1;
      }
    else  // Tensor already diagonal
      {
        if(eps[0][0] >= eps[1][1])  // eigenvalues in the right order
          {
            p_matrix[0][0] = 1;
            p_matrix[1][1] = 1;
          }
        else  // lambda matrix is a swapped eps matrix
          {
            p_matrix[0][1] = 1;
            p_matrix[1][0] = 1;
          }
      }
  }  // EOM


  template <int dim> inline
  void compute_eigenvalue_derivative(const Tensor<2,dim> &eps,
                                     const Tensor<2,dim> &eps_u,
                                     std::vector<double> &lambda_du)
  {
    const double trace_eps = trace(eps);
    const double det_eps = determinant(eps);
    const double trace_eps_u = trace(eps_u);

    double denom = 1/(2*std::sqrt(trace_eps*trace_eps/4 - det_eps));
    double tmp;
    if (numbers::is_finite(denom))
        tmp = denom *
        (eps_u[0][1]*eps[1][0] + eps[0][1]*eps_u[1][0] +
         0.5*(eps[0][0] - eps[1][1])*(eps[0][0] - eps_u[1][1]));
    else tmp = 0;

    lambda_du[0] = trace_eps_u/2 + tmp;
    lambda_du[1] = trace_eps_u/2 - tmp;
  }  // eof


  template <int dim> inline void
  assemble_eigenvalue_derivative_matrix(const std::vector<double> &lambda,
                                        const std::vector<double> &lambda_du,
                                        Tensor<2,dim>             &lambda_matrix_du)
  {
    lambda_matrix_du = 0;
    for (int i=0; i<dim; ++i)
      lambda_matrix_du[i][i] = ((lambda[i] > 0) ? lambda_du[i] : 0);
  }  // EOM


  template <int dim> inline
  void
  assemble_eigenvector_matrix_derivative(const Tensor<2,dim>       &eps,
                                         const Tensor<2,dim>       &eps_u,
                                         const std::vector<double> &lambda,
                                         const std::vector<double> &lambda_du,
                                         Tensor<2,dim>             &p_matrix_du)
  {
    p_matrix_du = 0; // clear values
    if (numbers::is_finite(1./eps[0][1]))  // Tensor not diagonal
      // Compute entries
      for (int i=0; i<dim; ++i)
        {
          double tmp0 = (lambda[i] - eps[0][0])/eps[0][1];
          double tmp2 =
            ((lambda_du[i] - eps_u[0][0])*eps[0][1] -
             (lambda[i]    - eps[0][0])  *eps_u[0][1])
            /(eps[0][1]*eps[0][1]);
          double tmp1 = -1./(1.+tmp0*tmp0) / (2.*sqrt(tmp0*tmp0))
            * 2.*tmp0 * tmp2;

          p_matrix_du[0][i] = tmp1;
          p_matrix_du[1][i] = tmp0*tmp1 + tmp2/std::sqrt(1 + tmp0*tmp0);
        }
    else  // Tensor already diagonal
      {
        // if(eps[0][0] >= eps[1][1])  // eigenvalues in the right order
        //   {
        //     p_matrix_du[0][0] = 1;
        //     p_matrix_du[1][1] = 1;
        //   }
        // else  // lambda matrix is a swapped eps matrix
        //   {
        //     p_matrix_du[0][1] = 1;
        //     p_matrix_du[1][0] = 1;
        //   }
      }

    // // Assert no nonzero values
    // for (int i=0; i<dim; ++ i)
    //   for (int j=0; j<dim; ++ j)
    //     if (!numbers::is_finite(p_matrix_du[i][j]))
    //       p_matrix_du[i][j] = 0;
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

    void get_stress_decomposition_derivatives(const Tensor<2,dim> &strain_tensor,
                                              const Tensor<2,dim> &eps_u,
                                              const double        lame_constant,
                                              const double        shear_modulus,
                                              Tensor<2,dim>       &sigma_u_plus_i,
                                              Tensor<2,dim>       &sigma_u_minus_i);

    Tensor<2,dim> p_matrix, p_matrix_du;
    std::vector<double> lambda, lambda_du;
    Tensor<2,dim> lambda_matrix, lambda_matrix_du;
    Tensor<2,dim> eps_plus, eps_u_plus;
  };


  template <int dim>
  EnergySpectralDecomposition<dim>::EnergySpectralDecomposition()
    :
    lambda(dim),
    lambda_du(dim)
  {}

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
  void EnergySpectralDecomposition<dim>::
  get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
                           const double        lame_constant,
                           const double        shear_modulus,
                           Tensor<2,dim>       &stress_tensor_plus,
                           Tensor<2,dim>       &stress_tensor_minus)
  {
    compute_eigenvalues(strain_tensor, lambda);
    assemble_eigenvalue_matrix(lambda, lambda_matrix);
    assemble_eigenvector_matrix(strain_tensor, lambda, p_matrix);
    eps_plus = p_matrix*lambda_matrix*transpose(p_matrix);

    // Finally, get stress tensor decomposition
    double trace_eps = trace(strain_tensor);
    double trace_eps_pos = std::max(trace_eps, 0.0);

    stress_tensor_plus = 2*shear_modulus*eps_plus;
    stress_tensor_minus = 2*shear_modulus*(strain_tensor - eps_plus);
    for (int i=0; i<dim; ++i)
      {
        stress_tensor_plus[i][i] += lame_constant*trace_eps_pos;
        stress_tensor_minus[i][i] += lame_constant*(trace_eps - trace_eps_pos);
      }

    // std::cout << "strain tensor =" << std::endl;
    // print_tensor(strain_tensor);
    // std::cout << "lambdas =" << std::endl;
    // std::cout << lambda[0] << "\t" << lambda[1] << std::endl;
    // std::cout << "Lambda matrix =" << std::endl;
    // print_tensor(lambda_matrix);
    // std::cout << "p_matrix =" << std::endl;
    // print_tensor(p_matrix);
    // std::cout << "stress tensor plus =" << std::endl;
    // print_tensor(stress_tensor_plus);
  }  // EOM


  template <int dim>
  void EnergySpectralDecomposition<dim>::
  get_stress_decomposition_derivatives(const Tensor<2,dim> &eps,
                                       const Tensor<2,dim> &eps_u,
                                       const double        lame_constant,
                                       const double        shear_modulus,
                                       Tensor<2,dim>       &sigma_u_plus,
                                       Tensor<2,dim>       &sigma_u_minus)
  {
    // already found for the current q-point
    // compute_eigenvalues(eps, lambda);
    // assemble_eigenvalue_matrix(lambda, lambda_matrix);

    compute_eigenvalue_derivative(eps, eps_u, lambda_du);
    assemble_eigenvalue_derivative_matrix(lambda, lambda_du, lambda_matrix_du);
    assemble_eigenvector_matrix_derivative(eps, eps_u, lambda, lambda_du,
                                           p_matrix_du);

    // already computed in the q-point
    // eps_plus = p_matrix*lambda_matrix*transpose(p_matrix);
    double trace_eps_plus = trace(eps_plus);

    eps_u_plus =
      p_matrix_du*lambda_matrix*transpose(p_matrix) +
      p_matrix*lambda_matrix_du*transpose(p_matrix) +
      p_matrix*lambda_matrix*transpose(p_matrix_du);

    // std::cout << std::endl;
    // std::cout << "strain tensor =" << std::endl;
    // print_tensor(eps);

    // std::cout << "strain tensor_u =" << std::endl;
    // print_tensor(eps_u);


    // std::cout << "lambda du =" << std::endl;
    // std::cout << lambda_du[0] << "\t" << lambda_du[1] << std::endl << std::endl;

    // std::cout << "strain tensor +" << std::endl;
    // print_tensor(eps_plus);

    // std::cout << "strain tensor u plus =" << std::endl;
    // print_tensor(eps_u_plus);

    double trace_eps_u_plus = trace_eps_plus>0 ? trace(eps_u_plus) : 0;
    double trace_eps_u = trace(eps_u);

    sigma_u_plus = 0;
    sigma_u_minus = 0;
    for (int i=0; i<dim; ++i)
      {
        for (int j=0; j<dim; ++j)
          {
            // eps_u_plus[i][j] = (eps_plus[i][j]>0 ? eps_u_plus[i][j] : 0);
            sigma_u_plus[i][j] += 2*shear_modulus*eps_u_plus[i][j];
            sigma_u_minus[i][j] += 2*shear_modulus*(eps_u[i][j] - eps_u_plus[i][j]);
          }
        sigma_u_plus[i][i] += lame_constant*trace_eps_u_plus;
        sigma_u_minus[i][i] += lame_constant*(trace_eps_u - trace_eps_u_plus);
      }

    // std::cout << "strain tensor +" << std::endl;
    // print_tensor(eps_plus);
    //
    // std::cout << "stress tensor u plus =" << std::endl;
    // print_tensor(sigma_u_plus);
  }  // EOM

}  // end of namespace
