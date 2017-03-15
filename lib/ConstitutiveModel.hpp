#pragma once

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>        // std::sqrt(double)


namespace ConstitutiveModel {
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

  template <int dim>
  inline Tensor<2,dim>
  get_identity_tensor()
  {
    Tensor<2,dim> identity_tensor;
    identity_tensor = 0;
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

    // Tensor<2,dim> p_matrix, p_matrix_du;
    // std::vector<double> lambda, lambda_du;
    // Tensor<2,dim> lambda_matrix, lambda_matrix_du;
    // Tensor<2,dim> eps_plus, eps_u_plus;
    Tensor<2,dim> identity_tensor;
  };


  template <int dim>
  EnergySpectralDecomposition<dim>::EnergySpectralDecomposition()
  {
    identity_tensor = get_identity_tensor<dim>();
  }

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
  //  double trace_eps_plus = std::max(trace_eps, 0.);

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

// template <int dim>
// void EnergySpectralDecomposition<dim>::
// get_stress_decomposition(const Tensor<2,dim> &strain_tensor,
//                          const double        lame_constant,
//                          const double        shear_modulus,
//                          Tensor<2,dim>       &stress_tensor_plus,
//                          Tensor<2,dim>       &stress_tensor_minus)
// {
//   compute_eigenvalues(strain_tensor, lambda);
//   assemble_eigenvalue_matrix(lambda, lambda_matrix);
//   assemble_eigenvector_matrix(strain_tensor, lambda, p_matrix);
//   eps_plus = p_matrix*lambda_matrix*transpose(p_matrix);
//
//   // Finally, get stress tensor decomposition
//   double trace_eps = trace(strain_tensor);
//   double trace_eps_pos = std::max(trace_eps, 0.0);
//
//   stress_tensor_plus = 2*shear_modulus*eps_plus;
//   stress_tensor_minus = 2*shear_modulus*(strain_tensor - eps_plus);
//   for (int i=0; i<dim; ++i)
//     {
//       stress_tensor_plus[i][i] += lame_constant*trace_eps_pos;
//       stress_tensor_minus[i][i] += lame_constant*(trace_eps - trace_eps_pos);
//     }
//
//   // std::cout << "strain tensor =" << std::endl;
//   // print_tensor(strain_tensor);
//   // std::cout << "lambdas =" << std::endl;
//   // std::cout << lambda[0] << "\t" << lambda[1] << std::endl;
//   // std::cout << "Lambda matrix =" << std::endl;
//   // print_tensor(lambda_matrix);
//   // std::cout << "p_matrix =" << std::endl;
//   // print_tensor(p_matrix);
//   // std::cout << "stress tensor plus =" << std::endl;
//   // print_tensor(stress_tensor_plus);
// }  // EOM


// template <int dim>
// void EnergySpectralDecomposition<dim>::
// get_stress_decomposition_derivatives(const Tensor<2,dim> &eps,
//                                      const Tensor<2,dim> &eps_u,
//                                      const double        lame_constant,
//                                      const double        shear_modulus,
//                                      Tensor<2,dim>       &sigma_u_plus,
//                                      Tensor<2,dim>       &sigma_u_minus)
// {
//   // already found for the current q-point
//   // compute_eigenvalues(eps, lambda);
//   // assemble_eigenvalue_matrix(lambda, lambda_matrix);
//
//   compute_eigenvalue_derivative(eps, eps_u, lambda_du);
//   assemble_eigenvalue_derivative_matrix(lambda, lambda_du, lambda_matrix_du);
//   assemble_eigenvector_matrix_derivative(eps, eps_u, lambda, lambda_du,
//                                          p_matrix_du);
//
//   // already computed in the q-point
//   // eps_plus = p_matrix*lambda_matrix*transpose(p_matrix);
//   double trace_eps_plus = trace(eps_plus);
//
//   eps_u_plus =
//     p_matrix_du*lambda_matrix*transpose(p_matrix) +
//     p_matrix*lambda_matrix_du*transpose(p_matrix) +
//     p_matrix*lambda_matrix*transpose(p_matrix_du);
//
//   double trace_eps_u_plus = trace_eps_plus>0 ? trace(eps_u_plus) : 0;
//   double trace_eps_u = trace(eps_u);
//
//   sigma_u_plus = 0;
//   sigma_u_minus = 0;
//   for (int i=0; i<dim; ++i)
//     {
//       for (int j=0; j<dim; ++j)
//         {
//           // eps_u_plus[i][j] = (eps_plus[i][j]>0 ? eps_u_plus[i][j] : 0);
//           sigma_u_plus[i][j] += 2*shear_modulus*eps_u_plus[i][j];
//           sigma_u_minus[i][j] += 2*shear_modulus*(eps_u[i][j] - eps_u_plus[i][j]);
//         }
//       sigma_u_plus[i][i] += lame_constant*trace_eps_u_plus;
//       sigma_u_minus[i][i] += lame_constant*(trace_eps_u - trace_eps_u_plus);
//     }
//
// }  // EOM

template <int dim>
void eigen_vectors_and_values(
  double &E_eigenvalue_1, double &E_eigenvalue_2,
  Tensor<2,dim> &ev_matrix,
  const Tensor<2,dim> &matrix)
{
  // Compute eigenvectors
  Tensor<1,dim> E_eigenvector_1;
  Tensor<1,dim> E_eigenvector_2;
  if (std::abs(matrix[0][1]) < 1e-10*std::abs(matrix[0][0])
      || std::abs(matrix[0][1]) < 1e-10*std::abs(matrix[1][1]))
    {
      // E is close to diagonal
      E_eigenvalue_1 = matrix[0][0];
      E_eigenvector_1[0]=1;
      E_eigenvector_1[1]=0;
      E_eigenvalue_2 = matrix[1][1];
      E_eigenvector_2[0]=0;
      E_eigenvector_2[1]=1;
    }
  else
    {
      double sq = std::sqrt((matrix[0][0] - matrix[1][1]) * (matrix[0][0] - matrix[1][1]) + 4.0*matrix[0][1]*matrix[1][0]);
      E_eigenvalue_1 = 0.5 * ((matrix[0][0] + matrix[1][1]) + sq);
      E_eigenvalue_2 = 0.5 * ((matrix[0][0] + matrix[1][1]) - sq);

      E_eigenvector_1[0] = 1.0/(std::sqrt(1 + (E_eigenvalue_1 - matrix[0][0])/matrix[0][1] * (E_eigenvalue_1 - matrix[0][0])/matrix[0][1]));
      E_eigenvector_1[1] = (E_eigenvalue_1 - matrix[0][0])/(matrix[0][1] * (std::sqrt(1 + (E_eigenvalue_1 - matrix[0][0])/matrix[0][1] * (E_eigenvalue_1 - matrix[0][0])/matrix[0][1])));
      E_eigenvector_2[0] = 1.0/(std::sqrt(1 + (E_eigenvalue_2 - matrix[0][0])/matrix[0][1] * (E_eigenvalue_2 - matrix[0][0])/matrix[0][1]));
      E_eigenvector_2[1] = (E_eigenvalue_2 - matrix[0][0])/(matrix[0][1] * (std::sqrt(1 + (E_eigenvalue_2 - matrix[0][0])/matrix[0][1] * (E_eigenvalue_2 - matrix[0][0])/matrix[0][1])));
    }

  ev_matrix[0][0] = E_eigenvector_1[0];
  ev_matrix[0][1] = E_eigenvector_2[0];
  ev_matrix[1][0] = E_eigenvector_1[1];
  ev_matrix[1][1] = E_eigenvector_2[1];

  // Sanity check if orthogonal
  double scalar_prod = 1.0e+10;
  scalar_prod = E_eigenvector_1[0] * E_eigenvector_2[0] + E_eigenvector_1[1] * E_eigenvector_2[1];

  if (scalar_prod > 1.0e-6)
    {
      std::cout << "Seems not to be orthogonal" << std::endl;
      abort();
    }
}

template <int dim>
void decompose_stress(
  Tensor<2,dim> &stress_term_plus,
  Tensor<2,dim> &stress_term_minus,
  const Tensor<2, dim> &E,
  const double tr_E,
  const Tensor<2, dim> &E_LinU,
  const double tr_E_LinU,
  const double lame_coefficient_lambda,
  const double lame_coefficient_mu,
  const bool derivative)
{
  static const Tensor<2, dim> Identity = get_identity_tensor<dim>();

  // static const Tensor<2, dim> Identity =
  //   Tensors::get_Identity<dim>();

  Tensor<2, dim> zero_matrix;
  zero_matrix.clear();


  // Compute first the eigenvalues for u (as in the previous function)
  // and then for \delta u

  // Compute eigenvalues/vectors
  double E_eigenvalue_1, E_eigenvalue_2;
  Tensor<2,dim> P_matrix;
  eigen_vectors_and_values(E_eigenvalue_1, E_eigenvalue_2,P_matrix,E);

  double E_eigenvalue_1_plus = std::max(0.0, E_eigenvalue_1);
  double E_eigenvalue_2_plus = std::max(0.0, E_eigenvalue_2);

  Tensor<2,dim> Lambda_plus;
  Lambda_plus[0][0] = E_eigenvalue_1_plus;
  Lambda_plus[0][1] = 0.0;
  Lambda_plus[1][0] = 0.0;
  Lambda_plus[1][1] = E_eigenvalue_2_plus;

  if (!derivative)
    {
      Tensor<2,dim> E_plus = P_matrix * Lambda_plus * transpose(P_matrix);

      double tr_E_positive = std::max(0.0, tr_E);

      stress_term_plus = lame_coefficient_lambda * tr_E_positive * Identity
                         + 2 * lame_coefficient_mu * E_plus;

      stress_term_minus = lame_coefficient_lambda * (tr_E - tr_E_positive) * Identity
                          + 2 * lame_coefficient_mu * (E - E_plus);
    }
  else
    {
      // Derviatives (\delta u)

      // Compute eigenvalues/vectors
      double E_eigenvalue_1_LinU, E_eigenvalue_2_LinU;
      Tensor<1,dim> E_eigenvector_1_LinU;
      Tensor<1,dim> E_eigenvector_2_LinU;
      Tensor<2,dim> P_matrix_LinU;

      // Compute linearized Eigenvalues
      double diskriminante = std::sqrt(E[0][1] * E[1][0] + (E[0][0] - E[1][1]) * (E[0][0] - E[1][1])/4.0);

      E_eigenvalue_1_LinU = 0.5 * tr_E_LinU + 1.0/(2.0 * diskriminante) *
                            (E_LinU[0][1] * E[1][0] + E[0][1] * E_LinU[1][0] + (E[0][0] - E[1][1])*(E_LinU[0][0] - E_LinU[1][1])/2.0);

      E_eigenvalue_2_LinU = 0.5 * tr_E_LinU - 1.0/(2.0 * diskriminante) *
                            (E_LinU[0][1] * E[1][0] + E[0][1] * E_LinU[1][0] + (E[0][0] - E[1][1])*(E_LinU[0][0] - E_LinU[1][1])/2.0);


      // Compute normalized Eigenvectors and P
      double normalization_1 = 1.0/(std::sqrt(1 + (E_eigenvalue_1 - E[0][0])/E[0][1] * (E_eigenvalue_1 - E[0][0])/E[0][1]));
      double normalization_2 = 1.0/(std::sqrt(1 + (E_eigenvalue_2 - E[0][0])/E[0][1] * (E_eigenvalue_2 - E[0][0])/E[0][1]));

      double normalization_1_LinU = 0.0;
      double normalization_2_LinU = 0.0;

      normalization_1_LinU = -1.0 * (1.0/(1.0 + (E_eigenvalue_1 - E[0][0])/E[0][1] * (E_eigenvalue_1 - E[0][0])/E[0][1])
                                     * 1.0/(2.0 * std::sqrt(1.0 + (E_eigenvalue_1 - E[0][0])/E[0][1] * (E_eigenvalue_1 - E[0][0])/E[0][1]))
                                     * (2.0 * (E_eigenvalue_1 - E[0][0])/E[0][1])
                                     * ((E_eigenvalue_1_LinU - E_LinU[0][0]) * E[0][1] - (E_eigenvalue_1 - E[0][0]) * E_LinU[0][1])/(E[0][1] * E[0][1]));

      normalization_2_LinU = -1.0 * (1.0/(1.0 + (E_eigenvalue_2 - E[0][0])/E[0][1] * (E_eigenvalue_2 - E[0][0])/E[0][1])
                                     * 1.0/(2.0 * std::sqrt(1.0 + (E_eigenvalue_2 - E[0][0])/E[0][1] * (E_eigenvalue_2 - E[0][0])/E[0][1]))
                                     * (2.0 * (E_eigenvalue_2 - E[0][0])/E[0][1])
                                     * ((E_eigenvalue_2_LinU - E_LinU[0][0]) * E[0][1] - (E_eigenvalue_2 - E[0][0]) * E_LinU[0][1])/(E[0][1] * E[0][1]));


      E_eigenvector_1_LinU[0] = normalization_1 * 1.0;
      E_eigenvector_1_LinU[1] = normalization_1 * (E_eigenvalue_1 - E[0][0])/E[0][1];

      E_eigenvector_2_LinU[0] = normalization_2 * 1.0;
      E_eigenvector_2_LinU[1] = normalization_2 * (E_eigenvalue_2 - E[0][0])/E[0][1];


      // Apply product rule to normalization and vector entries
      double EV_1_part_1_comp_1 = 0.0;  // LinU in vector entries, normalization U
      double EV_1_part_1_comp_2 = 0.0;  // LinU in vector entries, normalization U
      double EV_1_part_2_comp_1 = 0.0;  // vector entries U, normalization LinU
      double EV_1_part_2_comp_2 = 0.0;  // vector entries U, normalization LinU

      double EV_2_part_1_comp_1 = 0.0;  // LinU in vector entries, normalization U
      double EV_2_part_1_comp_2 = 0.0;  // LinU in vector entries, normalization U
      double EV_2_part_2_comp_1 = 0.0;  // vector entries U, normalization LinU
      double EV_2_part_2_comp_2 = 0.0;  // vector entries U, normalization LinU

      // Effizienter spaeter, aber erst einmal uebersichtlich und verstehen!
      EV_1_part_1_comp_1 = normalization_1 * 0.0;
      EV_1_part_1_comp_2 = normalization_1 *
                           ((E_eigenvalue_1_LinU - E_LinU[0][0]) * E[0][1] - (E_eigenvalue_1 - E[0][0]) * E_LinU[0][1])/(E[0][1] * E[0][1]);

      EV_1_part_2_comp_1 = normalization_1_LinU * 1.0;
      EV_1_part_2_comp_2 = normalization_1_LinU * (E_eigenvalue_1 - E[0][0])/E[0][1];


      EV_2_part_1_comp_1 = normalization_2 * 0.0;
      EV_2_part_1_comp_2 = normalization_2 *
                           ((E_eigenvalue_2_LinU - E_LinU[0][0]) * E[0][1] - (E_eigenvalue_2 - E[0][0]) * E_LinU[0][1])/(E[0][1] * E[0][1]);

      EV_2_part_2_comp_1 = normalization_2_LinU * 1.0;
      EV_2_part_2_comp_2 = normalization_2_LinU * (E_eigenvalue_2 - E[0][0])/E[0][1];



      // Build eigenvectors
      E_eigenvector_1_LinU[0] = EV_1_part_1_comp_1 + EV_1_part_2_comp_1;
      E_eigenvector_1_LinU[1] = EV_1_part_1_comp_2 + EV_1_part_2_comp_2;

      E_eigenvector_2_LinU[0] = EV_2_part_1_comp_1 + EV_2_part_2_comp_1;
      E_eigenvector_2_LinU[1] = EV_2_part_1_comp_2 + EV_2_part_2_comp_2;



      // P-Matrix
      P_matrix_LinU[0][0] = E_eigenvector_1_LinU[0];
      P_matrix_LinU[0][1] = E_eigenvector_2_LinU[0];
      P_matrix_LinU[1][0] = E_eigenvector_1_LinU[1];
      P_matrix_LinU[1][1] = E_eigenvector_2_LinU[1];


      double E_eigenvalue_1_plus_LinU = 0.0;
      double E_eigenvalue_2_plus_LinU = 0.0;


      // Very important: Set E_eigenvalue_1_plus_LinU to zero when
      // the corresponding rhs-value is set to zero and NOT when
      // the value itself is negative!!!
      if (E_eigenvalue_1 < 0.0)
        {
          E_eigenvalue_1_plus_LinU = 0.0;
        }
      else
        E_eigenvalue_1_plus_LinU = E_eigenvalue_1_LinU;


      if (E_eigenvalue_2 < 0.0)
        {
          E_eigenvalue_2_plus_LinU = 0.0;
        }
      else
        E_eigenvalue_2_plus_LinU = E_eigenvalue_2_LinU;



      Tensor<2,dim> Lambda_plus_LinU;
      Lambda_plus_LinU[0][0] = E_eigenvalue_1_plus_LinU;
      Lambda_plus_LinU[0][1] = 0.0;
      Lambda_plus_LinU[1][0] = 0.0;
      Lambda_plus_LinU[1][1] = E_eigenvalue_2_plus_LinU;

      Tensor<2,dim> E_plus_LinU = P_matrix_LinU * Lambda_plus * transpose(P_matrix) +  P_matrix * Lambda_plus_LinU * transpose(P_matrix) + P_matrix * Lambda_plus * transpose(P_matrix_LinU);


      double tr_E_positive_LinU = 0.0;
      if (tr_E < 0.0)
        {
          tr_E_positive_LinU = 0.0;

        }
      else
        tr_E_positive_LinU = tr_E_LinU;



      stress_term_plus = lame_coefficient_lambda * tr_E_positive_LinU * Identity
                         + 2 * lame_coefficient_mu * E_plus_LinU;

      stress_term_minus = lame_coefficient_lambda * (tr_E_LinU - tr_E_positive_LinU) * Identity
                          + 2 * lame_coefficient_mu * (E_LinU - E_plus_LinU);


      // Sanity check
      //Tensor<2,dim> stress_term = lame_coefficient_lambda * tr_E_LinU * Identity
      //  + 2 * lame_coefficient_mu * E_LinU;

      //std::cout << stress_term.norm() << "   " << stress_term_plus.norm() << "   " << stress_term_minus.norm() << std::endl;
    }


  }

}  // end of namespace
