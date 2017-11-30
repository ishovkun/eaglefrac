#pragma once
#include <deal.II/base/tensor.h>

using namespace dealii;

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

namespace Tensors
{
  using namespace dealii;

  template <int dim>
  inline Tensor<2, dim>
  get_Identity ()
  {
    Tensor<2, dim> identity;
    identity[0][0] = 1.0;
    identity[0][1] = 0.0;
    identity[1][0] = 0.0;
    identity[1][1] = 1.0;

    return identity;
  }

}


namespace Decomposition
{
  using namespace dealii;

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
  static const Tensor<2, dim> Identity =
    Tensors::get_Identity<dim>();

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

}
