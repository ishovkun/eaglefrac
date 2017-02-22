#include <deal.II/lac/solver_cg.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/trilinos_block_vector.h>


namespace LinearSolvers
{
  using namespace dealii;

  // ------------------------- InverseMatrix --------------------------------
  template <class Matrix, class Preconditioner>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const Matrix         &m,
                  const Preconditioner &preconditioner);

    template <typename VectorType>
    void vmult(VectorType       &dst,
               const VectorType &src) const;

  private:
    const SmartPointer<const Matrix> matrix;
    const Preconditioner &preconditioner;
  };


  template <class Matrix, class Preconditioner>
  InverseMatrix<Matrix,Preconditioner>::
  InverseMatrix(const Matrix &m,
                const Preconditioner &preconditioner)
    :
    matrix (&m),
    preconditioner (preconditioner)
  {}


  template <class Matrix, class Preconditioner>
  template <typename VectorType>
  void
  InverseMatrix<Matrix,Preconditioner>::
  vmult (VectorType       &dst,
         const VectorType &src) const
  {
    SolverControl solver_control(src.size(), 1e-8*src.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);
    dst = 0;
    try
      {
        cg.solve(*matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
  }  // EOM


  // ----------------------- BlockDiagonalPreconditioner ---------------------
  template <class PreconditionerA, class PreconditionerS>
  class BlockDiagonalPreconditioner : public Subscriptor
  {
  public:
    BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                const PreconditionerS &preconditioner_S);

    void vmult(TrilinosWrappers::MPI::BlockVector       &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const;
  private:
    const PreconditionerA &preconditioner_A;
    const PreconditionerS &preconditioner_S;
  };


  template <class PreconditionerA, class PreconditionerS>
  BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
  BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                              const PreconditionerS &preconditioner_S)
    :
    preconditioner_A(preconditioner_A),
    preconditioner_S(preconditioner_S)
  {}


  template <class PreconditionerA, class PreconditionerS>
  void
  BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
  vmult(TrilinosWrappers::MPI::BlockVector       &dst,
        const TrilinosWrappers::MPI::BlockVector &src) const
  {
    preconditioner_A.vmult(dst.block(0), src.block(0));
    preconditioner_S.vmult(dst.block(1), src.block(1));
  }

}  // end of namespace
