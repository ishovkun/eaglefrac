#pragma once

#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>

namespace Functions
{
  using namespace dealii;

  template <int dim>
  class FEFunction : public Function<dim>
  {
  public:
    FEFunction(const DoFHandler<dim>& dof_handler_,
                    const TrilinosWrappers::MPI::BlockVector& input_values_,
                    double (*mapping_)(const double key));
    double value(const Point<dim> &p,
                 const unsigned int /*component*/ c = 0) const;
  private:
    const DoFHandler<dim>& dof_handler;
    const TrilinosWrappers::MPI::BlockVector& input_values;
    double (&mapping)(const double key);
  };


  template <int dim>
  FEFunction<dim>::
  FEFunction(const DoFHandler<dim>& dof_handler_,
                  const TrilinosWrappers::MPI::BlockVector& input_values_,
                  double (*mapping_)(const double key))
    :
    dof_handler(dof_handler_),
    input_values(input_values_),
    mapping(*mapping_)
  {}  // eom


  template <int dim>
  double FEFunction<dim>::value(const Point<dim> &p,
                                     const unsigned int /*component*/ c) const
  {
    /* Don't call this function before setup_dofs */

    const auto & fe = dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    // std::cout << "dofs per cell " << dofs_per_cell << std::endl;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    AssertThrow(dof_handler.has_active_dofs(),
                ExcMessage("DofHandler is empty"));

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
		  endc = dof_handler.end();

    for (; cell != endc; ++cell)
      // if (cell->is_locally_owned())
      if (!cell->is_artificial())
      {
        if (cell->point_inside(p))
        {
          // std::cout << "found point" << std::endl;
          cell->get_dof_indices(local_dof_indices);
          return mapping(input_values[local_dof_indices[0]]);
          // break;
        }
      }  // end cell loop

    if (c == 0)
      return mapping(input_values[local_dof_indices[0]]);
    else
      return 0;
  }  // eom


}  // end of namespace
