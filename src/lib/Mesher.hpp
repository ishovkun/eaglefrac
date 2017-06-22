#pragma once

#include <deal.II/base/utilities.h>
// dealii mesh modules
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

// Custom modules
#include <PhaseFieldSolver.hpp>

namespace Mesher {
  using namespace dealii;


  template <int dim>
  bool
  prepare_phase_field_refinement(PhaseField::PhaseFieldSolver<dim> &pf,
                                 const double min_phi_value,
                                 const int max_refinement_level)
  { // refine if phase field < constant
    {
      pf.relevant_solution = pf.solution;

      const int dofs_per_cell = pf.fe.dofs_per_cell;
      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
        cell = pf.dof_handler.begin_active(),
        endc = pf.dof_handler.end();

        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
              for (int i=0; i<dofs_per_cell; ++i)
              {
                const int c = pf.fe.system_to_component_index(i).first;
                if (c == dim)
                {
                  double phi_value = pf.relevant_solution(local_dof_indices[i]);
                  if (phi_value < min_phi_value)
                    {
                      cell->set_refine_flag();
                      break; /*dof loop*/
                    }
                }  // end if comp
              }  // end dof loop
          }  // end cell looop
    }

    {  // don't refine beyond max level
      typename DoFHandler<dim>::active_cell_iterator
        cell = pf.dof_handler.begin_active(),
        endc = pf.dof_handler.end();

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned()
            && cell->level() == max_refinement_level)
          cell->clear_refine_flag();
    }

    {  // determine whether mesh is changed
      bool any_change = false;
      pf.triangulation.prepare_coarsening_and_refinement();

      typename DoFHandler<dim>::active_cell_iterator
        cell = pf.dof_handler.begin_active(),
        endc = pf.dof_handler.end();

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned() &&
            (cell->refine_flag_set() || cell->coarsen_flag_set()))
          {
            any_change = true;
            break;
          }

      if (Utilities::MPI::sum(any_change?1:0, pf.mpi_communicator)==0)
        return false;
      else
        return true;
    }  // end check

  }  // eom

  template <int dim>
  double
  compute_minimum_mesh_size(parallel::distributed::Triangulation<dim> &triangulation,
                            MPI_Comm &mpi_communicator)
  {
    double min_cell_size = std::numeric_limits<double>::max();

    typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        min_cell_size = std::min(cell->diameter(), min_cell_size);

    min_cell_size = -Utilities::MPI::max(-min_cell_size, mpi_communicator);
    return min_cell_size;

  }  // eom

  template <int dim>
  void refine_region(parallel::distributed::Triangulation<dim> &triangulation,
                     const std::vector< std::pair<double,double> > &region,
                     const int n_refinement_cycles=1)
  {
    AssertThrow(region.size() == dim, ExcMessage("Dimension mismatch"));
    Assert(dim==2, ExcNotImplemented());

    for (int cycle=0; cycle<n_refinement_cycles; ++cycle)
    {
      typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
				{
					if (
            cell->center()[0] > region[0].first
						&&
            cell->center()[0] < region[0].second
						&&
            cell->center()[1] > region[1].first
						&&
            cell->center()[1] < region[1].second
					)
					{
						cell->set_refine_flag();
						continue;
					}
          for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
            if (
              cell->vertex(v)[0] > region[0].first
              &&
              cell->vertex(v)[0] < region[0].second
              &&
              cell->vertex(v)[1] > region[1].first
              &&
              cell->vertex(v)[1] < region[1].second
            )
            {
              cell->set_refine_flag();
              break;
            }
				}

      triangulation.execute_coarsening_and_refinement();
    }  // end cycle loop

  }  // eof

}  // end of namespace
