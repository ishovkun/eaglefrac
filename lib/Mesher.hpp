#pragma once

// dealii mesh modules
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

// Custom modules
#include <PhaseFieldSolver.hpp>

namespace Mesher {
  using namespace dealii;


  // template <int dim>
  // class Mesher
  // {
  // public:
  //   // Mesher();
  //   bool prepare_phase_field_refinement(PhaseField::PhaseFieldSolver<dim> &,
  //                                       const double,
  //                                       const unsigned int);
  // };


  template <int dim>
  bool
  prepare_phase_field_refinement(PhaseField::PhaseFieldSolver<dim> &pf,
                                 const double min_phi_value,
                                 const int max_refinement_level)
  { // refine if phase field < constant
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
                    break; /*i loop*/
                  }
              }  // end if comp
            }  // end dof loop
        }  // end cell looop

    // estimate error per cell
    Vector<float> estimated_error_per_cell(pf.triangulation.n_active_cells());
    std::vector<bool> component_mask(dim+1, true);
    component_mask[dim] = false;

    KellyErrorEstimator<dim>::estimate(pf.dof_handler,
                                       QGauss<dim-1>(pf.fe.degree+2),
                                       typename FunctionMap<dim>::type(),
                                       pf.relevant_solution,
                                       estimated_error_per_cell,
                                       component_mask,
                                       0,
                                       0,
                                       pf.triangulation.locally_owned_subdomain());

    { // don't touch cells in the fracture
      typename DoFHandler<dim>::active_cell_iterator
        cell = pf.dof_handler.begin_active(),
        endc = pf.dof_handler.end();
      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      unsigned int idx = 0;
      for (; cell != endc; ++cell, ++idx)
        if (cell->is_locally_owned())
          if (cell->refine_flag_set())
            estimated_error_per_cell[idx] = 0.0;
    }

    parallel::distributed::GridRefinement::
    refine_and_coarsen_fixed_number(pf.triangulation,
                                    estimated_error_per_cell,
                                    0.3, 0.0);

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
      bool refine_or_coarsen = false;
      pf.triangulation.prepare_coarsening_and_refinement();

      typename DoFHandler<dim>::active_cell_iterator
        cell = pf.dof_handler.begin_active(),
        endc = pf.dof_handler.end();

      for (; cell != endc; ++cell)
        if (cell->is_locally_owned() &&
            (cell->refine_flag_set() || cell->coarsen_flag_set()))
          {
            refine_or_coarsen = true;
            break;
          }

      if (Utilities::MPI::sum(refine_or_coarsen?1:0, pf.mpi_communicator)==0)
        return false;
      else
        return true;
    }  // end check

  }  // eom

}  // end of namespace
