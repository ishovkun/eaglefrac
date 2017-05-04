#pragma once

namespace RHS
{
	using namespace dealii;

	template <int dim>
	class Well : public Function<dim>
	{
		/*
		This class returns the value of flow rate as a function of location (Point).
		If the well is not located (method locate hasn't been called,
		the value of flow rate is returned in the region
		(||Point - true_location|| < location_radius).
		Else, the flow rate value is returned only in the center of the cell
		closest to the true location
		*/
	public:
		Well(const Point<dim> &loc_,
				 const double     rate_ = 0.0,
			 	 double           location_radius_ = 1.0);

		virtual double value(const Point<dim> &p,
												 const unsigned int component=0) const;
    virtual void point_values(const std::vector<Point<dim> > &points,
															std::vector<double>            &dst) const;

    void set_location_radius(const double lr);

		// set_control(const double value, const int control);
		void locate(const DoFHandler<dim>  &dof_handler,
								MPI_Comm 				 &mpi_communicator);

		private:
			const Point<dim> &true_location;
			double 					 flow_rate;
			Point<dim>       closest_cell_center;
			double					 location_radius;
	};  //  end of class definition


	template <int dim>
	Well<dim>::Well(const Point<dim> &loc,
			 				 	 	const double		 rate,
									double           location_radius_)
  :
	Function<dim>(1),
	true_location(loc),
	flow_rate(rate),
	location_radius(location_radius_)
	{
	}  // eom


	template <int dim> void
	Well<dim>::set_location_radius(const double lr)
	{
		location_radius = lr;
	}  // eom

	template <int dim> void
	Well<dim>::locate(const DoFHandler<dim>  &dof_handler,
										MPI_Comm 				 			 &mpi_communicator)
	{ // in this function we find the coordinates of the cell center that
		// is closest to the well location
		double min_distance = std::numeric_limits<double>::max();
		Point<dim> best_cell(std::numeric_limits<double>::max(),
												 std::numeric_limits<double>::max());

		typename DoFHandler<dim>::active_cell_iterator
	    cell = dof_handler.begin_active(),
	    endc = dof_handler.end();

	  for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
			{
				auto const &p = cell->center();
				const double d = true_location.distance(p);
				if (d < min_distance)
				{
					min_distance = d;
					best_cell = p;
				}
			}  // end cell loop

    const double overall_min_distance =
			Utilities::MPI::min(min_distance, mpi_communicator);

		/* several processors can actually have cells at the same distance from
			the true source location.
			The following trick should let only one processor (with the minimum number)
			to set the source.
			if we sum the processor number and minimum distance this should be enough
			to uniquely set the well
		*/
		const unsigned int this_mpi_process =
			Utilities::MPI::this_mpi_process(mpi_communicator);

		const double magic_number =
			min_distance + static_cast<double>(this_mpi_process);

		const double min_magic_number =
			Utilities::MPI::min(magic_number, mpi_communicator);

		const	double large_num = std::numeric_limits<double>::max();

		if (min_distance == overall_min_distance &&
				magic_number == min_magic_number)
		{
			closest_cell_center = best_cell;
		}
		else  // assign to some cell far away
		{
			for (int i=0; i<dim; ++i)
				closest_cell_center[i] = large_num;
		}

		location_radius = 1e-10;

		std::cout << "Setting well into ("
							<< closest_cell_center
							<< ")"
							<< std::endl;
			// std::cout << "loc " << loc << std::endl;
	}  // eom


	template <int dim>
	double Well<dim>::value(const Point<dim> &p,
													const unsigned int component) const
  {
		if (component == 0)
		{
			// std::cout << "point " << p << std::endl;
			// std::cout << "rate " << flow_rate << std::endl;
			// std::cout << "best_cell " << p << std::endl;
			if (closest_cell_center.distance(p) < location_radius)
				return flow_rate;
			else
				return 0.0;
		}
		else
			return 0.0;
	}  // eom


	template <int dim> void
	Well<dim>::point_values(const std::vector<Point<dim> > &points,
													std::vector<double>            &dst) const
  {
		Assert(dst.size() == points.size(),
					 ExcDimensionMismatch(dst.size(), points.size()));

		for (unsigned int p=0; p<points.size(); ++p)
			dst[p] = Well<dim>::value(points[p], 0);
  }  // eom

}  // end of namespace
