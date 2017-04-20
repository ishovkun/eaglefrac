#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
// #include <deal.II/numerics/vector_tools.h>

namespace InitialValues
{
using namespace dealii;

// Initial phase-field distribution
template <int dim>
class Defects : public Function<dim>
{
public:
	Defects(const std::vector< std::vector<double> > &coords_,
					const double minimum_mesh_size);

	virtual double value(const Point<dim> &p,
											 const unsigned int component = 0) const;
	virtual void vector_value(const Point<dim> &p,
														Vector<double> &value) const;

private:
	const double thickness;
	const std::vector< std::vector<double> > &coords;
};  // end class declaration


template <int dim>
Defects<dim>::Defects(const std::vector< std::vector<double> > &coords_,
											const double minimum_mesh_size)
:
Function<dim>(dim+1),
thickness(minimum_mesh_size),
coords(coords_)
{
	for (auto &defect : coords)
	{
		AssertThrow(defect.size() == dim*2,
								ExcMessage("Wrong size of coords"));
	}

}  // eom

template <int dim>
double Defects<dim>::value(const Point<dim> &p,
													 const unsigned int component) const
{
	if (component == dim)
		{
			for (auto & defect : coords)
			{
				// std::cout << "d0 " << defect[0] << std::endl;
				double a = (defect[3] - defect[1])/(defect[2] - defect[0]);
				double b = defect[1] - a*defect[0];
				double dx = thickness*a*std::pow(0.5, a*a + 1.0);
				double dy = thickness*std::pow(0.5, a*a + 1.0);
				// std::cout << "dx " << dx << std::endl;
				if (   p(0) >= defect[0] - dx
			      && p(0) <= defect[2] + dx
			      && p(1) <= a*p(0) + b + dy
			      && p(1) >= a*p(0) + b - dy)
					{
						// std::cout << "o'm here " << std::endl;
						return 0.0;
					}

					// Vertical defect
					double reg = 1e-10;
					if (std::abs(defect[2] - defect[0]) < reg)
					{
						dx = thickness/2;
					  // std::cout << "Vertical" << std::endl;
						if (   p(0) >= defect[0] - dx
							  && p(0) <= defect[0] + dx
								&& p(1) >= defect[1]
								&& p(1) <= defect[3]
						   )
							 {
								//  std::cout << "Vertical1" << std::endl;
								 return 0;
							 }
					}

			}
			// std::cout << "I am here " << std::endl;
			return 1.0;
		}

	return 0.0;
}  // eom


template <int dim>
void
Defects<dim>::vector_value(const Point<dim> &p,
													 Vector<double>   &values) const
{
	for (unsigned int comp = 0; comp < this->n_components; ++comp)
		values(comp) = Defects<dim>::value(p, comp);
		// values(comp) = this->value(p, comp);
}  // eom


}  // end of namespace
