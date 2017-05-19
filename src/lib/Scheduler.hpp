#pragma once

#include<Well.hpp>

namespace RHS
{
	using namespace dealii;

	template <int dim>
	class Scheduler
	{
	public:
		std::vector<RHS::WellControl> get_well_controls(const double time) const;
		void add_well(const unsigned int idx, const std::string &name);
		void set_schedule(const std::vector<double>       &times,
			                const std::vector<std::string>  &well_names,
											const std::vector<unsigned int> &control_values,
											const std::vector<double>       &values);
    void add_line(const double       time,
			            const std::string  &well_name,
									const unsigned int control_value,
									const double       value);
	unsigned int get_well_index(const std::string &wname) const;

  private:
		std::vector<double> times, values;
		std::vector<std::string> schedule_well_names;
		std::vector<unsigned int> control_values;
		std::map<std::string, unsigned int> indexing;
	};


	template <int dim> void
	Scheduler<dim>::add_well(const unsigned int idx, const std::string &name)
	{
		indexing[name] = idx;
	} // eom


	template <int dim> void
	Scheduler<dim>::add_line(const double       time,
								           const std::string  &well_name,
													 const unsigned int control_value,
													 const double       value)
  {
		if (times.size() > 0)
			AssertThrow(time >= times.back(),
				ExcMessage("Schedule should be in an ascending order"));

		times.push_back(time);
		schedule_well_names.push_back(well_name);
		control_values.push_back(control_value);
		values.push_back(value);
	} // eom

	template <int dim> unsigned int
	Scheduler<dim>::get_well_index(const std::string &wname) const
	{
		// std::cout << "Well: " << wname << std::endl;
		// std::cout << "ind: " << indexing.find(wname)->second << std::endl;
		return indexing.find(wname)->second;
	}

	template <int dim> std::vector<RHS::WellControl>
	Scheduler<dim>::get_well_controls(const double time) const
	{
		AssertThrow(times.size() > 0, ExcMessage("Schedule is empty"));

		// std::cout << schedule_well_names[0] << "aaa" << std::endl;
		get_well_index(this->schedule_well_names[0]);

		unsigned int n_wells = indexing.size();
		std::vector<RHS::WellControl> controls(n_wells);
		for (unsigned int i=0; i<times.size(); ++i)
		{
			double t = this->times[i];
			if (t > time)
				break;
			else
			{
				unsigned int idx = get_well_index(schedule_well_names[i]);
				controls[idx].control_value = control_values[i];
				controls[idx].value = values[i];

			}
		}  // end loop though schedule table

		return controls;
	} // eom

} // end of namespace
