#pragma once

#include <deal.II/base/point.h>

namespace Parsers {
	using namespace dealii;

  template<typename T>
  std::vector<T> parse_string_list(std::string list_string,
                                   std::string delimiter=",")
  {
    std::vector<T> list;
    T item;
    if (list_string.size() == 0) return list;
    std::vector<std::string> strs;
    boost::split(strs, list_string, boost::is_any_of(delimiter));

    for (const auto &string_item : strs)
    {
      std::stringstream convert(string_item);
      convert >> item;
      list.push_back(item);
    }
    return list;
  }  // eom


  template<>
  std::vector<bool> parse_string_list(std::string list_string,
                                      std::string delimiter)
  {
    // std::cout << "Parsing bool list" << std::endl;
    std::vector<bool> list;
    bool item;
    if (list_string.size() == 0) return list;
    std::vector<std::string> strs;
    boost::split(strs, list_string, boost::is_any_of(delimiter));

    for (auto &string_item : strs)
    {
      std::istringstream is(string_item);
      is >> std::boolalpha >> item;
      // std::cout << std::endl << string_item << std::endl;
      // std::cout << item << std::endl;
      list.push_back(item);
    }
    return list;
  }  // eom


	// convert string to a base type
	template <typename T>
	T convert(const std::string &str)
	{
    std::stringstream conv(str);
		T result;
		conv >> result;
		return result;
	}  // eom

  template <int dim>
  std::vector< Point<dim> > parse_point_list(const std::string &str)
  {
    // std::cout << str << std::endl;
    std::vector< Point<dim> > points;
    // std::vector<std::string> point_strings;
    // int point_index = 0;
    unsigned int i = 0;
    // loop over symbols and get strings surrounded by ()
    while (i < str.size())
    {
      if (str.compare(i, 1, "(") == 0)  // if str[i] == "(" -> begin point
        {
          std::string tmp;
          while (i < str.size())
          {
            i++;

            if (str.compare(i, 1, ")") != 0)
              tmp.push_back(str[i]);
            else
              break;
          }
          // Add point
          std::vector<double> coords = parse_string_list<double>(tmp);
          Point<dim> point;
          for (int p=0; p<dim; ++p)
            point(p) = coords[p];
          points.push_back(point);
        }
        i++;
    }

    return points;
  }  // eom


  std::vector<std::string> parse_pathentheses_list(const std::string &str)
  {
    std::vector<std::string> result;
    unsigned int i = 0;
    // loop over symbols and get strings surrounded by ()
    while (i < str.size())
    {
      if (str.compare(i, 1, "(") == 0)  // if str[i] == "(" -> begin point
      {
        std::string tmp;
        while (i < str.size())
        {
          i++;

          if (str.compare(i, 1, ")") != 0)
            tmp.push_back(str[i]);
          else
            break;
       }  // end insize parentheses
       // add what's inside parantheses
       result.push_back(tmp);
      }
      i++;
    }
    return result;
  }  // eom


  std::string parse_command_line(int argc, char *const *argv) {
    std::string filename;
    if (argc < 2) {
      std::cout << "specify the file name" << std::endl;
      exit(1);
    }

    std::list<std::string> args;
    for (int i=1; i<argc; ++i)
      args.push_back(argv[i]);

    int arg_number = 1;
    while (args.size()){
      if (arg_number == 1)
        filename = args.front();
      args.pop_front();
      arg_number++;
    } // EO while args

    return filename;
  }  // EOM

} // end of namespace
