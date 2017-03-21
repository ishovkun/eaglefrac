#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>


namespace BitMap {
  using namespace dealii;

  class BitMapFile
  {
  public:
    BitMapFile(const std::string &name);
    double get_value(const double x, const double y) const;

  private:
    std::vector<double> image_data;
    double hx, hy;
    double maxvalue = 255;
    int nx, ny;
    double get_pixel_value(const int i, const int j) const;
  };


    BitMapFile::BitMapFile(const std::string &name)
    :
    image_data(0),
    hx(0),
    hy(0),
    nx(0),
    ny(0)
  {
    std::ifstream f(name.c_str());
    AssertThrow (f, ExcMessage (std::string("Can't read from file <") +
                                name + ">!"));

    std::string temp;
    getline(f, temp);
    f >> temp;
    // if (temp[0]=='#')
    if (temp.compare(0, 1, "#") == 0)
      getline(f, temp);

    f >> nx >> ny;
    // std::cout << "nx " << nx << " ny " << ny << std::endl;
    f >> maxvalue;
    // std::cout << "max_pixel " << maxvalue << std::endl;

    AssertThrow(nx > 0 && ny > 0,  ExcMessage("Invalid file format."));
    AssertThrow(maxvalue == 255.0, ExcMessage("Invalid file format."));

    for (int k = 0; k < nx * ny; k++)
      {
        unsigned int val;
        f >> val;
        image_data.push_back(val / maxvalue);
        AssertThrow(val <= maxvalue, ExcMessage("Invalid file format."));
      }

    hx = 1.0 / (nx - 1);
    hy = 1.0 / (ny - 1);
  }  // eom


  double BitMapFile::get_pixel_value(const int i,
                                     const int j) const
  {
    assert(i >= 0 && i < nx);
    assert(j >= 0 && j < ny);
    return image_data[nx * (ny - 1 - j) + i];
  }  // eom

  double BitMapFile::get_value(const double x,
                               const double y) const
  {
    const int ix = std::min(std::max((int) (x / hx), 0), nx - 2);
    const int iy = std::min(std::max((int) (y / hy), 0), ny - 2);
    // std::cout << "nx " << nx << std::endl;
    // std::cout << "ny " << ny << std::endl;
    // std::cout << "x " << x << std::endl;
    // std::cout << "y " << y << std::endl;
    // std::cout << "ix " << ix << std::endl;
    // std::cout << "iy " << iy << std::endl;

    const double xi  = std::min(std::max((x-ix*hx)/hx, 1.), 0.);
    const double eta = std::min(std::max((y-iy*hy)/hy, 1.), 0.);

    return ((1-xi)*(1-eta)*get_pixel_value(ix,iy)
            +
            xi*(1-eta)*get_pixel_value(ix+1,iy)
            +
            (1-xi)*eta*get_pixel_value(ix,iy+1)
            +
            xi*eta*get_pixel_value(ix+1,iy+1));
  }  // eom


  template <int dim>
  class BitMapFunction : public Function<dim>
  {
  public:
    BitMapFunction(const std::string &filename,
                   double x1_, double x2_,
                   double y1_, double y2_,
                   double minvalue_, double maxvalue_)
      :
      Function<dim>(1),
      f(filename),
      x1(x1_), x2(x2_),
      y1(y1_), y2(y2_),
      minvalue(minvalue_),
      maxvalue(maxvalue_)
    {}

    virtual
    double value (const Point<dim> &p,
                  const unsigned int /*component*/) const
    {
      Assert(dim==2, ExcNotImplemented());
      double x = (p(0)-x1)/(x2-x1);
      double y = (p(1)-y1)/(y2-y1);
      return minvalue + f.get_value(x,y)*(maxvalue-minvalue);
    }

  private:
    BitMapFile f;
    double x1,x2,y1,y2;
    double minvalue, maxvalue;
  };

}  // end of namespace
