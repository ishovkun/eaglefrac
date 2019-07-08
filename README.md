# EagleFrac

This is a C++ code written to model fracture propagation in a 2D porous medium.

The code is based on the following papers:

1. I Shovkun, DN Espinoza, 2019, Propagation of toughness-dominated fluid-driven fractures in reactive porous media
International Journal of Rock Mechanics and Mining Sciences 118, 42-51.

2. I. Shovkun, D. Nicolas Espinoza, Fracture Propagation in Heterogeneous Porous Media: Pore-Scale Implications of Mineral Dissolution. Rock Mechanics and Rock Engineering. 2019

Please cite them :-)

![Awesome screenshot](./Screenshot.png)

## Current state
This repo currently contains 2 models: eaglefrac-solid and eaglefrac-fluid.

The eaglefrac-solid model assumes no pore fluid and that the fractures are driven
by moving boundaries (dirichlet displacement conditions).

The eaglefrac-fluid assumes constant displacement boundaries fractures driven
by pressure source terms (currently implemented constant-rate wellbores).

A more thorough description of the model and the input file structure will
be added to the Wiki page when I'm in the mood.

## Features
- Works in parallel (MPI)
- Adaptive mesh refinement
- Input files
- Reads mesh in .msh format (exported from GMsh)
- Parallel output to .vtu

## Usage
**IMPORTANT**: apparently, this code only works with an older version of deal-ii 8.4.0.
Build deallii with mpi, Trilinos, and p4est from
- dealii http://www.dealii.org/download.html
- Trilinos https://trilinos.org/
- p4est http://www.p4est.org/

Also, you need to have Boost library built or installed
(libboost-all-dev on Ubuntu)

### Building Trilinos for dealii
The following installs dealii into /home/user_name/share/trilinos
~~~~
cmake \
-DTrilinos_ENABLE_Amesos=ON \
-DTrilinos_ENABLE_Epetra=ON \
-DTrilinos_ENABLE_Ifpack=ON \
-DTrilinos_ENABLE_AztecOO=ON \
-DTrilinos_ENABLE_Sacado=ON \
-DTrilinos_ENABLE_Teuchos=ON \
-DTrilinos_ENABLE_MueLu=ON \
-DTrilinos_ENABLE_ML=ON \
-DTrilinos_VERBOSE_CONFIGURE=OFF \
-DTPL_ENABLE_MPI=ON \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_VERBOSE_MAKEFILE=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX:PATH=$HOME/share/trilinos \
..

make
make install
~~~~

### Building dealii
This process is pretty straightforward and is described at http://www.dealii.org/8.4.1/readme.html.

I ran the following cmake command (for system-wide installations)
~~~~
cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DDEAL_II_WITH_MPI=ON \
      -DDEAL_II_WITH_TRILINOS=ON \
      -DDEAL_II_WITH_P4EST=ON
      -DTRILINOS_DIR=/path/to/trilinos \
      -DP4EST_DIR=/path/to/p4est \
      ..
~~~~

### Building p4est
That was super easy so I didn't even save instructions for it. Sorry about that.

### Running
The following runs the eaglefrac-solid model with three-point-bending input:
~~~~
cmake .
make
mpirun -np 1 ./eaglefrac-solid ./input/solid-three_point_bending.prm
~~~~
Likewise, to run eaglefrac-fluid (e.g. with a one-fracture case input) run:
~~~~
mpirun -np 1 ./eaglefrac-fluid ./input/fluid-one_frac.prm
~~~~

## Notes
Author: Igor Shovkun
