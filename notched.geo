xSize = 1e-2;
ySize = 1e-2;
notch_width = ySize/100;
m_size_coarse = xSize/5;
m_size_fine = notch_width/5;

// Outer points
Point(1) = {-xSize/2, -ySize/2, 0, m_size_coarse};  // bottom left
Point(2) = {+xSize/2, -ySize/2, 0, m_size_coarse};  // bottom right
Point(3) = {+xSize/2, +ySize/2, 0, m_size_coarse};  // top right
Point(4) = {-xSize/2, +ySize/2, 0, m_size_coarse};  // top left

// Notch point
Point(5) = {0, 0, 0, m_size_fine};

// notch corners
Point(6) = {+xSize/2, -notch_width/2, 0, m_size_coarse};
Point(7) = {+xSize/2, +notch_width/2, 0, m_size_coarse};

// refinement line corners
Point(8) = {-xSize/2, 0, 0, m_size_fine};
Point(9) = {-notch_width, 0, 0, m_size_fine};

// lines of the outer box:
// Bottom
Line(1) = {1, 2};
// Right bottom to notch corner
Line(2) = {2, 6};
// Notch corner to notch vertex
Line(3) = {6, 5};
Line(4) = {5, 7};
// notch corner to right top
Line(5) = {7, 3};
// top
Line(6) = {3, 4};
// Left
Line(7) = {4, 8};
Line(8) = {8, 9};
Line(9) = {8, 1};

//
Line Loop(10) = {1, 2, 3, 4, 5, 6, 7, 8, -8, 9};
Plane Surface(1) = {10};

// these define the boundary indicators in deal.II:
Physical Line(0) = {7, 9}; // Left
Physical Line(1) = {2, 5}; // Right
Physical Line(2) = {1}; // Bottom
Physical Line(3) = {6}; // Top

// you need the physical surface, because that is what deal.II reads in
Physical Surface(11) = {10};

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
// Mesh.CharacteristicLengthFactor = 0.09;
Mesh.SubdivisionAlgorithm = 2;
Mesh.Smoothing = 20;
Show "*";
