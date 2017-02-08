xSize = 10;
ySize = 10;
notch_width = 0.05;

// Outer points
Point(1) = {-xSize/2, -ySize/2, 0, 1};  // bottom left
Point(2) = {+xSize/2, -ySize/2, 0, 1};  // bottom right
Point(3) = {+xSize/2, +ySize/2, 0, 1};  // top right
Point(4) = {-xSize/2, +ySize/2, 0, 1};  // top left

// Notch point
Point(5) = {0, 0, 0, 1};

// notch corners
Point(6) = {+xSize/2, -notch_width/2, 0, 1};
Point(7) = {+xSize/2, +notch_width/2, 0, 1};


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
Line(7) = {4, 1};

//
Line Loop(8) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(9) = {8};

// you need the physical surface, because that is what deal.II reads in
Physical Surface(10) = {9};

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
// Mesh.CharacteristicLengthFactor = 0.09;
Mesh.SubdivisionAlgorithm = 2;
Mesh.Smoothing = 20;
Show "*";
