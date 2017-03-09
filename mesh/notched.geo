domain_size = 1;
notch_size = 1e-6;
element_size = 0.25;
Point(1) = {-domain_size/2, -domain_size/2,  0.0, element_size };  // bottom left
Point(2) = {+domain_size/2, -domain_size/2,  0.0, element_size };  // bottom right
Point(3) = {+domain_size/2, -notch_size/2,   0.0, element_size };  // middle right
Point(4) = {+0,             +0,              0.0, element_size };  // center
Point(5) = {+domain_size/2, +notch_size/2,   0.0, element_size };  // middle right
Point(6) = {+domain_size/2, +domain_size/2,  0.0, element_size };  // top right
Point(7) = {-domain_size/2, +domain_size/2,  0.0, element_size };  // top left

Line(8)  = {1,2};  // bottom
Line(9)  = {2,3};  // right bottom
Line(10) = {3,4};  // notch bottom
Line(11) = {4,5};  // notch top
Line(12) = {5,6};  // right top
Line(13) = {6,7};  // top
Line(14) = {7,1};  // left

Line Loop(15) = {8,9,10,11,12,13,14};
Plane Surface(16) = {15};

// Boundary indicators
Physical Line(0) = {14};     // left
Physical Line(1) = {9, 12};  // right
Physical Line(2) = {8};      // bottom
Physical Line(3) = {13};     // top
Physical Line(4) = {10};     // notch bottom
Physical Line(5) = {11};     // notch top


Physical Surface(17) = {16};

Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
// Mesh.CharacteristicLengthFactor = 4*element_size;
Mesh.SubdivisionAlgorithm = 3;
Mesh.Smoothing = 20;
Show "*";
