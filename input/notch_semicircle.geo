in = 0.0254;
rad = 3.1415926/180;

r = 0.75*in;
l_notch = r/5;
w_notch = l_notch/15;

coarse = r/10;
fine = w_notch/4;

Point(1) = {-r, 0, 0, coarse};  // left lower
Point(2) = {-w_notch, 0, 0, coarse};  // left notch
Point(3) = {0, l_notch, 0, 0, fine};  // notch center
Point(4) = {w_notch, 0, 0, coarse};  // right notcjh
Point(5) = {r, 0, 0, coarse};  // right lower
Point(6) = {0, r, 0, coarse};  // top center (point on ellipse)
Point(7) = {0, 0, 0, fine};   // bottom center (center of ellipse)
// points on ellipse
phi = 10*rad;
Point(8) = {r*Cos(90*rad-phi/2), r*Sin(90*rad-phi/2), 0, coarse};
Point(9) = {r*Cos(90*rad+phi/2), r*Sin(90*rad+phi/2), 0, coarse};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Circle(5) = {5, 7, 8};
Circle(6) = {8, 7, 6};
Circle(7) = {6, 7, 9};
Circle(8) = {9, 7, 1};
Line Loop(9) = {1, 2, 3, 4, 5, 6, 7, 8};

Plane Surface(10) = {9};
Physical Surface(11) = {10};

// physical boundary id's to compute load on (top boundary)
Physical Line(1) = {6, 7}; // top arc

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
/*Mesh.CharacteristicLengthFactor = w_notch;*/
Mesh.SubdivisionAlgorithm = 2;
Mesh.Smoothing = 20;
Show "*";
