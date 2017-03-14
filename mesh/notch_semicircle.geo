in = 0.0254;
r = 0.75*in;
l_notch = r/5;
w_notch = l_notch/15;

coarse = r/10;
fine = w_notch*4;

Point(1) = {-r, 0, 0, coarse};  // left lower
Point(2) = {-w_notch, 0, 0, fine};  // left notch
Point(3) = {0, l_notch, 0, 0, fine};  // notch center
Point(4) = {w_notch, 0, 0, fine};  // right notcjh
Point(5) = {r, 0, 0, coarse};  // right lower
Point(6) = {0, r, 0, coarse};  // top center (point on ellipse)
Point(7) = {0, 0, 0, fine};   // bottom center (center of ellipse)

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Circle(5) = {5, 7, 6};
Circle(6) = {6, 7, 1};
Line Loop(7) = {1, 2, 3, 4, 5, 6};

Plane Surface(8) = {7};
Physical Surface(9) = {8};

// some parameters for the meshing:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1; // to get quadrelaterals
/*Mesh.CharacteristicLengthFactor = w_notch;*/
Mesh.SubdivisionAlgorithm = 2;
Mesh.Smoothing = 20;
Show "*";
