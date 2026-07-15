% scale(1000) import("femur_right.stl");

// Sketch femur_right 200
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -20.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 200.000000;
translate([0, 0, -thickness]) {
  translate([9.000000, 4.000000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}
