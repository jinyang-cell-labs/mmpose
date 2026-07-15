% scale(1000) import("hip_center.stl");

// Sketch hip_center 130
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -80.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 130.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, -0.000000, 0]) {
    cylinder(r=65.000000,h=thickness);
  }
}
}
