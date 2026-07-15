% scale(1000) import("torso_main_body.stl");

// Sketch torso_main_body 200
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 89.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 200.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, -6.000000, 0]) {
    cylinder(r=80.000000,h=thickness);
  }
}
}
