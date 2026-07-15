% scale(1000) import("knee_pitch_housing_top_left.stl");

// Sketch knee_pitch_housing_top_left 70
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 36.75000000000001], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 70.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=60.000000,h=thickness);
  }
}
}
