% scale(1000) import("hip_pitch_holder_right.stl");

// Sketch hip_pitch_holder_right 70
multmatrix([[0.0, 0.0, -1.0, -60.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 70.000000;
translate([0, 0, -thickness]) {
  translate([38.039284, -120.645194, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}
