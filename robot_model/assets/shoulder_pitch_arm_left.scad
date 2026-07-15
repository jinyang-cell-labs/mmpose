% scale(1000) import("shoulder_pitch_arm_left.stl");

// Sketch shoulder_pitch_arm_left 70
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 36.2500000000031], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 70.000000;
translate([0, 0, -thickness]) {
  translate([55.000000, -17.500000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}
