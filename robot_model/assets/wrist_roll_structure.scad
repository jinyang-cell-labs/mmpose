% scale(1000) import("wrist_roll_structure.stl");

// Sketch wrist_roll_structure 145
multmatrix([[0.0, 0.0, 1.0, 85.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 145.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=22.500000,h=thickness);
  }
}
}
