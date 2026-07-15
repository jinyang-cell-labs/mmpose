% scale(1000) import("torso_roll_holder.stl");

// Sketch torso_roll_holder_cylinder 70
multmatrix([[0.0, 0.0, -1.0, -60.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 70.000000;
translate([0, 0, -thickness]) {
  translate([12.250000, -0.000000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}
