% scale(1000) import("gim_3505_8_simplified.stl");

// Sketch gim_3505_8_simplified 24
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 24.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=22.000000,h=thickness);
  }
}
}
