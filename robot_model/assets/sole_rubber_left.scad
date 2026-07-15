% scale(1000) import("sole_rubber_left.stl");

// Sketch sole_rubber_left 245
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 113.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 245.000000;
translate([0, 0, -thickness]) {
  translate([-25.000000, -743.000000, 0]) {
    rotate([0, 0, 180.0]) {
      cube([90.000000, 55.000000, thickness]);
    }
  }
}
}
