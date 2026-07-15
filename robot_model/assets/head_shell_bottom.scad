% scale(1000) import("head_shell_bottom.stl");

// Sketch head_shell_bottom 210
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 811.066191236011], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 210.000000;
translate([0, 0, -thickness]) {
  translate([-0.000000, 20.000000, 0]) {
    cylinder(r=105.000000,h=thickness);
  }
}
}
