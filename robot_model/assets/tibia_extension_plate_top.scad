% scale(1000) import("tibia_extension_plate_top.stl");

// Sketch tibia_extension_plate_top 300
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -95.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 300.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=43.750000,h=thickness);
  }
}
}
