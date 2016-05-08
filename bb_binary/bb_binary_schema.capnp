@0xa7d4a096084cee0e;

using Java = import "java.capnp";
$Java.package("de.fuberlin.biorobotics");
$Java.outerClassname("BeesBook");


struct Detection {
  idx @0 :UInt64;
  localizerIdx @1 :UInt64;
  x @2 :UInt16;
  y @3 :UInt16;
  zRotation @4 :Float32;
  yRotation @5 :Float32;
  xRotation @6 :Float32;
  radius @7 :Float32;
  decodedId @8 :List(UInt8);
  vote @9 :UInt16;
}


struct FrameColumnwise {
  imageName @0 :Text;
  tagIdx @1 :List(UInt16);            # unique sequential id of the tag
  candidateIdx @2 :List(UInt16);      # sequential id of the candidate per tag
  gridIdx @3 :List(UInt16);           # sequential id of the grid/decoding per candidate
  xpos @4 :List(UInt16);              # x coordinate of the grid center
  ypos @5 :List(UInt16);              # y coordinate of the grid center
  zRotation @6 :List(Float32);        # rotation of the grid in x plane
  yRotation @7 :List(Float32);        # rotation of the grid in y plane
  xRotation @8 :List(Float32);        # rotation of the grid in z plane
  lScore @9 :List(Float32);           # roi score (Localizer)
  eScore @10 :List(UInt16);           # ellipse score (EllipseFitter)
  gScore @11 :List(Float32);          # grid score (GridFitter)
  id @12 :List(UInt32);               # decoded id
}

# Corresponds to an image in the video
struct FrameRowwise {
  id @0 :UInt64;
  timestamp @1 :UInt64;
  detections @2 :List(Detection);
}

# Corresponds to a video
struct FrameContainer {
  id @0 :UInt64;
  hdVideoFilename @1 :Text;
  previewVideoFilename @2 :Text;
  cam @3 :UInt8;
  worldCoordinatesTransform @4 :List(Float32);
  from @5 :UInt64;
  to @6 :UInt64;
  frames: union {
    rowwise @7:List(FrameRowwise);
    columnwise @8:List(FrameColumnwise);
  }
}
