@0xec425939339b04cf;

using Java = import "java.capnp";
$Java.package("de.fuberlin.biorobotics");
$Java.outerClassname("BeesBook");


struct HiveMappedDetection {
  xpos @0 :Float32;               # x coordinate of the grid center wrt. the hive
  ypos @1 :Float32;               # y coordinate of the grid center wrt. the hive
  zRotation @2 :Float32;          # rotation of the grid in z plane wrt. the hive
  radius @3 :Float32;             # radius of the tag
}

struct DetectionCVP {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  candidateIdx @1 :UInt16;       # sequential index of the candidate per tag
  gridIdx @2 :UInt16;            # sequential index of the grid/decoding per candidate
  xpos @3 :UInt16;               # x coordinate of the grid center
  ypos @4 :UInt16;               # y coordinate of the grid center
  zRotation @5 :Float32;         # rotation of the grid in z plane
  yRotation @6 :Float32;         # rotation of the grid in y plane
  xRotation @7 :Float32;         # rotation of the grid in x plane
  hiveMappedDetection @8 :HiveMappedDetection;
  lScore @9 :Float32;           # roi score (Localizer)
  eScore @10 :UInt16;            # ellipse score (EllipseFitter)
  gScore @11 :Float32;           # grid score (GridFitter)
  decodedId @12 :UInt32;         # decoded id
}

struct DetectionDP {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  zRotation @3 :Float32;         # rotation of the grid in z plane
  yRotation @4 :Float32;         # rotation of the grid in y plane
  xRotation @5 :Float32;         # rotation of the grid in x plane
  radius @6 :Float32;            # radius of the tag
  hiveMappedDetection @7 :HiveMappedDetection;
  localizerSaliency @8 :Float32; # saliency of the localizer network
  decodedId @9 :List(UInt8);    # the decoded id, the bit probabilities are discretised to 0-255.
                                 # p(first bit == 1) = decodedId[0] / 255. bits are in most significant
                                 # bit first order starting at the 1 o'clock position on the tag in
                                 # clockwise orientation.
                                 # see https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical
                                 # representation
  descriptor @10 :List(UInt8);   # visual descriptor of the detection. ordered from most
                                 # significant eight bits to least significant eight bits.
}

struct DetectionTruth {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  hiveMappedDetection @3 :HiveMappedDetection;
  decodedId @4 :Int32;           # decoded id by human
  readability @5 :Grade;         # tags might be visible or (partially) obscured
  enum Grade {                   # ranks for evaluation of a tag's readability are:
    unknown @0;                  #  - not considered or evaluted
    completely @1;               #  - completely visible **and** human readable
    partially @2;                #  - only partially visible and therefore **not** human readable
    none @3;                     #  - **not** visible at all
  }
}

# Corresponds to an image in the video.
struct Frame {
  id @0 :UInt64;                 # global unique id of the frame
  dataSourceIdx @1:UInt32;       # the frame is from this data source
  frameIdx @6 :UInt32;           # sequential increasing index for every data source.
  timestamp @2 :Float64;         # unix time stamp of the frame
  timedelta @3 :UInt32;          # time difference between this frame and the frame before in microseconds
  detectionsUnion : union {
    detectionsCVP   @4 :List(DetectionCVP);   # detections format of the old computer vision pipeline
    detectionsDP    @5 :List(DetectionDP);    # detections format of the new deeppipeline
    detectionsTruth @7 :List(DetectionTruth); # detections format of ground truth data
  }
}

struct DataSource {
    idx @0 :UInt32;                  # the index of the data source
    filename @1 :Text;               # filename of the data source
    videoPreviewFilename @2 :Text;   # (optional) filename of the preview video
}

struct HiveMappingData {
  transformationMatrix @0 :List(Float64);
                                    # the transformation matrix to map the detections data to the
                                    # stitched panorama. These coordinates are also in image
                                    # coordinates. The HiveMappedDetections are calculate in
                                    # connection with the ratioPxMm and the origin.
                                    # The matrix is of dimension 3x3 and stored this way
                                    #     1 | 2 | 3
                                    #     4 | 5 | 6
                                    #     7 | 8 | 9
  origin @1 :List(Float32);         # origin (in px)
  ratioPxMm @2 :Float64;            # ratio px:mm
  frameSize @3 :List(UInt16);      # Size of the frame in px (width, height)
  mapsToCamId @4 :UInt16;          # the id of the cam which captures the other part of the hiveside
}

# Corresponds to a video
struct FrameContainer {
  id @0 :UInt64;                    # global unique id of the frame container
  dataSources @1:List(DataSource);  # list of data sources (videos / images)
  fromTimestamp @2 :Float64;        # unix timestamp of the first frame
  toTimestamp @3 :Float64;          # unix timestamp of the last frame
  frames @4 :List(Frame);           # frames must be sorted by in the order they where recorded.
  camId @5 :UInt16;                 # the cam number
  hiveId @6 :UInt16;                # the id of the hive
  hiveMappingData @7 :HiveMappingData;
}
