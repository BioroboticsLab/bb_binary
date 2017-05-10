@0xa7d4a096084cee0e;

using Java = import "java.capnp";
$Java.package("de.fuberlin.biorobotics");
$Java.outerClassname("BeesBook");


struct DetectionCVP {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  candidateIdx @1 :UInt16;       # sequential index of the candidate per tag
  gridIdx @2 :UInt16;            # sequential index of the grid/decoding per candidate
  xpos @3 :UInt16;               # x coordinate of the grid center
  ypos @4 :UInt16;               # y coordinate of the grid center
  xposHive @5 :UInt16;           # x coordinate of the grid center wrt. the hive
  yposHive @6 :UInt16;           # y coordinate of the grid center wrt. the hive
  zRotation @7 :Float32;         # rotation of the grid in z plane
  yRotation @8 :Float32;         # rotation of the grid in y plane
  xRotation @9 :Float32;         # rotation of the grid in x plane
  lScore @10 :Float32;           # roi score (Localizer)
  eScore @11 :UInt16;            # ellipse score (EllipseFitter)
  gScore @12 :Float32;           # grid score (GridFitter)
  decodedId @13 :UInt32;         # decoded id
}

struct DetectionDP {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  yposHive @3 :UInt16;           # y coordinate of the grid center wrt. the hive
  xposHive @4 :UInt16;           # x coordinate of the grid center wrt. the hive
  zRotation @5 :Float32;         # rotation of the grid in z plane
  yRotation @6 :Float32;         # rotation of the grid in y plane
  xRotation @7 :Float32;         # rotation of the grid in x plane
  radius @8 :Float32;            # radius of the tag
  localizerSaliency @9 :Float32; # saliency of the localizer network
  decodedId @10 :List(UInt8);    # the decoded id, the bit probabilities are discretised to 0-255.
                                 # p(first bit == 1) = decodedId[0] / 255. bits are in most significant 
                                 # bit first order starting at the 1 o'clock position on the tag in 
                                 # clockwise orientation.
                                 # see https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical
                                 # representation
  descriptor @11 :List(UInt8);   # visual descriptor of the detection. ordered from most
                                 # significant eight bits to least significant eight bits.
}

struct DetectionTruth {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  yposHive @3 :UInt16;           # y coordinate of the grid center wrt. the hive
  xposHive @4 :UInt16;           # x coordinate of the grid center wrt. the hive
  decodedId @5 :Int32;           # decoded id by human
  readability @6 :Grade;         # tags might be visible or (partially) obscured
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

# Corresponds to a video
struct FrameContainer {
  id @0 :UInt64;                    # global unique id of the frame container
  dataSources @1:List(DataSource);  # list of data sources (videos / images)
  fromTimestamp @2 :Float64;        # unix timestamp of the first frame
  toTimestamp @3 :Float64;          # unix timestamp of the last frame
  frames @4 :List(Frame);           # frames must be sorted by in the order they where recorded.
  camId @5 :UInt16;                 # the cam number
  hiveId @6 :UInt16;                # the id of the hive
  transformationMatrix @7 :List(Float32);
                                    # the transformation matrix from image coordinates to hive coordinates.
                                    # The matrix is of dimension 4x4 and stored this way
                                    #     1 | 2 | 3 | 4
                                    #     5 | 6
                                    #          ...
                                    #             15| 16
}
