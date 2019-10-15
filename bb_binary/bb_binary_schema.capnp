@0xc60633f2a1dd46c5;


struct DetectionDP {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  zRotation @3 :Float32;         # rotation of the grid in z plane
  yRotation @4 :Float32;         # rotation of the grid in y plane
  xRotation @5 :Float32;         # rotation of the grid in x plane
  radius @6 :Float32;            # radius of the tag
  localizerSaliency @7 :Float32; # saliency of the localizer network
  decodedId @8 :List(UInt8);     # the decoded id, the bit probabilities are discretised to 0-255.
                                 # p(first bit == 1) = decodedId[0] / 255. bits are in most significant
                                 # bit first order starting at the 1 o'clock position on the tag in
                                 # clockwise orientation.
                                 # see https://arxiv.org/pdf/1611.01331.pdf Figure 1(a) for a graphical
                                 # representation
}

struct DetectionTruth {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  decodedId @3 :Int32;           # decoded id by human, in ferwar format, -1 if not decoded
  readability @4 :Grade;         # tags might be visible or (partially) obscured
  enum Grade {                   # ranks for evaluation of a tag's readability are:
    unknown @0;                  #  - not considered or evaluted
    completely @1;               #  - completely visible **and** human readable
    unreadable @2;               #  - **not** human readable with certainty
    untagged @3;                 #  - bee has no tag
    inCell @4;                   #  - thorax is in cell and may not have a tag
    upsideDown @5;               #  - bee is walking on the glass and may not have a tag
  }
}


struct DetectionBee {
  idx @0 :UInt16;                # sequential index of the detection, counted from 0 for every frame
                                 # the combination (idx, Frame.id) is a global key
  ypos @1 :UInt16;               # y coordinate of the grid center wrt. the image
  xpos @2 :UInt16;               # x coordinate of the grid center wrt. the image
  localizerSaliency @3 :Float32; # saliency of the localizer network
  type @4: Type;                 # type of detection
  enum Type {
    untagged @0;                 #  - bee has no tag
    inCell @1;                   #  - thorax is in cell and may not have a tag
    upsideDown @2;               #  - bee is walking on the glass and may not have a tag
  }
}


# Corresponds to an image in the video.
struct Frame {
  id @0 :UInt64;                            # global unique id of the frame
  dataSourceIdx @1:UInt32;                  # the frame is from this data source
  frameIdx @2 :UInt32;                      # sequential increasing index for every data source.
  timestamp @3 :Float64;                    # unix time stamp of the frame
  detectionsDP @4 :List(DetectionDP);       # detections format of the new deeppipeline
  detectionsBees @5: List(DetectionBee);    # detections of bees with no or no visible tag
  detectionsTruth @6 :List(DetectionTruth); # detections format of ground truth data
}

struct DataSource {
    idx @0 :UInt32;                  # the index of the data source
    filename @1 :Text;               # filename of the data source
}

# Corresponds to a video
struct FrameContainer {
  id @0 :UInt64;                    # global unique id of the frame container
  dataSources @1:List(DataSource);  # list of data sources (videos / images)
  fromTimestamp @2 :Float64;        # unix timestamp of the first frame
  toTimestamp @3 :Float64;          # unix timestamp of the last frame
  frames @4 :List(Frame);           # frames must be sorted by in the order they where recorded.
  camId @5 :UInt16;                 # the cam number
}
