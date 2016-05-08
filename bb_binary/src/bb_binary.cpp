
#include <fstream>
#include <iostream>
#include "bb_binary/bb_binary_schema_generated.h"
#include <csv.h>
#include <boost/filesystem.hpp>
#include <sstream>

namespace bb_binary {

namespace fs = boost::filesystem;

int add(int a, int b) {
    return a + b;
}

unsigned int parse_id(const std::string & s) {
    unsigned int id = 0;
    for(size_t i = 0; i < s.size(); i++) {
        id = id << 1;
        if (s[i] == '1') {
            id += 1;
        }
    }
    return id;
}
std::string id_to_string(unsigned int id, size_t nb_bits) {
    std::stringstream ss;
    for(size_t i = 0; i < nb_bits; i++) {
        ss <<  id % 2;
        id = id >> 1;
    }
    std::string id_str = ss.str();
    std::reverse(id_str.begin(), id_str.end());
    return id_str;
}
flatbuffers::Offset<FrameColumnwise> frame_from_csv_file(
        flatbuffers::FlatBufferBuilder & builder,
        const std::string & fname) {
    std::ifstream inFile(fname);
    std::string fname_without_ext(fs::path(fname).stem().string());

    // counts the number of lines
    const size_t nb_detections = static_cast<size_t>(
            std::count(std::istreambuf_iterator<char>(inFile),
                       std::istreambuf_iterator<char>(), '\n'));

    std::vector<unsigned short> tagIdx(nb_detections);          // unique sequential id of the tag
    std::vector<unsigned short> candidateIdx(nb_detections);   // sequential id of the candidate per tag
    std::vector<unsigned short> gridIdx(nb_detections);        // sequential id of the grid/decoding per candidate
    std::vector<unsigned short> xpos(nb_detections);           // x coordinate of the grid center
    std::vector<unsigned short> ypos(nb_detections);           // y coordinate of the grid center
    std::vector<float> xRotation(nb_detections);               // rotation of the grid in x plane
    std::vector<float> yRotation(nb_detections);               // rotation of the grid in y plane
    std::vector<float> zRotation(nb_detections);               // rotation of the grid in z plane
    std::vector<float> lScore(nb_detections);                  // roi s core
    std::vector<unsigned short> eScore(nb_detections);         // ellipse score
    std::vector<float> gScore(nb_detections);                  // grid score
    std::vector<unsigned int> id(nb_detections);              // decoded id

    std::string id_temporary;
    io::CSVReader<12> csv_reader(fname);
    size_t index = 0;
    while(csv_reader.read_row(
            *(tagIdx.begin() + index),
            *(candidateIdx.begin() + index),
            *(gridIdx.begin() + index),
            *(xpos.begin() + index),
            *(ypos.begin() + index),
            *(xRotation.begin() + index),
            *(yRotation.begin() + index),
            *(zRotation.begin() + index),
            *(lScore.begin() + index),
            *(eScore.begin() + index),
            *(gScore.begin() + index),
            id_temporary
    )) {
        id.at(index) = parse_id(id_temporary);
        index++;
    };

    assert(index == nb_detections);
    auto off_fname = builder.CreateString(fname_without_ext);
    auto off_tagIdx = builder.CreateVector(tagIdx);
    auto off_candidateIdx  = builder.CreateVector(candidateIdx);
    auto off_gridIdx = builder.CreateVector(gridIdx);
    auto off_xpos = builder.CreateVector(xpos);
    auto off_ypos = builder.CreateVector(ypos);
    auto off_xRotation = builder.CreateVector(xRotation);
    auto off_yRotation = builder.CreateVector(yRotation);
    auto off_zRotation = builder.CreateVector(zRotation);
    auto off_lScore = builder.CreateVector(lScore);
    auto off_eScore = builder.CreateVector(eScore);
    auto off_gScore = builder.CreateVector(gScore);
    auto off_id = builder.CreateVector(id);

    FrameColumnwiseBuilder frame(builder);
    frame.add_image_name(off_fname);
    frame.add_tagIdx(off_tagIdx);
    frame.add_candidateIdx(off_candidateIdx);
    frame.add_gridIdx(off_gridIdx);
    frame.add_xpos(off_xpos);
    frame.add_ypos(off_ypos);
    frame.add_xRotation(off_xRotation);
    frame.add_yRotation(off_yRotation);
    frame.add_zRotation(off_zRotation);
    frame.add_lScore(off_lScore);
    frame.add_eScore(off_eScore);
    frame.add_gScore(off_gScore);
    frame.add_id(off_id);
    return frame.Finish();
}

std::string frame_to_csv(const FrameColumnwise * frame) {
    std::stringstream ss;
    for(size_t i = 0; i < frame->candidateIdx()->Length(); i++) {
        ss <<  frame->tagIdx()->Get(i) << ","
           << frame->candidateIdx()->Get(i) << ","
           << frame->gridIdx()->Get(i) << ","
           << frame->xpos()->Get(i) << ","
           << frame->ypos()->Get(i) << ","
           << frame->xRotation()->Get(i) << ","
           << frame->yRotation()->Get(i) << ","
           << frame->zRotation()->Get(i) << ","
           << frame->lScore()->Get(i) << ","
           << frame->eScore()->Get(i) << ","
           << frame->gScore()->Get(i) << ","
           << id_to_string(frame->id()->Get(i), 12) << "\n";
    }
    return ss.str();
}
}
