
#include <fstream>
#include <iostream>
#include <csv.h>
#include <boost/filesystem.hpp>
#include <sstream>

#include "../bb_binary_schema.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <iostream>

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
void frame_from_csv_file(
        FrameColumnwise::Builder & frame,
        const std::string & fname) {
    std::ifstream inFile(fname);
    std::string fname_without_ext(fs::path(fname).stem().string());
    // counts the number of lines
    const size_t nb_detections = static_cast<size_t>(
            std::count(std::istreambuf_iterator<char>(inFile),
                       std::istreambuf_iterator<char>(), '\n'));

    auto tagIdx = frame.initTagIdx(nb_detections);       // unique sequential id of the tag
    auto candidateIdx = frame.initCandidateIdx(nb_detections); // sequential id of the candidate per tag
    auto gridIdx = frame.initGridIdx(nb_detections);           // sequential id of the grid/decoding per candidate
    auto xpos = frame.initXpos(nb_detections);                 // x coordinate of the grid center
    auto ypos =  frame.initYpos(nb_detections);                // y coordinate of the grid center
    auto xRotation = frame.initXRotation(nb_detections);       // rotation of the grid in x plane
    auto yRotation = frame.initYRotation(nb_detections);       // rotation of the grid in y plane
    auto zRotation = frame.initZRotation(nb_detections);       // rotation of the grid in z plane
    auto lScore = frame.initLScore(nb_detections);             // roi s core
    auto eScore = frame.initEScore(nb_detections);             // ellipse score
    auto gScore = frame.initGScore(nb_detections);             // grid score
    auto id = frame.initId(nb_detections);                     // decoded id

    io::CSVReader<12> csv_reader(fname);
    size_t index = 0;

    unsigned short tagIdx_tmp, candidateIdx_tmp, gridIdx_tmp, xpos_tmp, ypos_tmp;
    float xRotation_tmp, yRotation_tmp, zRotation_tmp, lScore_tmp;
    unsigned short eScore_tmp;
    float gScore_tmp;
    std::string id_tmp;

    while(csv_reader.read_row(
                tagIdx_tmp, candidateIdx_tmp, gridIdx_tmp, xpos_tmp, ypos_tmp,
                xRotation_tmp, yRotation_tmp, zRotation_tmp, lScore_tmp, eScore_tmp,
                gScore_tmp, id_tmp
    )) {

        tagIdx.set(index, tagIdx_tmp);
        candidateIdx.set(index, candidateIdx_tmp);
        gridIdx.set(index, gridIdx_tmp);
        xpos.set(index, xpos_tmp);
        ypos.set(index, ypos_tmp);
        xRotation.set(index, xRotation_tmp);
        yRotation.set(index, yRotation_tmp);
        zRotation.set(index, zRotation_tmp);
        lScore.set(index, lScore_tmp);
        eScore.set(index, eScore_tmp);
        gScore.set(index, gScore_tmp);
        id.set(index, parse_id(id_tmp));
        index++;
    };
    assert(index == nb_detections);
}


std::string frame_to_csv(const FrameColumnwise::Reader & frame) {
    std::stringstream ss;
    for(size_t i = 0; i < frame.getCandidateIdx().size(); i++) {
        std::cout << i << std::endl;
        ss << frame.getTagIdx()[i] << ","
           << frame.getCandidateIdx()[i] << ","
           << frame.getGridIdx()[i] << ","
           << frame.getXpos()[i] << ","
           << frame.getYpos()[i] << ","
           << frame.getXRotation()[i] << ","
           << frame.getYRotation()[i] << ","
           << frame.getZRotation()[i] << ","
           << frame.getLScore()[i] << ","
           << frame.getEScore()[i] << ","
           << frame.getGScore()[i] << ","
           << id_to_string(frame.getId()[i], 12) << "\n";
    }
    return ss.str();
}
}
