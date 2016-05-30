
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
        Frame::Builder & frame,
        const std::string & fname) {
    std::ifstream inFile(fname);
    std::string fname_without_ext(fs::path(fname).stem().string());
    // counts the number of lines
    const size_t nb_detections = static_cast<size_t>(
            std::count(std::istreambuf_iterator<char>(inFile),
                       std::istreambuf_iterator<char>(), '\n'));

    auto detections_union = frame.initDetections();
    auto detections = detections_union.initDetectionsCVP(nb_detections);

    io::CSVReader<12> csv_reader(fname);
    size_t index = 0;

    unsigned short tagIdx, candidateIdx, gridIdx, xpos, ypos;
    float xRotation, yRotation, zRotation, lScore;
    unsigned short eScore;
    float gScore;
    std::string id;

    while(csv_reader.read_row(
                tagIdx, candidateIdx, gridIdx, xpos, ypos,
                xRotation, yRotation, zRotation, lScore, eScore,
                gScore, id
    )) {
        DetectionCVP::Builder detection = detections[index];
        detection.setTagIdx(tagIdx);
        detection.setCandidateIdx(candidateIdx);
        detection.setGridIdx(gridIdx);
        detection.setXpos(xpos);
        detection.setYpos(ypos);
        detection.setXRotation(xRotation);
        detection.setYRotation(yRotation);
        detection.setZRotation(zRotation);
        detection.setLScore(lScore);
        detection.setEScore(eScore);
        detection.setGScore(gScore);
        detection.setDecodedId(parse_id(id));
        index++;
    };
    assert(index == nb_detections);
}

std::string cv_frame_to_csv(const capnp::List<DetectionCVP>::Reader & detections) {
    std::stringstream ss;
    for(size_t i = 0; i < detections.size(); i++) {
        DetectionCVP::Reader d = detections[i];
        ss << d.getTagIdx() << ","
        << d.getCandidateIdx() << ","
        << d.getGridIdx() << ","
        << d.getXpos() << ","
        << d.getYpos() << ","
        << d.getXRotation() << ","
        << d.getYRotation() << ","
        << d.getZRotation() << ","
        << d.getLScore() << ","
        << d.getEScore() << ","
        << d.getGScore() << ","
        << id_to_string(d.getDecodedId(), 12) << "\n";
    }
    return ss.str();
}

std::string frame_to_csv(const Frame::Reader & frame) {
    auto detection_union = frame.getDetections();
    if (detection_union.hasDetectionsCVP()) {
        return cv_frame_to_csv(detection_union.getDetectionsCVP());
    } else if ( detection_union.hasDetectionsDP()) {
        throw  "CSV export is currently not implemented for the deeppipeline format";
    } else {
        throw  "Unknown union format";
    }

}
}
