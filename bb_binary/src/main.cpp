# include <csv.h>
#include <fstream>
#include <iostream>
#include "bb_binary/bb_binary_schema_generated.h"

using namespace bb_binary;

flatbuffers::Offset<FrameColumnwise> read_csv_file(
        flatbuffers::FlatBufferBuilder & builder,
        const std::string & fname) {
    std::ifstream inFile(fname);
    const size_t nb_detections = static_cast<size_t>(std::count(std::istreambuf_iterator<char>(inFile),
                                                          std::istreambuf_iterator<char>(), '\n')) + 1;

    unsigned long * tagIdx;         // unique sequential id of the tag
    unsigned short * candidateIdx;   // sequential id of the candidate per tag
    unsigned short * gridIdx;        // sequential id of the grid/decoding per candidate
    unsigned short * xpos;           // x coordinate of the grid center
    unsigned short * ypos;           // y coordinate of the grid center
    float * xRotation;    // rotation of the grid in x plane
    float * yRotation;    // rotation of the grid in y plane
    float * zRotation;    // rotation of the grid in z plane
    float * lScore;       // roi score
    unsigned short * eScore;       // ellipse score
    float * gScore;       // grid score
    unsigned long * id;             // decoded id

    auto buf_tagIdx = builder.CreateUninitializedVector(nb_detections, &tagIdx);
    auto buf_candidateIdx = builder.CreateUninitializedVector(nb_detections, &candidateIdx);
    auto buf_gridIdx = builder.CreateUninitializedVector(nb_detections, &gridIdx);
    auto buf_xpos = builder.CreateUninitializedVector(nb_detections, &xpos);
    auto buf_ypos = builder.CreateUninitializedVector(nb_detections, &ypos);
    auto buf_xRotation = builder.CreateUninitializedVector(nb_detections, &xRotation);
    auto buf_yRotation = builder.CreateUninitializedVector(nb_detections, &yRotation);
    auto buf_zRotation = builder.CreateUninitializedVector(nb_detections, &zRotation);
    auto buf_lScore = builder.CreateUninitializedVector(nb_detections, &lScore);
    auto buf_eScore = builder.CreateUninitializedVector(nb_detections, &eScore);
    auto buf_gScore = builder.CreateUninitializedVector(nb_detections, &gScore);
    auto buf_id = builder.CreateUninitializedVector(nb_detections, &id);

    io::CSVReader<12> csv_reader(fname);
    size_t index = 0;
    while(csv_reader.read_row(
                *(tagIdx + index),
                *(candidateIdx + index),
                *(gridIdx + index),
                *(xpos + index),
                *(ypos + index),
                *(xRotation + index),
                *(yRotation + index),
                *(zRotation + index),
                *(lScore + index),
                *(eScore + index),
                *(gScore + index),
                *(id + index)
    )) {
        index++;
    };
    assert(index + 1== nb_detections);
    std::cout << nb_detections << std::endl;
    std::cout << index << std::endl;
    return CreateFrameColumnwise(builder, buf_tagIdx, buf_candidateIdx, buf_gridIdx, buf_xpos, buf_ypos, buf_xRotation,
                           buf_yRotation, buf_zRotation, buf_lScore, buf_eScore, buf_gScore, buf_id);
}
int main(){
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<FrameWrapper>> frames;
    std::vector<std::string> lines;
    for (std::string line; std::getline(std::cin, line);) {
        // remove trailing \n
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        lines.emplace_back(std::move(line));
    }

    for (size_t i = 0; i < lines.size(); i++) {
        const auto & line = lines.at(i);
        auto frame = read_csv_file(builder, line);
        frames.push_back(CreateFrameWrapper(builder, Frame_FrameColumnwise, frame.Union()));
    }
    auto buf_frames = builder.CreateVector(frames);
    auto container = FrameContainerBuilder(builder);
    container.add_frames(buf_frames);
    auto container_loc = container.Finish();
    builder.Finish(container_loc);
    builder.GetBufferPointer();
    builder.GetSize();
    FILE * f;
    f = fopen("frames.bbb", "wb");
    fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);
}
