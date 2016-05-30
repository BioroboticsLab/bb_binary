

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "bb_binary.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>

using namespace bb_binary;

TEST_CASE( "csv files can be converted to bbb (BeesBook Binary) and back", "[]" ) {
     ::capnp::MallocMessageBuilder message;

    Frame::Builder frame = message.initRoot<Frame>();
    const auto & csv_fname = "data/Cam_0_20150920000000_185682.csv";
    frame_from_csv_file(frame, csv_fname);

    const auto frame_reader = frame.asReader();

    std::string csv_deserialized = frame_to_csv(frame_reader);
    std::ofstream f("data/deserialized.csv");
    f << csv_deserialized;

    std::ifstream t(csv_fname);
    std::string csv_original((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    REQUIRE(std::equal(csv_original.begin(), csv_original.end(),
                       csv_deserialized.begin(), csv_deserialized.end()));

}
