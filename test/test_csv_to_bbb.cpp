

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "bb_binary.h"
#include "bb_binary_schema_generated.h"

using namespace bb_binary;

TEST_CASE( "csv files can be converted to bbb (BeesBook Binary) and back", "[]" ) {
    flatbuffers::FlatBufferBuilder builder;
    const auto & csv_fname = "data/Cam_0_20150920000000_185682.csv";
    auto frame = frame_from_csv_file(builder, csv_fname);
    builder.Finish(frame);
    auto flatbuf_pointer = builder.GetBufferPointer();
    auto flatbuf_size = builder.GetSize();

    auto flatbuf = builder.ReleaseBufferPointer();
    flatbuffers::Verifier verifier(flatbuf_pointer, flatbuf_size);

    // First, verify the buffers integrity (optional)
    FrameColumnwise const * loaded_frame = flatbuffers::GetRoot<FrameColumnwise>(
            reinterpret_cast<void*>(flatbuf_pointer));
    REQUIRE(loaded_frame->Verify(verifier));

    std::string csv_deserialized = frame_to_csv(loaded_frame);
    std::ofstream f("data/deserialized.csv");
    f << csv_deserialized;

    std::ifstream t(csv_fname);
    std::string csv_original((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    REQUIRE(std::equal(csv_original.begin(), csv_original.end(),
                       csv_deserialized.begin(), csv_deserialized.end()));

}
