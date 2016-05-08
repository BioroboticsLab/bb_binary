#include "bb_binary_schema_generated.h"

namespace bb_binary {
    flatbuffers::Offset<FrameColumnwise> frame_from_csv_file(
            flatbuffers::FlatBufferBuilder & builder,
            const std::string & fname);

    std::string frame_to_csv(const FrameColumnwise * frame);
}
