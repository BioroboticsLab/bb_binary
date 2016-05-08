#include "bb_binary_schema.capnp.h"

namespace bb_binary {

    void frame_from_csv_file(
            FrameColumnwise::Builder & frame,
            const std::string & fname);

    std::string frame_to_csv(const FrameColumnwise::Reader & frame);
}
