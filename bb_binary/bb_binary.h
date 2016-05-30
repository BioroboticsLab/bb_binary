#include "bb_binary_schema.capnp.h"

namespace bb_binary {

    void frame_from_csv_file(
            Frame::Builder & frame,
            const std::string & fname);

    std::string frame_to_csv(const Frame::Reader & frame);
}
