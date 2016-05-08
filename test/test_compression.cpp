
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "src/compress.h"

TEST_CASE( "data can be compressed and decompressed", "[]" ) {
    // This test fails
    FILE * f;
    f = fopen("compression_tmp.bin", "w");
    std::string data = "Hello World! My name is Ben";
    size_t bytes_written;
    lz4_compress_and_write_to_file(data.data(), data.size(), f, &bytes_written);
    fclose(f);
    f = fopen("compression_tmp.bin", "r");

    void * buf;
    size_t buf_length;
    lz4_decompress_file(f, &buf, &buf_length);
    std::string decompressed_data(reinterpret_cast<char*>(buf), buf_length);
    REQUIRE(decompressed_data == data);
    free(buf);
}
