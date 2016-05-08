#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdint>

int lz4_compress_and_write_to_file(const char * buf_in, const size_t buf_in_size,
                                   FILE * out_file, size_t * size_out);

int lz4_decompress_file(FILE *in_file, void **buf_out, size_t * buf_out_size);

