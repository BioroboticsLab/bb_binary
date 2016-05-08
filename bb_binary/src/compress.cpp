#include "compress.h"

#include <iostream>
#include <lz4frame.h>
#include <cstdio>
#include <vector>

#define BUF_SIZE (16*1024)
#define LZ4_HEADER_SIZE 19
#define LZ4_FOOTER_SIZE 4

LZ4F_preferences_t get_lz4_preferences(const size_t content_size) {
    return {
        { LZ4F_max256KB, LZ4F_blockLinked, LZ4F_contentChecksumEnabled, LZ4F_frame, content_size, { 0, 0 } },
        0,   /* compression level */
        0,   /* autoflush */
        { 0, 0, 0, 0 },  /* reserved, must be set to 0 */
    };
}

int lz4_compress_and_write_to_file(
		const char * in, const size_t in_size,
		FILE * out, size_t * bytes_written)  {
	// adapted from  https://github.com/Cyan4973/lz4/blob/master/examples/frameCompress.c
	LZ4F_errorCode_t r;
	LZ4F_compressionContext_t ctx;
    LZ4F_preferences_t lz4_preferences = get_lz4_preferences(in_size);
	char *src, *buf = NULL;
	size_t size, n, k, count_in = 0, count_out, offset = 0, frame_size;

	r = LZ4F_createCompressionContext(&ctx, LZ4F_VERSION);
	if (LZ4F_isError(r)) {
		printf("Failed to create context: error %zu", r);
		return 1;
	}
	r = 1;

	frame_size = LZ4F_compressBound(BUF_SIZE, &lz4_preferences);
	size =  frame_size + LZ4_HEADER_SIZE + LZ4_FOOTER_SIZE;
	buf = reinterpret_cast<char*>(malloc(size));
	if (!buf) {
		printf("Not enough memory");
		goto cleanup;
	}

	n = offset = count_out = LZ4F_compressBegin(ctx, buf, size, &lz4_preferences);
	if (LZ4F_isError(n)) {
		printf("Failed to start compression: error %zu", n);
		goto cleanup;
	}

	printf("Buffer size is %zu bytes, header size %zu bytes\n", size, n);

	for (;;) {
		if(count_in + BUF_SIZE < in_size) {
			k = BUF_SIZE;
		} else {
			k = in_size - count_in;
		}
		src = const_cast<char*>(in + count_in);
		if (k == 0)
			break;
		count_in += k;

		n = LZ4F_compressUpdate(ctx, buf + offset, size - offset, src, k, NULL);
		if (LZ4F_isError(n)) {
			printf("Compression failed: error %zu", n);
			goto cleanup;
		}

		offset += n;
		count_out += n;
		if (size - offset < frame_size + LZ4_FOOTER_SIZE) {
			printf("Writing %zu bytes\n", offset);

			k = fwrite(buf, 1, offset, out);
			if (k < offset) {
				if (ferror(out))
					printf("Write failed");
				else
					printf("Short write");
				goto cleanup;
			}

			offset = 0;
		}
	}

	n = LZ4F_compressEnd(ctx, buf + offset, size - offset, NULL);
	if (LZ4F_isError(n)) {
		printf("Failed to end compression: error %zu", n);
		goto cleanup;
	}

	offset += n;
	count_out += n;
	printf("Writing %zu bytes\n", offset);

	k = fwrite(buf, 1, offset, out);
	if (k < offset) {
		if (ferror(out))
			printf("Write failed");
		else
			printf("Short write");
		goto cleanup;
	}

	*bytes_written = count_out;
	r = 0;
 cleanup:
	if (ctx)
		LZ4F_freeCompressionContext(ctx);
	free(buf);
	return r;
}

int lz4_decompress_file(FILE *in_file, void **buf_out, size_t * buf_out_size) {
    LZ4F_decompressionContext_t dctx;
    LZ4F_frameInfo_t frameInfo;
    LZ4F_decompressOptions_t decOpt = {1, {0, 0, 0}};
    LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);

	std::vector<char> header_buf(LZ4_HEADER_SIZE);
    fseek(in_file, 0, SEEK_SET);
    fread(header_buf.data(), 1, LZ4_HEADER_SIZE, in_file);
    size_t header_size = LZ4_HEADER_SIZE;
    size_t n = LZ4F_getFrameInfo(dctx, &frameInfo, header_buf.data(), &header_size);
    if (LZ4F_isError(n)) {
        throw std::runtime_error("Failed to read frame info");
    }
    fseek(in_file, 0, SEEK_END); // seek to end of file
    size_t file_size = static_cast<size_t>(ftell(in_file)); // get current file pointer
    fseek(in_file, 0, SEEK_SET);

    *buf_out_size = frameInfo.contentSize + 1000;
    *buf_out = calloc(*buf_out_size, 1);

	std::vector<char> file_buf(file_size);
    fread(file_buf.data(), 1, file_size, in_file);
    size_t source_size = file_size - header_size;

	// Debug Output
	std::cout << "Content Size: " << frameInfo.contentSize << std::endl;
	std::cout << "Header Size: " << header_size << std::endl;
	std::cout << "File Size: " << file_size << std::endl;
    std::cout << "Source size : " << source_size << std::endl;
	std::cout << "Bufout: " << buf_out << std::endl;
	std::cout << "Bufout size: " << *buf_out_size << std::endl;
	std::cout << "Filebuf: " << file_buf.data() + header_size << std::endl;
    n = LZ4F_decompress(dctx, *buf_out, buf_out_size,
                       file_buf.data() + header_size, &source_size, nullptr);
    if (LZ4F_isError(n)) {
        std::cout << LZ4F_getErrorName(n) << std::endl;
        throw std::runtime_error("Decompression Failed");
    }
	LZ4F_freeDecompressionContext(dctx);
    return 0;
}




