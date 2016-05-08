# include <csv.h>
#include <fstream>
#include <iostream>
#include "bb_binary/bb_binary.h"
#include "src/compress.h"

#include <boost/test/unit_test.hpp>
#include <boost/program_options.hpp>
#include <boost/variant/variant.hpp>
#include <boost/variant/get.hpp>

using namespace bb_binary;

int main_old(int argc, char **argv) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<FrameWrapper>> frames;
    std::vector<std::string> lines;
    std::istream * istream = nullptr;
    for(int i = 0; i < argc; i++) {
        std::cout << argv[i] << std::endl;
    }
    std::ifstream input_file;
    if (argc == 2) {
        input_file.open(argv[1]);
        istream = &input_file;
    } else {
        istream = &std::cin;
    }
    for (std::string line; std::getline(*istream, line);) {
        // remove trailing \n
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        lines.emplace_back(std::move(line));
    }

    for (size_t i = 0; i < lines.size(); i++) {
        const auto & line = lines.at(i);
        auto frame = frame_from_csv_file(builder, line);
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
    f = fopen("frames.bbb", "w");
    fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f);

    f = fopen("frames.bbb.lz4", "w");
    size_t bytes_out;
    lz4_compress_and_write_to_file(reinterpret_cast<char*>(builder.GetBufferPointer()), builder.GetSize(), f, &bytes_out);
    fclose(f);
    std::cout << "bytes compressed: " << bytes_out << std::endl;
    f = fopen("frames.bbb.lz4", "r");

    void * decom_buf;
    size_t decom_size;
    try {
        lz4_decompress_file(f, &decom_buf, &decom_size);
    }
    catch( const std::exception & ex ) {
        std::cerr << ex.what() << std::endl;
    }
    fclose(f);
    if (decom_size != builder.GetSize()) {
        std::cout << "Buffer sizes not equal" << std::endl;
    } else if (memcmp(builder.GetBufferPointer(), decom_buf, decom_size) == 0) {
        std::cout << "Equal" << std::endl;
        FILE * f_decom = fopen("frame_decom.bbb", "w");
        fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), f_decom);
        fclose((f_decom));

    } else {
        std::cout << "Not Equal" << std::endl;
    }
}

struct GenericOptions {
    bool debug_;
};

struct CompressCSVCommand : public GenericOptions {
    std::string pathfile;
};

struct BBBtoCSVCommand : public GenericOptions {
    bool recurse_;
    std::string perms_;
    std::string path_;
};

typedef boost::variant<CompressCSVCommand, BBBtoCSVCommand> Command;

Command ParseOptions(int argc, const char *argv[])
{
    namespace po = boost::program_options;

    po::options_description global("Global options");
    global.add_options()
        ("debug", "Turn on debug output")
        ("command", po::value<std::string>(), "command to execute")
        ("subargs", po::value<std::vector<std::string> >(), "Arguments for command");

    po::positional_options_description pos;
    pos.add("command", 1).
        add("subargs", -1);

    po::variables_map vm;

    po::parsed_options parsed = po::command_line_parser(argc, argv).
        options(global).
        positional(pos).
        allow_unregistered().
        run();

    po::store(parsed, vm);

    std::string cmd = vm["command"].as<std::string>();

    if (cmd == "ls")
    {
        // ls command has the following options:
        po::options_description ls_desc("ls options");
        ls_desc.add_options()
            ("hidden", "Show hidden files")
            ("path", po::value<std::string>(), "Path to list");

        // Collect all the unrecognized options from the first pass. This will include the
        // (positional) command name, so we need to erase that.
        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        opts.erase(opts.begin());

        // Parse again...
        po::store(po::command_line_parser(opts).options(ls_desc).run(), vm);

        LsCommand ls;
        ls.debug_ = vm.count("debug");
        ls.hidden_ = vm.count("hidden");
        ls.path_ = vm["path"].as<std::string>();

        return ls;
    }
    else if (cmd == "chmod")
    {
        // Something similar
    }

    // unrecognised command
    throw po::invalid_option_value(cmd);
}