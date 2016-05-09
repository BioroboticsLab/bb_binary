# include <csv.h>
#include <fstream>
#include <iostream>
#include "bb_binary/bb_binary.h"
#include "compress.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <capnp/serialize-packed.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/variant/variant.hpp>
#include <boost/variant/get.hpp>
#include <fcntl.h>

using namespace bb_binary;
namespace fs = boost::filesystem;

struct GenericOptions {
};
enum Format {
    CSV,
    BBB,
};

struct ConvertCommand : public GenericOptions {
    Format from;
    Format to;
    std::string output;
    bool use_compression;
    std::vector<std::string> files;

    void adopt_compression_based_on_output_name(const std::string & output_fname) {
        fs::path output_as_path(output_fname);
        std::string first_ext = output_as_path.extension().string();
        this->use_compression = first_ext.compare(".lz4") == 0;
        this->output = output_fname;
    }
};


using Command = boost::variant<ConvertCommand>;

std::vector<std::string> get_lines_from_stdin() {
    std::vector<std::string> lines;
    for (std::string line; std::getline(std::cin, line);) {
        // remove trailing \n
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        lines.emplace_back(std::move(line));
    }
    return lines;
}

Command ParseOptions(int argc, const char *argv[])
{
    namespace po = boost::program_options;

    po::options_description global("Global options");
    global.add_options()
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

    if (cmd == "convert")
    {
        // ls command has the following options:
        po::options_description convert_desc("convert options");
        convert_desc.add_options()
                ("to", po::value<std::string>(), "convert to this format")
                ("output,o", po::value<std::string>(), "name of the output file")
                ("files", po::value<std::vector<std::string>>(), "list of files");

        po::positional_options_description pos;
        pos.add("files", -1);

        // Collect all the unrecognized options from the first pass. This will include the
        // (positional) command name, so we need to erase that.
        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        opts.erase(opts.begin());

        // Parse again...
        po::store(po::command_line_parser(opts).options(convert_desc).positional(pos).run(), vm);
        ConvertCommand cc;
        cc.to = Format::BBB;
        cc.from = Format::CSV;
        cc.adopt_compression_based_on_output_name(vm["output"].as<std::string>());
        if(vm.count("files") == 0) {
            cc.files = get_lines_from_stdin();
        } else {
            cc.files = vm["files"].as<std::vector<std::string>>();
        }
        return cc;
    }
    else if (cmd == "chmod")
    {
        // Something similar
    }

    // unrecognised command
    throw po::invalid_option_value(cmd);
}


void writeMessageUncompressed(capnp::MessageBuilder & message,
                 const std::string & filename) {
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 644);
    capnp::writeMessageToFd(fd, message);
}

void writeMessageCompressed(capnp::MessageBuilder & message,
                             const std::string & filename) {
    size_t message_size = capnp::computeSerializedSizeInWords(message);
    auto arr = kj::heapArray<kj::byte>(8*message_size);

    kj::ArrayOutputStream os(arr.asPtr());
    capnp::writeMessage(os, message);
    FILE * f_compressed = fopen(filename.c_str(), "w");
    size_t bytes_out;
    lz4_compress_and_write_to_file(
            os.getArray().asChars().begin(), os.getArray().size(), f_compressed, &bytes_out);
    fclose(f_compressed);
}

void writeMessage(capnp::MessageBuilder & message,
                 const std::string & filname,
                 const bool use_compression) {
    if (use_compression) {
        std::string full_filename = filname;
        writeMessageCompressed(message, full_filename);
    } else {
        std::string full_filename = filname;
        writeMessageUncompressed(message, full_filename);
    }
}

int convert(const ConvertCommand & cmd) {
    capnp::MallocMessageBuilder message;
    FrameContainer::Builder container = message.initRoot<FrameContainer>();

    auto frames_union = container.initFrames();
    auto frames = frames_union.initColumnwise(cmd.files.size());
    for (size_t i = 0; i < cmd.files.size(); i++) {
        const auto & fname = cmd.files.at(i);
        auto frame = frames[i];
        frame_from_csv_file(frame, fname);
    }
    writeMessage(message, cmd.output, cmd.use_compression);
    return 0;
}

int main(int argc, const char *argv[]) {
    Command cmd = ParseOptions(argc, argv);
    convert(boost::get<ConvertCommand>(cmd));
}
