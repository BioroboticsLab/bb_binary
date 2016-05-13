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
namespace po = boost::program_options;

struct GenericCommand {
    bool show_help;
    virtual int parseOptions(const std::vector<std::string> & options) { return 0; };
    virtual int run() const { return 0; }
    virtual std::string help() const { return ""; }
    virtual std::string short_description() const { return ""; }
    virtual ~GenericCommand() = default;
};


enum Format {
    CSV,
    BBB,
};


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

std::vector<std::string> get_lines_from_stdin() {
    std::vector<std::string> lines;
    for (std::string line; std::getline(std::cin, line);) {
        // remove trailing \n
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
        lines.emplace_back(std::move(line));
    }
    return lines;
}


struct ConvertCommand : public GenericCommand {
    Format from;
    Format to;
    std::string output;
    bool use_compression;
    std::vector<std::string> files;

    po::options_description get_option_description() const {
        po::options_description desc("convert options");
        desc.add_options()
                ("help,h", "print help message")
                // ("to", po::value<std::string>(), "convert to this format")    allways convert from CSV to BBB
                ("output,o", po::value<std::string>(), "name of the output file")
                ("files", po::value<std::vector<std::string>>(), "list of files");
        return desc;
    }

    po::positional_options_description get_postional_options_description() {
        po::positional_options_description pos;
        pos.add("files", -1);
        return pos;
    }

    void adopt_compression_based_on_output_name(const std::string & output_fname) {
        fs::path output_as_path(output_fname);
        std::string first_ext = output_as_path.extension().string();
        this->use_compression = first_ext.compare(".lz4") == 0;
        this->output = output_fname;
    }
    int parseOptions(const std::vector<std::string> & options) override {
        po::variables_map vm;
        po::options_description convert_desc = get_option_description();
        auto pos = get_postional_options_description();

        // Parse again...
        po::store(po::command_line_parser(options).options(convert_desc).positional(pos).run(), vm);

        this->show_help = static_cast<bool>(vm.count("help"));
        if (this->show_help) {
            return 0;
        }
        this->to = Format::BBB;
        this->from = Format::CSV;
        if(vm.count("output") != 0) {
            this->adopt_compression_based_on_output_name(vm["output"].as<std::string>());
        }
        if(vm.count("files") == 0) {
            this->files = get_lines_from_stdin();
        } else {
            this->files = vm["files"].as<std::vector<std::string>>();
        }
        return 0;
    }
    int run() const override {
        capnp::MallocMessageBuilder message;
        FrameContainer::Builder container = message.initRoot<FrameContainer>();

        auto frames_union = container.initFrames();
        auto frames = frames_union.initColumnwise(this->files.size());
        for (size_t i = 0; i < this->files.size(); i++) {
            const auto & fname = this->files.at(i);
            auto frame = frames[i];
            frame_from_csv_file(frame, fname);
        }
        writeMessage(message, this->output, this->use_compression);
        return 0;
    }
    std::string help() const override  {
        std::stringstream ss;
        ss << "bbb convert: Converts different detections formats\n";
        ss << get_option_description();
        return ss.str();
    }
    std::string short_description() const override {
        return "  convert    Converts different detections formats";
    }
};

struct HelpCommand : public GenericCommand {
    static const std::vector<std::shared_ptr<GenericCommand>> available_cmds;

    int run() const override {
        std::cout << "bbb: tool for the BeesBook detections format"  << std::endl;
        std::cout << "Available commands" << std::endl;
        for(const auto & cmd : this->available_cmds) {
            std::cout << cmd->short_description() << std::endl;
        }
        return 0;
    }
};

const std::vector<std::shared_ptr<GenericCommand>> HelpCommand::available_cmds = {
        std::make_shared<ConvertCommand>()
};


std::unique_ptr<GenericCommand> ParseOptions(int argc, const char *argv[])
{
    po::options_description global("Global options");
    global.add_options()
            ("command", po::value<std::string>()->default_value("help"), "command to execute")
            ("help", "command to execute")
            ("subargs", po::value<std::vector<std::string>>(), "Arguments for command");

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
        // Collect all the unrecognized options from the first pass. This will include the
        // (positional) command name, so we need to erase that.
        std::vector<std::string> opts = po::collect_unrecognized(parsed.options, po::include_positional);
        opts.erase(opts.begin());
        auto cc = std::make_unique<ConvertCommand>();
        if(vm.count("help")) {
            std::vector<std::string> opt_with_help{"--help"};
            opt_with_help.insert(opt_with_help.end(), opts.begin(), opts.end());
            opts = opt_with_help;
        }
        cc->parseOptions(opts);
        return cc;
    }
    else if (cmd == "help")
    {
        return std::make_unique<HelpCommand>();
    }
    else if (cmd == "chmod")
    {
        // Something similar
    }

    // unrecognised command
    throw po::invalid_option_value(cmd);
}




int main(int argc, const char *argv[]) {
    std::unique_ptr<GenericCommand> cmd = ParseOptions(argc, argv);
    if(cmd->show_help) {
        std::cout << cmd->help();
        return 0;
    } else {
        return cmd->run();
    }
}
