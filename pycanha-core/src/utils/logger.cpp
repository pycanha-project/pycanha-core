#include "pycanha-core/utils/logger.hpp"

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace pycanha {

std::shared_ptr<spdlog::logger> get_logger() {
    auto logger = spdlog::get("pycanha-core");
    if (!logger) {
        logger = spdlog::stdout_color_mt("pycanha-core");
        logger->set_pattern("[%^%l%$] %v");
        logger->set_level(spdlog::level::info);
    }

    return logger;
}

std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    const spdlog::level::level_enum level) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream, true);
    auto logger =
        std::make_shared<spdlog::logger>(std::string{name}, std::move(sink));
    logger->set_pattern("%v");
    logger->set_level(level);

    return logger;
}

void set_logger_level(const spdlog::level::level_enum level) {
    get_logger()->set_level(level);
}

}  // namespace pycanha
