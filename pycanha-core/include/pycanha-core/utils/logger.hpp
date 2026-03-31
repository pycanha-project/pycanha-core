#pragma once

#include <memory>
#include <ostream>
#include <string_view>

#include <spdlog/common.h>
#include <spdlog/logger.h>

namespace pycanha {

std::shared_ptr<spdlog::logger> get_logger();

std::shared_ptr<spdlog::logger> create_ostream_logger(
    std::string_view name, std::ostream& stream,
    spdlog::level::level_enum level = spdlog::level::info);

void set_logger_level(spdlog::level::level_enum level);

}  // namespace pycanha