#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pycanha {

class ThermalData;
class ThermalMathematicalModel;

enum class TMDNodeAttribute : std::uint8_t {
    T = 0,
    C = 1,
    QA = 2,
    QE = 3,
    QI = 4,
    QR = 5,
    QS = 6,
    A = 7,
    APH = 8,
    EPS = 9,
    FX = 13,
    FY = 14,
    FZ = 15,
};

std::vector<int> read_tmd_transient(
    const std::string& filepath, ThermalData& thermal_data,
    const std::string& table_prefix, bool overwrite = false,
    const std::vector<TMDNodeAttribute>& attributes = {
        TMDNodeAttribute::T, TMDNodeAttribute::C, TMDNodeAttribute::QA,
        TMDNodeAttribute::QE, TMDNodeAttribute::QI, TMDNodeAttribute::QR,
        TMDNodeAttribute::QS});

class ESATANReader {
  public:
    explicit ESATANReader(ThermalMathematicalModel& model);

    void read_tmd(const std::string& filepath);

    bool verbose{false};

  private:
    ThermalMathematicalModel& _model;
};

}  // namespace pycanha
