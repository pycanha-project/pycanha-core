#pragma once

#include <string>

namespace pycanha {

class ThermalMathematicalModel;

class ESATANReader {
  public:
    explicit ESATANReader(ThermalMathematicalModel& model);

    void read_tmd(const std::string& filepath);

    bool verbose{false};

  private:
    ThermalMathematicalModel& _model;
};

}  // namespace pycanha
