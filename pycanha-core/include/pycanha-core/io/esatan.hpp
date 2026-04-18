#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pycanha-core/thermaldata/data_model.hpp"

namespace pycanha {

class ThermalData;
class ThermalMathematicalModel;

std::vector<Index> read_tmd_transient(
    const std::string& filepath, ThermalData& thermal_data,
    const std::string& model_name, bool overwrite = false,
    const std::vector<DataModelAttribute>& attributes = {
        DataModelAttribute::T, DataModelAttribute::C, DataModelAttribute::QA,
        DataModelAttribute::QE, DataModelAttribute::QI, DataModelAttribute::QR,
        DataModelAttribute::QS});

class ESATANReader {
  public:
    explicit ESATANReader(ThermalMathematicalModel& model);

    void read_tmd(const std::string& filepath);

    bool verbose{false};

  private:
    ThermalMathematicalModel& _model;
};

}  // namespace pycanha
