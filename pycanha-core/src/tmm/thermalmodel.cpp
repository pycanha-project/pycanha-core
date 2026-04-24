#include "pycanha-core/tmm/thermalmodel.hpp"

#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/callback_registry.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

ThermalModel::ThermalModel(const std::string& model_name)
    : _name(model_name),
      _tmm(std::make_shared<ThermalMathematicalModel>(model_name)),
      _gmm(std::make_shared<gmm::GeometryModel>(model_name)),
      _parameters(_tmm->parameters_ptr()),
      _formulas(_tmm->formulas_ptr()),
      _thermal_data(_tmm->thermal_data_ptr()),
      _solvers(std::make_shared<SolverRegistry>(_tmm)) {
    _tmm->associate_solvers(*_solvers);
    _callbacks = std::make_shared<CallbackRegistry>(*this);
}

ThermalModel::ThermalModel(std::string model_name,
                           std::shared_ptr<ThermalMathematicalModel> tmm,
                           std::shared_ptr<gmm::GeometryModel> gmm,
                           std::shared_ptr<Parameters> parameters,
                           std::shared_ptr<Formulas> formulas,
                           std::shared_ptr<ThermalData> thermal_data,
                           std::shared_ptr<SolverRegistry> solvers)
    : _name(std::move(model_name)),
      _tmm(tmm != nullptr ? std::move(tmm)
                          : std::make_shared<ThermalMathematicalModel>(_name)),
      _gmm(gmm != nullptr ? std::move(gmm)
                          : std::make_shared<gmm::GeometryModel>(_name)),
      _parameters(parameters != nullptr ? std::move(parameters)
                                        : _tmm->parameters_ptr()),
      _formulas(formulas != nullptr ? std::move(formulas)
                                    : _tmm->formulas_ptr()),
      _thermal_data(thermal_data != nullptr ? std::move(thermal_data)
                                            : _tmm->thermal_data_ptr()),
      _solvers(solvers != nullptr ? std::move(solvers)
                                  : std::make_shared<SolverRegistry>(_tmm)) {
    _tmm->associate_solvers(*_solvers);
    _callbacks = std::make_shared<CallbackRegistry>(*this);
}

ThermalMathematicalModel& ThermalModel::tmm() noexcept { return *_tmm; }

const ThermalMathematicalModel& ThermalModel::tmm() const noexcept {
    return *_tmm;
}

gmm::GeometryModel& ThermalModel::gmm() noexcept { return *_gmm; }

const gmm::GeometryModel& ThermalModel::gmm() const noexcept { return *_gmm; }

Parameters& ThermalModel::parameters() noexcept { return *_parameters; }

const Parameters& ThermalModel::parameters() const noexcept {
    return *_parameters;
}

Formulas& ThermalModel::formulas() noexcept { return *_formulas; }

const Formulas& ThermalModel::formulas() const noexcept { return *_formulas; }

ThermalData& ThermalModel::thermal_data() noexcept { return *_thermal_data; }

const ThermalData& ThermalModel::thermal_data() const noexcept {
    return *_thermal_data;
}

SolverRegistry& ThermalModel::solvers() noexcept { return *_solvers; }

const SolverRegistry& ThermalModel::solvers() const noexcept {
    return *_solvers;
}

CallbackRegistry& ThermalModel::callbacks() noexcept { return *_callbacks; }

const CallbackRegistry& ThermalModel::callbacks() const noexcept {
    return *_callbacks;
}

}  // namespace pycanha
