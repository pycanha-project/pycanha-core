#include <Eigen/Core>
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/thread_pool.hpp"
#include "pycanha-core/utils/threads.hpp"

#ifdef __linux__
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

// More formulas of one shape than one block holds, so the runs below are
// evaluated in two blocks.
constexpr int node_count = 300;

// cp(T) on [200, 400] K.
LookupTable1D cp_table() {
    return {Eigen::Vector3d(200.0, 300.0, 400.0),
            Eigen::Vector3d(900.0, 950.0, 1100.0)};
}

// Nodes 1..node_count at temperatures spread over [200, 400] K, the
// parameters `a` = 0.3 and `tk` = 250 and the table `cp`.
struct BlockModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    BlockModel() {
        for (int number = 1; number <= node_count; ++number) {
            Node node(number);
            node.set_T(200.0 + std::fmod(61.8 * number, 200.0));
            network->add_node(node);
        }
        parameters->add_parameter("a", 0.3);
        parameters->add_parameter("tk", 250.0);
        parameters->add_table("cp", cp_table());
    }

    [[nodiscard]] double qi(int node) const {
        return network->nodes().get_qi(node);
    }
};

// `pattern` with every "#" replaced by `node` and every "@" by `number`.
std::string expand(const std::string& pattern, int node, double number) {
    std::string result;
    for (const char character : pattern) {
        if (character == '#') {
            result += std::to_string(node);
        } else if (character == '@') {
            result += std::to_string(number);
        } else {
            result += character;
        }
    }
    return result;
}

// Adds QI<i> = pattern for every node, applies them, and returns the largest
// relative difference between a formula's applied value and its own
// evaluation, which runs one formula at a time.
double block_difference(const std::string& pattern) {
    BlockModel model;
    for (int node = 1; node <= node_count; ++node) {
        model.formulas.add_formula("QI" + std::to_string(node),
                                   expand(pattern, node, 0.5 + (1e-3 * node)));
    }
    model.formulas.apply_formulas();
    double largest = 0.0;
    for (int node = 1; node <= node_count; ++node) {
        const double expected =
            model.formulas.find("QI" + std::to_string(node))->get_value();
        const double scale = std::max(std::abs(expected), 1.0);
        largest =
            std::max(largest, std::abs(model.qi(node) - expected) / scale);
    }
    return largest;
}

// `count` nodes at temperatures spread over [200, 400] K, the parameter
// `tk` = 250 and the table `cp`.
struct ThreadedModel {
    std::shared_ptr<ThermalNetwork> network =
        std::make_shared<ThermalNetwork>();
    std::shared_ptr<Parameters> parameters = std::make_shared<Parameters>();
    Formulas formulas{network, parameters};

    explicit ThreadedModel(int count) {
        for (int number = 1; number <= count; ++number) {
            Node node(number);
            node.set_T(200.0 + std::fmod(61.8 * number, 200.0));
            network->add_node(node);
        }
        parameters->add_parameter("tk", 250.0);
        parameters->add_table("cp", cp_table());
    }
};

}  // namespace

TEST_CASE("Runs of one shape give the values of their formulas",
          "[formulas][blocks]") {
    const std::vector<std::string> patterns{
        "@",
        "@ * T#",
        "T# + @ - a",
        "(T# - @) / @",
        "a * T#**2 + sqrt(T#) - exp(-T# / 100) * abs(@ - T#)",
        "max(T#, 300) * min(T#, a * 1000) / atan2(T#, @)",
        "cp(T#) * @ + cp((T# + tk) / 2)",
        "T# * gamma(@ + 1) + log(T#) * sin(@)",
        "@ * gamma(T# / 100)"};
    for (const auto& pattern : patterns) {
        INFO(pattern);
        REQUIRE(block_difference(pattern) < 1e-14);
    }
}

TEST_CASE("A run whose formulas read each other keeps insertion order",
          "[formulas][blocks]") {
    BlockModel model;
    // QI1 = 1, QI2 = QI1 + 1, ...: one shape, each formula reading the one
    // before it.
    model.formulas.add_formula("QI1", "1.0");
    for (int node = 2; node <= node_count; ++node) {
        model.formulas.add_formula("QI" + std::to_string(node),
                                   "QI" + std::to_string(node - 1) + " + 1");
    }
    model.formulas.apply_formulas();
    REQUIRE(model.qi(node_count) == Catch::Approx(node_count));
    REQUIRE(model.qi(node_count / 2) == Catch::Approx(node_count / 2));
}

TEST_CASE("A run may read the targets of an earlier run",
          "[formulas][blocks]") {
    BlockModel model;
    for (int node = 1; node <= node_count; ++node) {
        const auto number = std::to_string(node);
        model.formulas.add_formula("QS" + number, "2 * T" + number);
        model.formulas.add_formula("QI" + number, "QS" + number + " + 1");
    }
    model.formulas.apply_formulas();
    auto& nodes = model.network->nodes();
    REQUIRE(model.qi(7) == Catch::Approx((2.0 * nodes.get_T(7)) + 1.0));
    REQUIRE(model.qi(node_count) ==
            Catch::Approx((2.0 * nodes.get_T(node_count)) + 1.0));
}

TEST_CASE("Formulas that read only parameters follow every change",
          "[formulas][parameters]") {
    BlockModel model;
    for (int node = 1; node <= 20; ++node) {
        model.formulas.add_formula("QI" + std::to_string(node),
                                   std::to_string(node) + " * a");
    }
    model.formulas.apply_formulas();
    REQUIRE(model.qi(20) == Catch::Approx(6.0));

    model.parameters->set_parameter("a", 0.5);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(20) == Catch::Approx(10.0));

    // Nothing changed: the stored values are written again, also over a
    // target written by someone else since.
    model.network->nodes().set_qi(20, -1.0);
    model.parameters->set_parameter("a", 0.5);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(20) == Catch::Approx(10.0));
}

TEST_CASE("Formulas that read only parameters follow table edits",
          "[formulas][parameters]") {
    BlockModel model;
    model.formulas.add_formula("QI1", "2 * cp(tk)");
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(1850.0));

    const std::vector<double> values{1000.0, 1000.0, 1000.0};
    model.parameters->set_table_values("cp", values);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(2000.0));
}

TEST_CASE("Table lookups follow edits of the table's x and y",
          "[formulas][blocks]") {
    BlockModel model;
    for (int node = 1; node <= node_count; ++node) {
        model.formulas.add_formula("QI" + std::to_string(node),
                                   "cp(T" + std::to_string(node) + ")");
    }
    model.formulas.apply_formulas();
    // T7 = 200 + fmod(61.8 * 7, 200) = 232.6 K, on the first segment.
    REQUIRE(model.qi(7) == Catch::Approx(900.0 + (0.5 * 32.6)));

    const std::vector<double> values{1000.0, 2000.0, 3000.0};
    model.parameters->set_table_values("cp", values);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(7) == Catch::Approx(1000.0 + (10.0 * 32.6)));

    const std::vector<double> x{100.0, 300.0, 500.0};
    model.parameters->set_table_x("cp", x);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(7) == Catch::Approx(1000.0 + (5.0 * 132.6)));
    REQUIRE(model.parameters->table("cp").slope(400.0) == Catch::Approx(5.0));
}

TEST_CASE("Formulas evaluated on several threads give the same values",
          "[formulas][blocks][threads]") {
    // Enough formulas for the blocks to be shared among threads.
    constexpr int many = 20000;
    const auto build = [](Formulas& formulas, Parameters& parameters) {
        parameters.add_parameter("a", 0.3);
        parameters.add_table("cp", cp_table());
        for (int node = 1; node <= many; ++node) {
            const auto number = std::to_string(node);
            formulas.add_formula("QI" + number, "a * cp(T" + number + ")");
            formulas.add_formula("QS" + number, "T" + number + " / 2 + a");
        }
    };
    auto network = std::make_shared<ThermalNetwork>();
    for (int number = 1; number <= many; ++number) {
        Node node(number);
        node.set_T(200.0 + std::fmod(61.8 * number, 200.0));
        network->add_node(node);
    }
    auto parameters = std::make_shared<Parameters>();
    Formulas formulas(network, parameters);
    build(formulas, *parameters);

    set_threads(1);
    formulas.apply_formulas();
    auto& nodes = network->nodes();
    std::vector<double> serial;
    for (int node = 1; node <= many; ++node) {
        serial.push_back(nodes.get_qi(node));
        serial.push_back(nodes.get_qs(node));
    }
    // The first apply with more threads warms the caches and the second
    // times the blocks on one thread; from the third, they are shared among
    // threads when that is faster, which it is for this many formulas.
    set_threads(4);
    formulas.apply_formulas();
    formulas.apply_formulas();
    network->nodes().set_qi(1, 0.0);
    network->nodes().set_qs(many, 0.0);
    formulas.apply_formulas();
    set_threads(0);
    for (int node = 1; node <= many; ++node) {
        const auto index = static_cast<std::size_t>(2 * (node - 1));
        REQUIRE(nodes.get_qi(node) == serial[index]);
        REQUIRE(nodes.get_qs(node) == serial[index + 1U]);
    }
}

TEST_CASE("A table that throws stops formulas evaluated on several threads",
          "[formulas][blocks][threads]") {
    constexpr int many = 20000;
    auto network = std::make_shared<ThermalNetwork>();
    for (int number = 1; number <= many; ++number) {
        Node node(number);
        node.set_T(300.0);
        network->add_node(node);
    }
    auto parameters = std::make_shared<Parameters>();
    const LookupTable1D table(
        Eigen::Vector2d(200.0, 400.0), Eigen::Vector2d(1.0, 2.0),
        InterpolationMethod::Linear, ExtrapolationMethod::Throw);
    parameters->add_table("strict", table);
    Formulas formulas(network, parameters);
    for (int node = 1; node <= many; ++node) {
        const auto number = std::to_string(node);
        formulas.add_formula("QI" + number, "strict(T" + number + ")");
    }
    // Two applies inside the range: from the third, the formulas run on
    // several threads (see the test above).
    set_threads(4);
    formulas.apply_formulas();
    formulas.apply_formulas();
    network->nodes().set_T(12345, 500.0);
    REQUIRE_THROWS_AS(formulas.apply_formulas(), std::out_of_range);
    set_threads(0);
}

TEST_CASE("The thread count follows the blocks each apply evaluates",
          "[formulas][blocks][threads]") {
    // One group: a long run that reads only parameters and a run of two
    // blocks that reads temperatures, of cheap formulas. The timed apply
    // replays the long run, so it evaluates only the two blocks, which take
    // less time than waking a thread.
    constexpr int many = 200000;
    constexpr int few = 512;
    ThreadedModel model(many);
    for (int node = 1; node <= many; ++node) {
        model.formulas.add_formula(
            "QS" + std::to_string(node),
            expand("@ * cp(tk)", node, 0.5 + (1e-6 * node)));
    }
    for (int node = 1; node <= few; ++node) {
        model.formulas.add_formula(
            "QI" + std::to_string(node),
            expand("@ * cp(T#)", node, 0.5 + (1e-3 * node)));
    }
    auto& pool = detail::ThreadPool::shared();
    // Measured now, so that the applies below run no measuring jobs.
    REQUIRE(pool.overhead_ns(2).has_value());
    REQUIRE(pool.overhead_ns(4).has_value());

    set_threads(4);
    model.formulas.apply_formulas();
    model.formulas.apply_formulas();
    // Every block of the long run is evaluated again: they are worth
    // sharing although the timed apply was not.
    model.parameters->set_parameter("tk", 260.0);
    const auto jobs = pool.jobs();
    model.formulas.apply_formulas();
    const auto threaded_jobs = pool.jobs() - jobs;
    set_threads(0);
    REQUIRE(threaded_jobs == 1U);
    REQUIRE(model.network->nodes().get_qs(many) ==
            Catch::Approx((0.5 + (1e-6 * many)) * 930.0));
}

#ifdef __linux__
TEST_CASE("A child process evaluates formulas on threads after fork",
          "[formulas][blocks][threads]") {
    // Formulas costly enough (an escaped gamma function) that these are
    // worth sharing among threads.
    constexpr int many = 20000;
    ThreadedModel model(many);
    for (int node = 1; node <= many; ++node) {
        model.formulas.add_formula(
            "QI" + std::to_string(node),
            expand("@ * gamma(T# / 100)", node, 0.5 + (1e-3 * node)));
    }
    // From the third apply the blocks are shared among threads, so the pool
    // has workers when the process is copied.
    set_threads(2);
    model.formulas.apply_formulas();
    model.formulas.apply_formulas();
    model.formulas.apply_formulas();
    set_threads(0);
    auto& nodes = model.network->nodes();
    const double expected = nodes.get_qi(many);
    nodes.set_qi(many, 0.0);

    const pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        // The child reports through its exit code alone: 0 when its apply
        // ran on threads and gave the parent's value. A child that hangs is
        // ended by its alarm.
        alarm(60U);
        int code = 1;
        try {
            set_threads(2);
            model.formulas.apply_formulas();
            const bool threaded = detail::ThreadPool::shared().jobs() == 1U;
            code = (threaded && (nodes.get_qi(many) == expected)) ? 0 : 1;
        } catch (...) {
            code = 2;
        }
        _exit(code);
    }

    int status = -1;
    REQUIRE(waitpid(child, &status, 0) == child);
    // 0 only for a normal exit with code 0; not for a child ended by its
    // alarm.
    REQUIRE(status == 0);
}
#endif

TEST_CASE("A parameter set by a callable formula reaches later formulas",
          "[formulas][parameters]") {
    BlockModel model;
    auto calls = std::make_shared<double>(0.0);
    model.formulas.add_formula("QI2", "10 * a");
    model.formulas.add_callable_formula(
        "QI1", [parameters = model.parameters, calls] {
            *calls += 1.0;
            parameters->set_parameter("a", *calls);
            return *calls;
        });
    model.formulas.add_formula("QI3", "10 * a");
    model.formulas.apply_formulas();
    model.formulas.apply_formulas();
    REQUIRE(model.qi(2) == Catch::Approx(10.0));
    REQUIRE(model.qi(3) == Catch::Approx(20.0));
}

TEST_CASE("Formulas read tables evaluated at the time as it changes",
          "[formulas][parameters]") {
    BlockModel model;
    model.parameters->add_internal_parameter("time", 0.0);
    model.parameters->add_table(
        "heater",
        LookupTable1D(Eigen::Vector2d(0.0, 10.0), Eigen::Vector2d(0.0, 100.0)),
        "time");
    model.formulas.add_formula("QI1", "2 * heater");
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(0.0));

    model.parameters->set_internal_parameter("time", 5.0);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(100.0));
}

TEST_CASE("Inactive formulas leave their targets but tables still follow time",
          "[formulas][active]") {
    BlockModel model;
    model.parameters->add_internal_parameter("time", 0.0);
    model.parameters->add_table(
        "heater",
        LookupTable1D(Eigen::Vector2d(0.0, 10.0), Eigen::Vector2d(0.0, 100.0)),
        "time");
    model.formulas.add_formula("QI1", "2 * heater");
    model.formulas.active = false;
    model.parameters->set_internal_parameter("time", 5.0);
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(0.0));
    REQUIRE(std::get<double>(model.parameters->get_parameter("heater")) ==
            Catch::Approx(50.0));

    model.formulas.active = true;
    model.formulas.apply_formulas();
    REQUIRE(model.qi(1) == Catch::Approx(100.0));
}

TEST_CASE("Time and formulas are updated with the user callbacks off",
          "[formulas][active]") {
    ThermalMathematicalModel model("switches");
    model.add_node(1);
    model.formulas().add_formula("QI1", "2 * time");
    model.c_extern_callback_solver_loop = [](ThermalMathematicalModel* tmm) {
        tmm->nodes().set_qr(1, 1.0);
    };
    model.c_callbacks_active = true;
    model.callbacks_active = false;
    model.time = 3.0;
    model.callback_solver_loop();
    REQUIRE(model.nodes().get_qi(1) == Catch::Approx(6.0));
    REQUIRE(model.nodes().get_qr(1) == Catch::Approx(0.0));

    model.formulas().active = false;
    model.time = 4.0;
    model.callback_solver_loop();
    REQUIRE(model.nodes().get_qi(1) == Catch::Approx(6.0));
    REQUIRE(std::get<double>(model.parameters().get_parameter("time")) ==
            Catch::Approx(4.0));
}
