/// @file pybind_module.cpp
/// @brief pybind11 bindings exposing the C++ pricing + risk engine to Python.
///
/// Exposes:
///   - Pricing: price_option, solve_implied_vol, price_batch, solve_iv_batch
///   - Market Making: as_quote, lt_quote
///   - Hedging: leland_number, crb_hedge_action, vega_neutralize
///   - Dispersion: dirty_correlation, zscore_signal
///   - Execution: round_to_lots, confidence_quotes

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "pricing/black_scholes.hpp"
#include "pricing/implied_vol.hpp"
#include "market_making/avellaneda_stoikov.hpp"
#include "market_making/lucic_tse.hpp"
#include "hedging/leland.hpp"
#include "hedging/central_risk_book.hpp"
#include "hedging/vega_neutralizer.hpp"
#include "dispersion/dirty_correlation.hpp"
#include "dispersion/zscore_signal.hpp"
#include "execution/position_sizer.hpp"
#include "execution/confidence_quoter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(davinci_py, m) {
    m.doc() = "Da Vinci Volatility Project — C++ core engine bindings";

    // ═════════════════════════════════════════════════════════════════════
    // PRICING MODULE
    // ═════════════════════════════════════════════════════════════════════

    auto pricing = m.def_submodule("pricing", "Black-Scholes pricing and IV solver");

    py::class_<davinci::pricing::GreeksResult>(pricing, "GreeksResult")
        .def_readonly("price", &davinci::pricing::GreeksResult::price)
        .def_readonly("delta", &davinci::pricing::GreeksResult::delta)
        .def_readonly("gamma", &davinci::pricing::GreeksResult::gamma)
        .def_readonly("vega", &davinci::pricing::GreeksResult::vega)
        .def_readonly("theta", &davinci::pricing::GreeksResult::theta)
        .def_readonly("rho", &davinci::pricing::GreeksResult::rho)
        .def_readonly("vanna", &davinci::pricing::GreeksResult::vanna)
        .def_readonly("volga", &davinci::pricing::GreeksResult::volga)
        .def("vega_1pct", &davinci::pricing::GreeksResult::vega_1pct)
        .def("theta_1day", &davinci::pricing::GreeksResult::theta_1day,
             py::arg("trading_days") = 252)
        .def("__repr__", [](const davinci::pricing::GreeksResult& g) {
            return "<GreeksResult price=" + std::to_string(g.price)
                 + " delta=" + std::to_string(g.delta)
                 + " gamma=" + std::to_string(g.gamma)
                 + " vega=" + std::to_string(g.vega) + ">";
        });

    pricing.def("price_option", &davinci::pricing::price_and_greeks,
        py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
        py::arg("sigma"), py::arg("q") = 0.0, py::arg("is_call") = true,
        "Price an option and compute all Greeks");

    pricing.def("bs_call_price", &davinci::pricing::bs_call_price,
        py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
        py::arg("sigma"), py::arg("q") = 0.0);

    pricing.def("bs_put_price", &davinci::pricing::bs_put_price,
        py::arg("S"), py::arg("K"), py::arg("r"), py::arg("T"),
        py::arg("sigma"), py::arg("q") = 0.0);

    // IV solver
    py::enum_<davinci::pricing::IVMethod>(pricing, "IVMethod")
        .value("CORRADO_MILLER", davinci::pricing::IVMethod::CORRADO_MILLER)
        .value("NEWTON_RAPHSON", davinci::pricing::IVMethod::NEWTON_RAPHSON)
        .value("BRENT", davinci::pricing::IVMethod::BRENT)
        .value("INTRINSIC_ONLY", davinci::pricing::IVMethod::INTRINSIC_ONLY);

    py::class_<davinci::pricing::IVResult>(pricing, "IVResult")
        .def_readonly("sigma", &davinci::pricing::IVResult::sigma)
        .def_readonly("iterations", &davinci::pricing::IVResult::iterations)
        .def_readonly("method_used", &davinci::pricing::IVResult::method_used)
        .def_readonly("converged", &davinci::pricing::IVResult::converged)
        .def_readonly("residual", &davinci::pricing::IVResult::residual);

    pricing.def("solve_implied_vol", &davinci::pricing::solve_implied_vol,
        py::arg("market_price"), py::arg("S"), py::arg("K"), py::arg("r"),
        py::arg("T"), py::arg("q") = 0.0, py::arg("is_call") = true,
        "Solve for implied volatility (Corrado-Miller → Newton-Raphson → Brent)");

    // ═════════════════════════════════════════════════════════════════════
    // MARKET MAKING MODULE
    // ═════════════════════════════════════════════════════════════════════

    auto mm = m.def_submodule("market_making", "Inventory-aware quoting models");

    py::class_<davinci::market_making::ASParams>(mm, "ASParams")
        .def(py::init<double, double, double, double>(),
             py::arg("gamma"), py::arg("kappa"), py::arg("sigma"), py::arg("T"))
        .def_readwrite("gamma", &davinci::market_making::ASParams::gamma)
        .def_readwrite("kappa", &davinci::market_making::ASParams::kappa)
        .def_readwrite("sigma", &davinci::market_making::ASParams::sigma)
        .def_readwrite("T", &davinci::market_making::ASParams::T);

    py::class_<davinci::market_making::ASQuote>(mm, "ASQuote")
        .def_readonly("bid", &davinci::market_making::ASQuote::bid)
        .def_readonly("ask", &davinci::market_making::ASQuote::ask)
        .def_readonly("reservation_price", &davinci::market_making::ASQuote::reservation_price)
        .def_readonly("optimal_spread", &davinci::market_making::ASQuote::optimal_spread)
        .def_readonly("skew", &davinci::market_making::ASQuote::skew);

    mm.def("as_quote", &davinci::market_making::compute_quotes,
        py::arg("mid_price"), py::arg("inventory"),
        py::arg("params"), py::arg("time_elapsed") = 0.0,
        "Compute Avellaneda-Stoikov optimal quotes");

    py::class_<davinci::market_making::LTParams>(mm, "LTParams")
        .def(py::init<double, double, double, double, double>(),
             py::arg("gamma"), py::arg("kappa"),
             py::arg("sigma_realized"), py::arg("sigma_implied"), py::arg("xi"))
        .def_readwrite("gamma", &davinci::market_making::LTParams::gamma)
        .def_readwrite("sigma_realized", &davinci::market_making::LTParams::sigma_realized)
        .def_readwrite("sigma_implied", &davinci::market_making::LTParams::sigma_implied);

    py::class_<davinci::market_making::LTQuote>(mm, "LTQuote")
        .def_readonly("bid", &davinci::market_making::LTQuote::bid)
        .def_readonly("ask", &davinci::market_making::LTQuote::ask)
        .def_readonly("spread", &davinci::market_making::LTQuote::spread)
        .def_readonly("vol_arb_component", &davinci::market_making::LTQuote::vol_arb_component)
        .def_readonly("elasticity_component", &davinci::market_making::LTQuote::elasticity_component)
        .def_readonly("inventory_component", &davinci::market_making::LTQuote::inventory_component);

    mm.def("lt_quote",
        static_cast<davinci::market_making::LTQuote(*)(
            double, double, double, double, double,
            const davinci::market_making::LTParams&, double, double)>(
            &davinci::market_making::compute_quotes),
        py::arg("mid_price"), py::arg("inventory"),
        py::arg("gamma_greek"), py::arg("vega_greek"),
        py::arg("S"), py::arg("params"),
        py::arg("time_remaining"), py::arg("dt"),
        "Compute Lucic-Tse vol-arb-adjusted quotes");

    // ═════════════════════════════════════════════════════════════════════
    // HEDGING MODULE
    // ═════════════════════════════════════════════════════════════════════

    auto hedging = m.def_submodule("hedging", "Leland, CRB, and Vega neutralization");

    hedging.def("leland_number", &davinci::hedging::leland_number,
        py::arg("txn_cost"), py::arg("sigma"), py::arg("dt"),
        "Compute the Leland number");

    hedging.def("effective_volatility", &davinci::hedging::effective_volatility,
        py::arg("sigma"), py::arg("le"), py::arg("gamma_sign"),
        "Compute modified volatility under Leland framework");

    py::enum_<davinci::hedging::OrderType>(hedging, "OrderType")
        .value("HOLD", davinci::hedging::OrderType::HOLD)
        .value("LIMIT_ORDER", davinci::hedging::OrderType::LIMIT_ORDER)
        .value("MARKET_ORDER", davinci::hedging::OrderType::MARKET_ORDER);

    py::class_<davinci::hedging::CRBParams>(hedging, "CRBParams")
        .def(py::init<double, double, double, double, double>(),
             py::arg("adverse_selection"), py::arg("half_spread"),
             py::arg("sigma"), py::arg("gamma_portfolio"),
             py::arg("max_delta_tolerance"))
        .def("inner_threshold", &davinci::hedging::CRBParams::inner_threshold)
        .def("outer_threshold", &davinci::hedging::CRBParams::outer_threshold);

    py::class_<davinci::hedging::CRBOrder>(hedging, "CRBOrder")
        .def_readonly("type", &davinci::hedging::CRBOrder::type)
        .def_readonly("shares", &davinci::hedging::CRBOrder::shares)
        .def_readonly("limit_price", &davinci::hedging::CRBOrder::limit_price)
        .def_readonly("urgency", &davinci::hedging::CRBOrder::urgency)
        .def_readonly("reason", &davinci::hedging::CRBOrder::reason);

    hedging.def("crb_hedge_action", &davinci::hedging::evaluate_hedge_action,
        py::arg("delta_actual"), py::arg("delta_target"),
        py::arg("S"), py::arg("params"),
        "Evaluate CRB limit-vs-market order hedge action");

    py::class_<davinci::hedging::PortfolioGreeks>(hedging, "PortfolioGreeks")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("delta"), py::arg("gamma"), py::arg("vega"),
             py::arg("theta"), py::arg("vanna"), py::arg("volga"))
        .def_readwrite("vega", &davinci::hedging::PortfolioGreeks::vega);

    py::class_<davinci::hedging::NeutralizeResult>(hedging, "NeutralizeResult")
        .def_readonly("portfolio_vega_before", &davinci::hedging::NeutralizeResult::portfolio_vega_before)
        .def_readonly("portfolio_vega_after", &davinci::hedging::NeutralizeResult::portfolio_vega_after)
        .def_readonly("units_required", &davinci::hedging::NeutralizeResult::units_required)
        .def_readonly("is_neutralized", &davinci::hedging::NeutralizeResult::is_neutralized);

    hedging.def("neutralize_with_varswap", &davinci::hedging::neutralize_with_varswap,
        py::arg("greeks"), py::arg("sigma_strike"), py::arg("T"),
        py::arg("tolerance") = 1e-6);

    hedging.def("neutralize_with_straddle", &davinci::hedging::neutralize_with_straddle,
        py::arg("greeks"), py::arg("S"), py::arg("T"), py::arg("sigma"),
        py::arg("straddle_gamma") = 0.0, py::arg("straddle_vanna") = 0.0,
        py::arg("straddle_volga") = 0.0, py::arg("tolerance") = 1e-6);

    hedging.def("gamma_scalping_pnl", &davinci::hedging::gamma_scalping_pnl,
        py::arg("gamma"), py::arg("S"),
        py::arg("sigma_realized"), py::arg("sigma_implied"),
        py::arg("dt") = 1.0 / 252.0);

    // ═════════════════════════════════════════════════════════════════════
    // DISPERSION MODULE
    // ═════════════════════════════════════════════════════════════════════

    auto disp = m.def_submodule("dispersion", "Dispersion trading signals");

    disp.def("dirty_correlation", &davinci::dispersion::compute_dirty_correlation,
        py::arg("index_iv"), py::arg("constituent_ivs"), py::arg("weights"),
        "Compute dirty implied correlation");

    py::class_<davinci::dispersion::ZScoreParams>(disp, "ZScoreParams")
        .def(py::init<>())
        .def_readwrite("window", &davinci::dispersion::ZScoreParams::window)
        .def_readwrite("entry_threshold", &davinci::dispersion::ZScoreParams::entry_threshold)
        .def_readwrite("exit_threshold", &davinci::dispersion::ZScoreParams::exit_threshold);

    py::enum_<davinci::dispersion::DispersionState>(disp, "DispersionState")
        .value("FLAT", davinci::dispersion::DispersionState::FLAT)
        .value("SHORT_DISPERSION", davinci::dispersion::DispersionState::SHORT_DISPERSION)
        .value("LONG_DISPERSION", davinci::dispersion::DispersionState::LONG_DISPERSION);

    py::class_<davinci::dispersion::ZScoreResult>(disp, "ZScoreResult")
        .def_readonly("zscore", &davinci::dispersion::ZScoreResult::zscore)
        .def_readonly("mean", &davinci::dispersion::ZScoreResult::mean)
        .def_readonly("stddev", &davinci::dispersion::ZScoreResult::stddev)
        .def_readonly("signal", &davinci::dispersion::ZScoreResult::signal)
        .def_readonly("is_entry", &davinci::dispersion::ZScoreResult::is_entry)
        .def_readonly("is_exit", &davinci::dispersion::ZScoreResult::is_exit);

    py::class_<davinci::dispersion::ZScoreEngine>(disp, "ZScoreEngine")
        .def(py::init<const davinci::dispersion::ZScoreParams&>(),
             py::arg("params") = davinci::dispersion::ZScoreParams{})
        .def("update", &davinci::dispersion::ZScoreEngine::update)
        .def("state", &davinci::dispersion::ZScoreEngine::state)
        .def("reset", &davinci::dispersion::ZScoreEngine::reset)
        .def("is_warmed_up", &davinci::dispersion::ZScoreEngine::is_warmed_up);

    // ═════════════════════════════════════════════════════════════════════
    // EXECUTION MODULE
    // ═════════════════════════════════════════════════════════════════════

    auto exec = m.def_submodule("execution", "Position sizing and quoting");

    exec.def("round_to_lots", &davinci::execution::round_to_lots,
        py::arg("theoretical_contracts"), py::arg("lot_size") = 100,
        py::arg("delta_per_contract") = 1.0);

    exec.def("approximate_pnl", &davinci::execution::approximate_pnl,
        py::arg("delta"), py::arg("gamma"), py::arg("vega"), py::arg("theta"),
        py::arg("dS"), py::arg("d_sigma") = 0.0, py::arg("dt") = 0.0);

    py::class_<davinci::execution::PnLApproximation>(exec, "PnLApproximation")
        .def_readonly("vega_pnl", &davinci::execution::PnLApproximation::vega_pnl)
        .def_readonly("delta_pnl", &davinci::execution::PnLApproximation::delta_pnl)
        .def_readonly("gamma_pnl", &davinci::execution::PnLApproximation::gamma_pnl)
        .def_readonly("theta_pnl", &davinci::execution::PnLApproximation::theta_pnl)
        .def_readonly("total_pnl", &davinci::execution::PnLApproximation::total_pnl);

    exec.def("quick_atm_vega", &davinci::execution::quick_atm_vega,
        py::arg("S"), py::arg("T"));
    exec.def("quick_atm_call", &davinci::execution::quick_atm_call,
        py::arg("S"), py::arg("sigma"), py::arg("T"));

    py::class_<davinci::execution::ConfidenceBand>(exec, "ConfidenceBand")
        .def_readonly("confidence_level", &davinci::execution::ConfidenceBand::confidence_level)
        .def_readonly("bid", &davinci::execution::ConfidenceBand::bid)
        .def_readonly("ask", &davinci::execution::ConfidenceBand::ask)
        .def_readonly("spread", &davinci::execution::ConfidenceBand::spread);

    py::class_<davinci::execution::ConfidenceQuotes>(exec, "ConfidenceQuotes")
        .def_readonly("fair_value", &davinci::execution::ConfidenceQuotes::fair_value)
        .def_readonly("uncertainty", &davinci::execution::ConfidenceQuotes::uncertainty)
        .def_readonly("bands", &davinci::execution::ConfidenceQuotes::bands);

    exec.def("confidence_quotes", &davinci::execution::generate_quotes,
        py::arg("fair_value"), py::arg("uncertainty"),
        py::arg("min_spread") = 0.01,
        "Generate 100/75/50% confidence bid-ask bands");
}
