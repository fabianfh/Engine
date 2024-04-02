/*
 Copyright (C) 2023 Quaternion Risk Management Ltd
 All rights reserved.

 This file is part of ORE, a free-software/open-source library
 for transparent pricing and risk analysis - http://opensourcerisk.org

 ORE is free software: you can redistribute it and/or modify it
 under the terms of the Modified BSD License.  You should have received a
 copy of the license along with this program.
 The license is also available online at <http://opensourcerisk.org>

 This program is distributed on the basis that it will form a useful
 contribution to risk analytics and model standardisation, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the license for more details.
*/

#include <orea/app/analytics/scenarioanalytic.hpp>
#include <orea/scenario/scenariowriter.hpp>

using namespace ore::analytics;

namespace ore {
namespace analytics {
	
// ScenarioAnalytic
ScenarioAnalytic::ScenarioAnalytic(const boost::shared_ptr<InputParameters>& inputs)
    : Analytic(std::make_unique<ScenarioAnalyticImpl>(inputs), {"SCENARIO"}, inputs, true, false, false, false) {}

void ScenarioAnalyticImpl::setUpConfigurations() {
    analytic()->configurations().todaysMarketParams = inputs_->todaysMarketParams();
    analytic()->configurations().simMarketParams = inputs_->scenarioSimMarketParams();
}

void ScenarioAnalyticImpl::runAnalytic(const boost::shared_ptr<InMemoryLoader>& loader,
                                   const std::set<std::string>& runTypes) {

    if (!analytic()->match(runTypes))
        return;

    LOG("ScenarioAnalytic::runAnalytic called");
        
    auto scenarioAnalytic = static_cast<ScenarioAnalytic*>(analytic());
    QL_REQUIRE(scenarioAnalytic, "Analytic must be of type ScenarioAnalytic");

    analytic()->buildMarket(loader);

    LOG("Building scenario simulation market for date " << io::iso_date(inputs_->asof()));
    // FIXME: *configurations_.todaysMarketParams uninitialized?
    auto ssm = boost::make_shared<ScenarioSimMarket>(
        analytic()->market(), scenarioAnalytic->configurations().simMarketParams, Market::defaultConfiguration,
        *scenarioAnalytic->configurations().curveConfig, *scenarioAnalytic->configurations().todaysMarketParams, true,
        false, false, false, *inputs_->iborFallbackConfig());

    setScenarioSimMarket(ssm);
    auto scenario = ssm->baseScenario();
    setScenario(scenario);

    boost::shared_ptr<InMemoryReport> report = boost::make_shared<InMemoryReport>();
    auto sw = ScenarioWriter(nullptr, report);
    sw.writeScenario(scenario, true);
    analytic()->reports()[label()]["scenario"] = report;
}

} // namespace analytics
} // namespace ore
