/*
 Copyright (C) 2016 Quaternion Risk Management Ltd
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

/*! \file ored/configuration/curveconfigurations.hpp
    \brief Curve configuration repoistory
    \ingroup configuration
*/

#pragma once

#include <ored/configuration/basecorrelationcurveconfig.hpp>
#include <ored/configuration/capfloorvolcurveconfig.hpp>
#include <ored/configuration/cdsvolcurveconfig.hpp>
#include <ored/configuration/commoditycurveconfig.hpp>
#include <ored/configuration/commodityvolcurveconfig.hpp>
#include <ored/configuration/defaultcurveconfig.hpp>
#include <ored/configuration/equitycurveconfig.hpp>
#include <ored/configuration/equityvolcurveconfig.hpp>
#include <ored/configuration/fxspotconfig.hpp>
#include <ored/configuration/fxvolcurveconfig.hpp>
#include <ored/configuration/inflationcapfloorpricesurfaceconfig.hpp>
#include <ored/configuration/inflationcapfloorvolcurveconfig.hpp>
#include <ored/configuration/inflationcurveconfig.hpp>
#include <ored/configuration/securityconfig.hpp>
#include <ored/configuration/swaptionvolcurveconfig.hpp>
#include <ored/configuration/yieldcurveconfig.hpp>
#include <ored/configuration/correlationcurveconfig.hpp>
#include <ored/marketdata/curvespec.hpp>
#include <ored/marketdata/todaysmarketparameters.hpp>
#include <ored/utilities/xmlutils.hpp>


namespace ore {
namespace data {
using ore::data::XMLSerializable;
using ore::data::XMLNode;

//! Container class for all Curve Configurations
/*!
  \ingroup configuration
*/
class CurveConfigurations : public XMLSerializable {
public:
    //! Default constructor
    CurveConfigurations() { }

    //! \name Setters and Getters
    //@{
    boost::shared_ptr<YieldCurveConfig>& yieldCurveConfig(const string& curveID) { return yieldCurveConfigs_[curveID]; }
    const boost::shared_ptr<YieldCurveConfig>& yieldCurveConfig(const string& curveID) const;

    boost::shared_ptr<FXVolatilityCurveConfig>& fxVolCurveConfig(const string& curveID) {
        return fxVolCurveConfigs_[curveID];
    }
    const boost::shared_ptr<FXVolatilityCurveConfig>& fxVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<SwaptionVolatilityCurveConfig>& swaptionVolCurveConfig(const string& curveID) {
        return swaptionVolCurveConfigs_[curveID];
    }
    const boost::shared_ptr<SwaptionVolatilityCurveConfig>& swaptionVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<CapFloorVolatilityCurveConfig>& capFloorVolCurveConfig(const string& curveID) {
        return capFloorVolCurveConfigs_[curveID];
    }
    const boost::shared_ptr<CapFloorVolatilityCurveConfig>& capFloorVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<DefaultCurveConfig>& defaultCurveConfig(const string& curveID) {
        return defaultCurveConfigs_[curveID];
    }
    const boost::shared_ptr<DefaultCurveConfig>& defaultCurveConfig(const string& curveID) const;

    boost::shared_ptr<CDSVolatilityCurveConfig>& cdsVolCurveConfig(const string& curveID) {
        return cdsVolCurveConfigs_[curveID];
    }
    const boost::shared_ptr<CDSVolatilityCurveConfig>& cdsVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<BaseCorrelationCurveConfig>& baseCorrelationCurveConfig(const string& curveID) {
        return baseCorrelationCurveConfigs_[curveID];
    }
    const boost::shared_ptr<BaseCorrelationCurveConfig>& baseCorrelationCurveConfig(const string& curveID) const;

    boost::shared_ptr<InflationCurveConfig>& inflationCurveConfig(const string& curveID) {
        return inflationCurveConfigs_[curveID];
    };
    const boost::shared_ptr<InflationCurveConfig>& inflationCurveConfig(const string& curveID) const;

    boost::shared_ptr<InflationCapFloorPriceSurfaceConfig>& inflationCapFloorPriceSurfaceConfig(const string& curveID) {
        return inflationCapFloorPriceSurfaceConfigs_[curveID];
    };
    const boost::shared_ptr<InflationCapFloorPriceSurfaceConfig>&
        inflationCapFloorPriceSurfaceConfig(const string& curveID) const;

    boost::shared_ptr<InflationCapFloorVolatilityCurveConfig>& inflationCapFloorVolCurveConfig(const string& curveID) {
        return inflationCapFloorVolCurveConfigs_[curveID];
    };
    const boost::shared_ptr<InflationCapFloorVolatilityCurveConfig>&
        inflationCapFloorVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<EquityCurveConfig>& equityCurveConfig(const string& curveID) {
        return equityCurveConfigs_[curveID];
    };
    const boost::shared_ptr<EquityCurveConfig>& equityCurveConfig(const string& curveID) const;

    boost::shared_ptr<EquityVolatilityCurveConfig>& equityVolCurveConfig(const string& curveID) {
        return equityVolCurveConfigs_[curveID];
    };
    const boost::shared_ptr<EquityVolatilityCurveConfig>& equityVolCurveConfig(const string& curveID) const;

    boost::shared_ptr<SecurityConfig>& securityConfig(const string& curveID) { return securityConfigs_[curveID]; };
    const boost::shared_ptr<SecurityConfig>& securityConfig(const string& curveID) const;

    boost::shared_ptr<FXSpotConfig>& fxSpotConfig(const string& curveID) { return fxSpotConfigs_[curveID]; };
    const boost::shared_ptr<FXSpotConfig>& fxSpotConfig(const string& curveID) const;

    boost::shared_ptr<CommodityCurveConfig>& commodityCurveConfig(const std::string& curveID) {
        return commodityCurveConfigs_[curveID];
    };
    const boost::shared_ptr<CommodityCurveConfig>& commodityCurveConfig(const std::string& curveID) const;

    boost::shared_ptr<CommodityVolatilityCurveConfig>& commodityVolatilityCurveConfig(const std::string& curveID) {
        return commodityVolatilityCurveConfigs_[curveID];
    };
    const boost::shared_ptr<CommodityVolatilityCurveConfig>& commodityVolatilityCurveConfig(const std::string& curveID) const;
    
    boost::shared_ptr<CorrelationCurveConfig>& correlationCurveConfig(const std::string& curveID) {
        return correlationCurveConfigs_[curveID];
    };
    const boost::shared_ptr<CorrelationCurveConfig>& correlationCurveConfig(const std::string& curveID) const;
    
    /*! Return the set of quotes that are required by the CurveConfig elements in CurveConfigurations.
        
        If \p todaysMarketParams is a `nullptr`, the set of quotes required by all CurveConfig elements is returned.
        If \p todaysMarketParams is provided, the set of quotes required by only those CurveConfig elements appearing 
        in \p todaysMarketParams for the given configuration(s) is returned. 
    */
    std::set<string> quotes(boost::shared_ptr<const TodaysMarketParameters> todaysMarketParams = nullptr, 
        const std::set<std::string>& configurations = { "" }) const;
    //@}

    //! \name Serialisation
    //@{
    void fromXML(XMLNode* node) override;
    XMLNode* toXML(XMLDocument& doc) override;
    //@}
private:
    std::map<std::string, boost::shared_ptr<YieldCurveConfig>> yieldCurveConfigs_;
    std::map<std::string, boost::shared_ptr<FXVolatilityCurveConfig>> fxVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<SwaptionVolatilityCurveConfig>> swaptionVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<CapFloorVolatilityCurveConfig>> capFloorVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<DefaultCurveConfig>> defaultCurveConfigs_;
    std::map<std::string, boost::shared_ptr<CDSVolatilityCurveConfig>> cdsVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<BaseCorrelationCurveConfig>> baseCorrelationCurveConfigs_;
    std::map<std::string, boost::shared_ptr<InflationCurveConfig>> inflationCurveConfigs_;
    std::map<std::string, boost::shared_ptr<InflationCapFloorPriceSurfaceConfig>> inflationCapFloorPriceSurfaceConfigs_;
    std::map<std::string, boost::shared_ptr<InflationCapFloorVolatilityCurveConfig>> inflationCapFloorVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<EquityCurveConfig>> equityCurveConfigs_;
    std::map<std::string, boost::shared_ptr<EquityVolatilityCurveConfig>> equityVolCurveConfigs_;
    std::map<std::string, boost::shared_ptr<SecurityConfig>> securityConfigs_;
    std::map<std::string, boost::shared_ptr<FXSpotConfig>> fxSpotConfigs_;
    std::map<std::string, boost::shared_ptr<CommodityCurveConfig>> commodityCurveConfigs_;
    std::map<std::string, boost::shared_ptr<CommodityVolatilityCurveConfig>> commodityVolatilityCurveConfigs_;
    std::map<std::string, boost::shared_ptr<CorrelationCurveConfig>> correlationCurveConfigs_;
};
} // namespace data
} // namespace ore
