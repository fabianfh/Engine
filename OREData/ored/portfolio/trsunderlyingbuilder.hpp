/*
 Copyright (C) 2021 Quaternion Risk Management Ltd
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

#pragma once

#include <ored/portfolio/enginefactory.hpp>
#include <ored/portfolio/simmcreditqualifiermapping.hpp>
#include <ored/portfolio/equityposition.hpp>
#include <ored/portfolio/commodityposition.hpp>
#include <ored/utilities/bondindexbuilder.hpp>
#include <qle/cashflows/trscashflow.hpp>

namespace ore {
namespace data {

struct TrsUnderlyingBuilder {
    virtual ~TrsUnderlyingBuilder() {}
    virtual void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>&
              getFxIndex, 
        const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const = 0;
    virtual void updateUnderlying(const boost::shared_ptr<ReferenceDataManager>& refData,
                                  boost::shared_ptr<Trade>& underlying, const std::string& parentId) const {}
};

class TrsUnderlyingBuilderFactory
    : public QuantLib::Singleton<TrsUnderlyingBuilderFactory, std::integral_constant<bool, true>> {
    map<std::string, boost::shared_ptr<TrsUnderlyingBuilder>> builders_;
    mutable std::shared_mutex mutex_;

public:
    std::map<std::string, boost::shared_ptr<TrsUnderlyingBuilder>> getBuilders() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return builders_;
    }
    boost::shared_ptr<TrsUnderlyingBuilder> getBuilder(const std::string& tradeType) const;
    void addBuilder(const std::string& tradeType, const boost::shared_ptr<TrsUnderlyingBuilder>& builder,
                    const bool allowOverwrite = false);
};

struct BondTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities,
          std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>&
              getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;
};

struct ForwardBondTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities,
          std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>& getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;
};

template<class T>
struct AssetPositionTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities,
          std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>& getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;

    void updateQuantities(std::map<std::string, double>& indexQuantities, const std::string& indexName,
                          const double qty) const;

    std::string getIndexCurrencyFromPosition(boost::shared_ptr<T> position, size_t i) const;

};

typedef AssetPositionTrsUnderlyingBuilder<ore::data::EquityPosition> EquityPositionTrsUnderlyingBuilder;
typedef AssetPositionTrsUnderlyingBuilder<ore::data::CommodityPosition> CommodityPositionTrsUnderlyingBuilder;

struct EquityOptionPositionTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>& getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;
};


struct BondPositionTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>& getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;
};

struct DerivativeTrsUnderlyingBuilder : public TrsUnderlyingBuilder {
    void
    build(const std::string& parentId, const boost::shared_ptr<Trade>& underlying,
          const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates, 
          const std::string& fundingCurrency, const boost::shared_ptr<EngineFactory>& engineFactory,
          boost::shared_ptr<QuantLib::Index>& underlyingIndex, Real& underlyingMultiplier,
          std::map<std::string, double>& indexQuantities, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices,
          Real& initialPrice, std::string& assetCurrency, std::string& creditRiskCurrency,
          std::map<std::string, SimmCreditQualifierMapping>& creditQualifierMapping, Date& maturity,
          const std::function<boost::shared_ptr<QuantExt::FxIndex>(
              const boost::shared_ptr<Market> market, const std::string& configuration, const std::string& domestic,
              const std::string& foreign, std::map<std::string, boost::shared_ptr<QuantExt::FxIndex>>& fxIndices)>& getFxIndex,
          const std::string& underlyingDerivativeId, RequiredFixings& fixings, std::vector<Leg>& returnLegs) const override;
};

// utility functions for bond TRS
void modifyBondTRSLeg(QuantLib::Leg& leg, QuantLib::Date issueDate);
Leg makeBondTRSLeg(const std::vector<Date>& valuationDates, const std::vector<Date>& paymentDates,
                   const BondIndexBuilder& bondIndexBuilder,
                   QuantLib::Real initialPrice = QuantLib::Null<QuantLib::Real>(),
                   QuantLib::ext::shared_ptr<QuantExt::FxIndex> fxIndex = nullptr);

} // namespace data
} // namespace ore
