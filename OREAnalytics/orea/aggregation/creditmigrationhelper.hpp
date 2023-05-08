/*
 Copyright (C) 2017 Quaternion Risk Management Ltd
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

/*! \file aggregation/creditmigrationhelper.hpp
    \brief Credit migration helper class
    \ingroup engine
*/

#pragma once

#include <orea/aggregation/creditsimulationparameters.hpp>
#include <orea/cube/npvcube.hpp>
#include <orea/scenario/aggregationscenariodata.hpp>

#include <ored/portfolio/bond.hpp>
#include <ored/portfolio/trade.hpp>
#include <ored/utilities/log.hpp>

#include <qle/models/hullwhitebucketing.hpp>

#include <ql/math/matrix.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>

namespace ore {
namespace analytics {

/*! Helper for credit migration risk calculation
   Dynamics of entity i's state X_i:
     dX_i = dY_i + dZ_i
   with
     - systemic part dY_i = \sum_{j=1}^n \beta_{ij} dG_j 
     - n correlated global factors G_j
     - entity specific factor loadings \beta_{ij}
     - idiosyncratic part dZ_i = \sigma_i dW_i 
     - independent  Wiener processes W, i.e. dW_k dW_l = 0 and dW_k dG_j = 0

   \warning Evaluation modes ForwardSimulationA and ForwardSimulationB are untested and need review
*/
class CreditMigrationHelper {
public:
    enum class CreditMode { Migration, Default };
    enum class LoanExposureMode { Notional, Value };
    enum class Evaluation { Analytic, ForwardSimulationA, ForwardSimulationB, TerminalSimulation };

    CreditMigrationHelper(const boost::shared_ptr<CreditSimulationParameters> parameters,
                          const boost::shared_ptr<NPVCube> cube, const boost::shared_ptr<NPVCube> nettedCube,
                          const boost::shared_ptr<AggregationScenarioData> aggData, const Size cubeIndexCashflows,
                          const Size cubeIndexStateNpvs, const Real distributionLowerBound,
                          const Real distributionUpperBound, const Size buckets, const Matrix& globalFactorCorrelation,
                          const std::string& baseCurrency);

    //! builds the helper for a specific subset of trades stored in the cube
    void build(const std::map<std::string, boost::shared_ptr<Trade>>& trades);

    const std::vector<Real>& upperBucketBound() const { return bucketing_.upperBucketBound(); }
    // 
    Array pnlDistribution(const Size date);

    //! Correlation of factor1/factor2 moves from today to date
    void stateCorrelations(const Size date, Size entity1, Size entity2, string index1, Real initialIndexValue1,
                           string index2, Real initialIndexValue2);

private:
    /*! Get the transition matrix from today to date by entity, 
      sanitise the annual transition matrix input, 
      rescale to the desired horizon/date using the generator, 
      cache the result so that we do the sanitising/rescaling only once
    */
    std::map<string, Matrix> rescaledTransitionMatrices(const Size date);

    /*! Initialise 
      - the variance of the global part Y_i of entity state X_i, for all entities
      - the global part Y_i of entity i's state X_i by date index, entity index and sample number
        using the simulated global state paths stored in the aggregation scenario data object
     */
    void init();

    //! Allocate 3d storage for the simulated idiosyncratic factors by entity, date and sample
    void initEntityStateSimulation();

    /*!
      Initialise the entity state simulationn for a given date and global path and return transition
      matrices by simulation date along the path and by entity 
      Case Evaluation::TerminalSimulation:
        - return transition matrix for each entity for one date (the terminal date) only, 
          conditional on the global terminal state on the given path     
      Case Evaluation::ForwardSimulationB:
        - return forward transition matrices M(t_{j-1},t_j) for each cube date and each entity 
          conditional on the terminal date, across all cube dates on the given path
          such that the product of these matrices along the path matches the terminal 
          conditional transition matrix
        - M(t_{j-1},t_j) = M^{-1}(0, t_{j-1}) * M(0, t_j)
      Case Evaluation::ForwardSimulationA:
        - return forward transition matrices M_i(t_{j-1},t_j) where each of the underlying
          transition matrices M_i(0,t_j) is conditional on the global state at t_j 
     */
    std::vector<std::vector<Matrix>> initEntityStateSimulation(const Size date, const Size path);

    /*! 
      Generate one entity state sample path for all entities given the global state path
      and given the forward transition matrices
    */
    void simulateEntityStates(const std::vector<std::vector<Matrix>>& cond, const Size path,
                              const MersenneTwisterUniformRng& mt, const Size date);

    //! Look up the simulated entity credit state for the given entity, date and path
    Size simulatedEntityState(const Size i, const Size date, const Size path) const;

    //! Return true if both A and B default on the given path and A defaults before B
    bool simulatedDefaultOrder(const Size entityA, const Size entityB, const Size date, const Size path,
                               const Size n) const;

    /*! Return a single PnL impact due to credit migration or default of Bond/CDS issuers and default of 
      netting set counterparties on the given global path 
    */
    Real generateMigrationPnl(const Size date, const Size path, const Size n) const;

    /*! Return a vector of PnL impacts and associated conditional probabilities for the specified global path,
      due to credit migration or default of Bond/CDS issuers and default of netting set counterparties
    */
    void generateConditionalMigrationPnl(const Size date, const Size path, const std::map<string, Matrix>& transMat,
                                         std::vector<Array>& condProbs, std::vector<Array>& pnl) const;

    boost::shared_ptr<CreditSimulationParameters> parameters_;
    boost::shared_ptr<NPVCube> cube_, nettedCube_;
    boost::shared_ptr<AggregationScenarioData> aggData_;
    Size cubeIndexCashflows_, cubeIndexStateNpvs_;
    Matrix globalFactorCorrelation_;
    std::string baseCurrency_;

    CreditMode creditMode_;
    LoanExposureMode loanExposureMode_;
    Evaluation evaluation_;
    std::vector<Real> cubeTimes_;

    QuantExt::Bucketing bucketing_;

    std::vector<std::set<std::string>> issuerTradeIds_;
    std::vector<std::set<std::string>> cptyNettingSetIds_;

    std::map<std::string, std::string> tradeCreditCurves_;
    std::map<std::string, Real> tradeNotionals_;
    std::map<std::string, std::string> tradeCurrencies_;
    std::map<std::string, Size> tradeCdsCptyIdx_;

    // Transition matrix rows
    Size n_;
    std::vector<std::map<string, Matrix>> rescaledTransitionMatrices_;
    // Variance of the systemic part (Y_i) of entity state X_i 
    std::vector<Real> globalVar_;
    // Storage for the simulated idiosyncratic factors Z by entity, date, sample number
    std::vector<std::vector<std::vector<Size>>> simulatedEntityState_;

    std::vector<std::vector<Matrix>> entityStateSimulationMatrices_;
    // Systemic part (Y_i) of entity state X_i by date index, entity index, sample number
    std::vector<std::vector<std::vector<Real>>> globalStates_;
};

CreditMigrationHelper::CreditMode parseCreditMode(const std::string& s);
CreditMigrationHelper::LoanExposureMode parseLoanExposureMode(const std::string& s);
CreditMigrationHelper::Evaluation parseEvaluation(const std::string& s);

} // namespace analytics
} // namespace ore
