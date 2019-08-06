/*
 Copyright (C) 2019 Quaternion Risk Management Ltd
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

/*! \file blackvariancesurfacesparse.hpp
 \brief Black volatility surface modelled as variance surface
 */

#ifndef quantext_black_variance_surface_stddevs_hpp
#define quantext_black_variance_surface_stddevs_hpp

#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <qle/interpolators/optioninterpolator2d.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <qle/termstructures/blackvariancesurfacemoneyness.hpp>

namespace QuantExt {
using namespace QuantLib;


//! Black volatility surface based on forward moneyness
//! \ingroup termstructures
    class BlackVarianceSurfaceStdDevs : public BlackVarianceSurfaceMoneyness {
    public:
        /*! Moneyness is defined here as forward moneyness, ie K/F */
        BlackVarianceSurfaceStdDevs(const Calendar& cal, const Handle<Quote>& spot, const std::vector<Time>& times,
            const std::vector<Real>& moneyness,
            const std::vector<std::vector<Handle<Quote> > >& blackVolMatrix,
            const DayCounter& dayCounter, const Handle<YieldTermStructure>& forTS,
            const Handle<YieldTermStructure>& domTS, bool stickyStrike = false, bool flatExtrapMoneyness = false);

    private:
        virtual Real moneyness(Time t, Real strike) const;
        Handle<YieldTermStructure> forTS_; // calculates fwd if StickyStrike==false
        Handle<YieldTermStructure> domTS_;
        std::vector<Real> forwards_; // cache fwd values if StickyStrike==true
        Interpolation forwardCurve_;
        Interpolation atmVarCurve_;
        std::vector<Time> atmTimes_;
        std::vector<Real> atmVariances_;
        bool flatExtrapolateMoneyness_; // flatly extraplate on moneyness axis
    };


} // namespace QuantExt

#endif
