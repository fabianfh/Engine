/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2009 Chris Kenyon

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
 */

#include <qle/cashflows/nonstandardcapflooredyoyinflationcoupon.hpp>
#include <ql/cashflows/inflationcouponpricer.hpp>

namespace QuantExt {

    void NonStandardCappedFlooredYoYInflationCoupon::setCommon(
        Rate cap, Rate floor) {

        isCapped_ = false;
        isFloored_ = false;

        if (gearing_ > 0) {
            if (cap != Null<Rate>()) {
                isCapped_ = true;
                cap_ = cap;
            }
            if (floor != Null<Rate>()) {
                floor_ =  floor;
                isFloored_ = true;
            }
        } else {
            if (cap != Null<Rate>()){
                floor_ = cap;
                isFloored_ = true;
            }
            if (floor != Null<Rate>()){
                isCapped_ = true;
                cap_ = floor;
            }
        }
        if (isCapped_ && isFloored_) {
            QL_REQUIRE(cap >= floor, "cap level (" << cap <<
                       ") less than floor level (" << floor << ")");
        }
    }


    NonStandardCappedFlooredYoYInflationCoupon::NonStandardCappedFlooredYoYInflationCoupon(
                const ext::shared_ptr<NonStandardYoYInflationCoupon>& underlying,
                        Rate cap, Rate floor)
        : NonStandardYoYInflationCoupon(
                         underlying->date(),
                         underlying->nominal(),
                         underlying->accrualStartDate(),
                         underlying->accrualEndDate(),
                         underlying->fixingDays(),
                         underlying->index(),
                         underlying->observationLag(),
                         underlying->dayCounter(),
                         underlying->gearing(),
                         underlying->spread(),
                         underlying->referencePeriodStart(),
                         underlying->referencePeriodEnd(),
                         underlying->addInflationNotional()),
      underlying_(underlying), isFloored_(false), isCapped_(false) {

        setCommon(cap, floor);
        registerWith(underlying);
    }

    NonStandardCappedFlooredYoYInflationCoupon::NonStandardCappedFlooredYoYInflationCoupon(const Date& paymentDate, Real nominal, const Date& startDate, const Date& endDate, Natural fixingDays,
            const ext::shared_ptr<ZeroInflationIndex>& index, const Period& observationLag, const DayCounter& dayCounter,
            Real gearing, Spread spread, const Rate cap, const Rate floor,
            const Date& refPeriodStart, const Date& refPeriodEnd, bool addInflationNotional) : NonStandardYoYInflationCoupon(paymentDate, nominal, startDate, endDate, fixingDays, index, observationLag, dayCounter, gearing, spread, refPeriodStart,referencePeriodEnd,addInflationNotional),
        isFloored_(false), isCapped_(false){}

    void
    NonStandardCappedFlooredYoYInflationCoupon::setPricer(
            const ext::shared_ptr<NonStandardYoYInflationCouponPricer>& pricer) {

        NonStandardYoYInflationCoupon::setPricer(pricer);
        if (underlying_) underlying_->setPricer(pricer);
    }


    Rate NonStandardCappedFlooredYoYInflationCoupon::rate() const {
        Rate swapletRate = underlying_ ? underlying_->rate() : NonStandardYoYInflationCoupon::rate();

        if(isFloored_ || isCapped_) {
            if (underlying_) {
                QL_REQUIRE(underlying_->pricer(), "pricer not set");
            } else {
                QL_REQUIRE(pricer_, "pricer not set");
            }
        }

        Rate floorletRate = 0.;
        if(isFloored_) {
            floorletRate =
            underlying_ ?
            underlying_->pricer()->floorletRate(effectiveFloor()) :
            pricer()->floorletRate(effectiveFloor())
            ;
        }
        Rate capletRate = 0.;
        if(isCapped_) {
            capletRate =
            underlying_ ?
            underlying_->pricer()->capletRate(effectiveCap()) :
            pricer()->capletRate(effectiveCap())
            ;
        }

        return swapletRate + floorletRate - capletRate;
    }


    Rate NonStandardCappedFlooredYoYInflationCoupon::cap() const {
        if ( (gearing_ > 0) && isCapped_)
            return cap_;
        if ( (gearing_ < 0) && isFloored_)
            return floor_;
        return Null<Rate>();
    }


    Rate NonStandardCappedFlooredYoYInflationCoupon::floor() const {
        if ( (gearing_ > 0) && isFloored_)
            return floor_;
        if ( (gearing_ < 0) && isCapped_)
            return cap_;
        return Null<Rate>();
    }


    Rate NonStandardCappedFlooredYoYInflationCoupon::effectiveCap() const {
        return (cap_ - (addInflationNotional_ ? 1 : 0) -spread()) / gearing();
    }


    Rate NonStandardCappedFlooredYoYInflationCoupon::effectiveFloor() const {
        return (floor_ - (addInflationNotional_ ? 1 : 0) - spread()) / gearing();
    }


    void NonStandardCappedFlooredYoYInflationCoupon::update() {
        notifyObservers();
    }


    void NonStandardCappedFlooredYoYInflationCoupon::accept(AcyclicVisitor& v) {
        typedef YoYInflationCoupon super;
        Visitor<NonStandardCappedFlooredYoYInflationCoupon>* v1 =
            dynamic_cast<Visitor<NonStandardCappedFlooredYoYInflationCoupon>*>(&v);

        if (v1 != 0)
            v1->visit(*this);
        else
            NonStandardYoYInflationCoupon::accept(v);
    }

}

