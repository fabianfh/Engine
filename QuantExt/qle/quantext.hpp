// Autogenerated by cmake
// Do not edit

#ifdef BOOST_MSVC
#  include <qle/auto_link.hpp>
#endif

#include <qle/calendars/chile.hpp>
#include <qle/calendars/colombia.hpp>
#include <qle/calendars/france.hpp>
#include <qle/calendars/malaysia.hpp>
#include <qle/calendars/netherlands.hpp>
#include <qle/calendars/peru.hpp>
#include <qle/calendars/philippines.hpp>
#include <qle/calendars/thailand.hpp>
#include <qle/cashflows/averageonindexedcoupon.hpp>
#include <qle/cashflows/averageonindexedcouponpricer.hpp>
#include <qle/cashflows/brlcdicouponpricer.hpp>
#include <qle/cashflows/couponpricer.hpp>
#include <qle/cashflows/equitycoupon.hpp>
#include <qle/cashflows/equitycouponpricer.hpp>
#include <qle/cashflows/floatingannuitycoupon.hpp>
#include <qle/cashflows/floatingannuitynominal.hpp>
#include <qle/cashflows/floatingratefxlinkednotionalcoupon.hpp>
#include <qle/cashflows/fxlinkedcashflow.hpp>
#include <qle/cashflows/lognormalcmsspreadpricer.hpp>
#include <qle/cashflows/quantocouponpricer.hpp>
#include <qle/cashflows/strippedcapflooredyoyinflationcoupon.hpp>
#include <qle/cashflows/subperiodscoupon.hpp>
#include <qle/cashflows/subperiodscouponpricer.hpp>
#include <qle/currencies/africa.hpp>
#include <qle/currencies/america.hpp>
#include <qle/currencies/asia.hpp>
#include <qle/currencies/metals.hpp>
#include <qle/indexes/bmaindexwrapper.hpp>
#include <qle/indexes/bondindex.hpp>
#include <qle/indexes/cacpi.hpp>
#include <qle/indexes/dkcpi.hpp>
#include <qle/indexes/equityindex.hpp>
#include <qle/indexes/fxindex.hpp>
#include <qle/indexes/genericiborindex.hpp>
#include <qle/indexes/ibor/audbbsw.hpp>
#include <qle/indexes/ibor/brlcdi.hpp>
#include <qle/indexes/ibor/chfsaron.hpp>
#include <qle/indexes/ibor/chftois.hpp>
#include <qle/indexes/ibor/clpcamara.hpp>
#include <qle/indexes/ibor/copibr.hpp>
#include <qle/indexes/ibor/corra.hpp>
#include <qle/indexes/ibor/czkpribor.hpp>
#include <qle/indexes/ibor/demlibor.hpp>
#include <qle/indexes/ibor/dkkcibor.hpp>
#include <qle/indexes/ibor/dkkois.hpp>
#include <qle/indexes/ibor/hkdhibor.hpp>
#include <qle/indexes/ibor/hufbubor.hpp>
#include <qle/indexes/ibor/idridrfix.hpp>
#include <qle/indexes/ibor/idrjibor.hpp>
#include <qle/indexes/ibor/ilstelbor.hpp>
#include <qle/indexes/ibor/inrmifor.hpp>
#include <qle/indexes/ibor/krwcd.hpp>
#include <qle/indexes/ibor/krwkoribor.hpp>
#include <qle/indexes/ibor/mxntiie.hpp>
#include <qle/indexes/ibor/myrklibor.hpp>
#include <qle/indexes/ibor/noknibor.hpp>
#include <qle/indexes/ibor/nowa.hpp>
#include <qle/indexes/ibor/nzdbkbm.hpp>
#include <qle/indexes/ibor/phpphiref.hpp>
#include <qle/indexes/ibor/plnpolonia.hpp>
#include <qle/indexes/ibor/plnwibor.hpp>
#include <qle/indexes/ibor/rubmosprime.hpp>
#include <qle/indexes/ibor/saibor.hpp>
#include <qle/indexes/ibor/seksior.hpp>
#include <qle/indexes/ibor/sekstibor.hpp>
#include <qle/indexes/ibor/sgdsibor.hpp>
#include <qle/indexes/ibor/sgdsor.hpp>
#include <qle/indexes/ibor/skkbribor.hpp>
#include <qle/indexes/ibor/thbbibor.hpp>
#include <qle/indexes/ibor/tonar.hpp>
#include <qle/indexes/ibor/twdtaibor.hpp>
#include <qle/indexes/inflationindexobserver.hpp>
#include <qle/indexes/inflationindexwrapper.hpp>
#include <qle/indexes/region.hpp>
#include <qle/indexes/secpi.hpp>
#include <qle/instruments/averageois.hpp>
#include <qle/instruments/brlcdiswap.hpp>
#include <qle/instruments/cdsoption.hpp>
#include <qle/instruments/commodityforward.hpp>
#include <qle/instruments/creditdefaultswap.hpp>
#include <qle/instruments/crossccybasismtmresetswap.hpp>
#include <qle/instruments/crossccybasisswap.hpp>
#include <qle/instruments/crossccyfixfloatswap.hpp>
#include <qle/instruments/crossccyswap.hpp>
#include <qle/instruments/currencyswap.hpp>
#include <qle/instruments/deposit.hpp>
#include <qle/instruments/equityforward.hpp>
#include <qle/instruments/fixedbmaswap.hpp>
#include <qle/instruments/forwardbond.hpp>
#include <qle/instruments/fxforward.hpp>
#include <qle/instruments/impliedbondspread.hpp>
#include <qle/instruments/makeaverageois.hpp>
#include <qle/instruments/makecds.hpp>
#include <qle/instruments/oibasisswap.hpp>
#include <qle/instruments/oiccbasisswap.hpp>
#include <qle/instruments/payment.hpp>
#include <qle/instruments/subperiodsswap.hpp>
#include <qle/instruments/tenorbasisswap.hpp>
#include <qle/interpolators/optioninterpolator2d.hpp>
#include <qle/math/deltagammavar.hpp>
#include <qle/math/fillemptymatrix.hpp>
#include <qle/math/flatextrapolation.hpp>
#include <qle/math/nadarayawatson.hpp>
#include <qle/math/stabilisedglls.hpp>
#include <qle/math/trace.hpp>
#include <qle/methods/multipathgeneratorbase.hpp>
#include <qle/models/cdsoptionhelper.hpp>
#include <qle/models/cmscaphelper.hpp>
#include <qle/models/cpicapfloorhelper.hpp>
#include <qle/models/crlgm1fparametrization.hpp>
#include <qle/models/crossassetanalytics.hpp>
#include <qle/models/crossassetanalyticsbase.hpp>
#include <qle/models/crossassetmodel.hpp>
#include <qle/models/crossassetmodelimpliedeqvoltermstructure.hpp>
#include <qle/models/crossassetmodelimpliedfxvoltermstructure.hpp>
#include <qle/models/dkimpliedyoyinflationtermstructure.hpp>
#include <qle/models/dkimpliedzeroinflationtermstructure.hpp>
#include <qle/models/eqbsconstantparametrization.hpp>
#include <qle/models/eqbsparametrization.hpp>
#include <qle/models/eqbspiecewiseconstantparametrization.hpp>
#include <qle/models/fxbsconstantparametrization.hpp>
#include <qle/models/fxbsparametrization.hpp>
#include <qle/models/fxbspiecewiseconstantparametrization.hpp>
#include <qle/models/fxeqoptionhelper.hpp>
#include <qle/models/gaussian1dcrossassetadaptor.hpp>
#include <qle/models/infdkparametrization.hpp>
#include <qle/models/irlgm1fconstantparametrization.hpp>
#include <qle/models/irlgm1fparametrization.hpp>
#include <qle/models/irlgm1fpiecewiseconstanthullwhiteadaptor.hpp>
#include <qle/models/irlgm1fpiecewiseconstantparametrization.hpp>
#include <qle/models/irlgm1fpiecewiselinearparametrization.hpp>
#include <qle/models/lgm.hpp>
#include <qle/models/lgmimplieddefaulttermstructure.hpp>
#include <qle/models/lgmimpliedyieldtermstructure.hpp>
#include <qle/models/linkablecalibratedmodel.hpp>
#include <qle/models/parametrization.hpp>
#include <qle/models/piecewiseconstanthelper.hpp>
#include <qle/models/pseudoparameter.hpp>
#include <qle/pricingengines/analyticcclgmfxoptionengine.hpp>
#include <qle/pricingengines/analyticdkcpicapfloorengine.hpp>
#include <qle/pricingengines/analyticlgmcdsoptionengine.hpp>
#include <qle/pricingengines/analyticlgmswaptionengine.hpp>
#include <qle/pricingengines/analyticxassetlgmeqoptionengine.hpp>
#include <qle/pricingengines/baroneadesiwhaleyengine.hpp>
#include <qle/pricingengines/blackcdsoptionengine.hpp>
#include <qle/pricingengines/cpibacheliercapfloorengine.hpp>
#include <qle/pricingengines/cpiblackcapfloorengine.hpp>
#include <qle/pricingengines/cpicapfloorengines.hpp>
#include <qle/pricingengines/crossccyswapengine.hpp>
#include <qle/pricingengines/depositengine.hpp>
#include <qle/pricingengines/discountingcommodityforwardengine.hpp>
#include <qle/pricingengines/discountingcurrencyswapengine.hpp>
#include <qle/pricingengines/discountingequityforwardengine.hpp>
#include <qle/pricingengines/discountingforwardbondengine.hpp>
#include <qle/pricingengines/discountingfxforwardengine.hpp>
#include <qle/pricingengines/discountingriskybondengine.hpp>
#include <qle/pricingengines/discountingswapenginemulticurve.hpp>
#include <qle/pricingengines/lgmconvolutionsolver.hpp>
#include <qle/pricingengines/midpointcdsengine.hpp>
#include <qle/pricingengines/numericlgmswaptionengine.hpp>
#include <qle/pricingengines/oiccbasisswapengine.hpp>
#include <qle/pricingengines/paymentdiscountingengine.hpp>
#include <qle/processes/crossassetstateprocess.hpp>
#include <qle/processes/irlgm1fstateprocess.hpp>
#include <qle/quotes/exceptionquote.hpp>
#include <qle/quotes/logquote.hpp>
#include <qle/termstructures/averageoisratehelper.hpp>
#include <qle/termstructures/basistwoswaphelper.hpp>
#include <qle/termstructures/blackinvertedvoltermstructure.hpp>
#include <qle/termstructures/blackvariancecurve3.hpp>
#include <qle/termstructures/blackvariancesurfacemoneyness.hpp>
#include <qle/termstructures/blackvariancesurfacesparse.hpp>
#include <qle/termstructures/blackvolsurfacewithatm.hpp>
#include <qle/termstructures/brlcdiratehelper.hpp>
#include <qle/termstructures/capfloorhelper.hpp>
#include <qle/termstructures/capfloortermvolcurve.hpp>
#include <qle/termstructures/capfloortermvolsurface.hpp>
#include <qle/termstructures/correlationtermstructure.hpp>
#include <qle/termstructures/crossccybasismtmresetswaphelper.hpp>
#include <qle/termstructures/crossccybasisswaphelper.hpp>
#include <qle/termstructures/crossccyfixfloatswaphelper.hpp>
#include <qle/termstructures/datedstrippedoptionlet.hpp>
#include <qle/termstructures/datedstrippedoptionletadapter.hpp>
#include <qle/termstructures/datedstrippedoptionletbase.hpp>
#include <qle/termstructures/defaultprobabilityhelpers.hpp>
#include <qle/termstructures/discountratiomodifiedcurve.hpp>
#include <qle/termstructures/dynamicblackvoltermstructure.hpp>
#include <qle/termstructures/dynamicoptionletvolatilitystructure.hpp>
#include <qle/termstructures/dynamicstype.hpp>
#include <qle/termstructures/dynamicswaptionvolmatrix.hpp>
#include <qle/termstructures/dynamicyoyoptionletvolatilitystructure.hpp>
#include <qle/termstructures/equityforwardcurvestripper.hpp>
#include <qle/termstructures/equityvolconstantspread.hpp>
#include <qle/termstructures/flatcorrelation.hpp>
#include <qle/termstructures/fxblackvolsurface.hpp>
#include <qle/termstructures/fxsmilesection.hpp>
#include <qle/termstructures/fxvannavolgasmilesection.hpp>
#include <qle/termstructures/hazardspreadeddefaulttermstructure.hpp>
#include <qle/termstructures/immfraratehelper.hpp>
#include <qle/termstructures/interpolatedcorrelationcurve.hpp>
#include <qle/termstructures/interpolateddiscountcurve.hpp>
#include <qle/termstructures/interpolateddiscountcurve2.hpp>
#include <qle/termstructures/interpolatedyoycapfloortermpricesurface.hpp>
#include <qle/termstructures/kinterpolatedyoyoptionletvolatilitysurface.hpp>
#include <qle/termstructures/oibasisswaphelper.hpp>
#include <qle/termstructures/oiccbasisswaphelper.hpp>
#include <qle/termstructures/oisratehelper.hpp>
#include <qle/termstructures/optionletcurve.hpp>
#include <qle/termstructures/optionletstripper.hpp>
#include <qle/termstructures/optionletstripper1.hpp>
#include <qle/termstructures/optionletstripper2.hpp>
#include <qle/termstructures/optionletstripperwithatm.hpp>
#include <qle/termstructures/optionpricesurface.hpp>
#include <qle/termstructures/piecewiseatmoptionletcurve.hpp>
#include <qle/termstructures/piecewiseoptionletcurve.hpp>
#include <qle/termstructures/piecewiseoptionletstripper.hpp>
#include <qle/termstructures/pricecurve.hpp>
#include <qle/termstructures/pricetermstructure.hpp>
#include <qle/termstructures/pricetermstructureadapter.hpp>
#include <qle/termstructures/probabilitytraits.hpp>
#include <qle/termstructures/spreadedoptionletvolatility.hpp>
#include <qle/termstructures/spreadedsmilesection.hpp>
#include <qle/termstructures/spreadedswaptionvolatility.hpp>
#include <qle/termstructures/staticallycorrectedyieldtermstructure.hpp>
#include <qle/termstructures/strippedcpivolatilitystructure.hpp>
#include <qle/termstructures/strippedoptionletadapter.hpp>
#include <qle/termstructures/strippedoptionletadapter2.hpp>
#include <qle/termstructures/strippedyoyinflationoptionletvol.hpp>
#include <qle/termstructures/subperiodsswaphelper.hpp>
#include <qle/termstructures/survivalprobabilitycurve.hpp>
#include <qle/termstructures/swaptionvolatilityconverter.hpp>
#include <qle/termstructures/swaptionvolconstantspread.hpp>
#include <qle/termstructures/swaptionvolcube2.hpp>
#include <qle/termstructures/swaptionvolcubewithatm.hpp>
#include <qle/termstructures/tenorbasisswaphelper.hpp>
#include <qle/termstructures/yoyinflationcurveobservermoving.hpp>
#include <qle/termstructures/yoyinflationcurveobserverstatic.hpp>
#include <qle/termstructures/yoyinflationoptionletvolstripper.hpp>
#include <qle/termstructures/yoyoptionletvolatilitysurface.hpp>
#include <qle/termstructures/zeroinflationcurveobservermoving.hpp>
#include <qle/termstructures/zeroinflationcurveobserverstatic.hpp>
#include <qle/time/yearcounter.hpp>
#include <qle/version.hpp>
