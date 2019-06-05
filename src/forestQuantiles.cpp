#include "forestQuantiles.hpp"
#include "various.hpp"

#include <range/v3/all.hpp>

using namespace ranges;

template<class Rng, CONCEPT_REQUIRES_(Range<Rng>())>
auto indirect_lambda(const Rng& v) {
    return [&v](const auto& i){ return v[i]; };
}

template<class Rng>
auto less_rng = std::less<range_value_type_t<Rng>>();

template<class Rng, class F = decltype(less_rng<Rng>), CONCEPT_REQUIRES_(Range<Rng>())>
auto indirect_comparison_lambda (const Rng& v, const F& f = less_rng<Rng>) {
    return [&v,&f](const auto& a, const auto& b) { return f(v[a],v[b]); };
}

std::vector<double> forestQuantiles(const std::vector<double> &origObs,
                                    const std::vector<double> &origWeights,
                                    const std::vector<double> &asked)
{
    size_t n = origObs.size();
    std::vector<std::vector<double>> quant(asked.size(),std::vector<double>(n));
    const auto& ord = view::iota(0_z,n) 
        | to_vector 
        | action::sort(indirect_comparison_lambda(origObs));

    const auto& obs = ord | view::transform(indirect_lambda(origObs));
    const auto& weights = ord | view::transform(indirect_lambda(origObs));

    std::vector<double> cumweights = weights | view::partial_sum;
    double lastcum = cumweights[n-1];
    cumweights |= action::transform([&lastcum](const auto& d){ return d/lastcum; });

    for(auto qc : asked) {
        
    }

}
