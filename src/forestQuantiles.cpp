#include "forestQuantiles.hpp"
#include "threadpool.hpp"

#include <range/v3/all.hpp>

constexpr std::size_t operator "" _z ( unsigned long long n )
    { return n; }

using namespace ranges;

CPP_template(class Rng) (requires range<Rng>)
auto indirect_lambda(const Rng& v) {
    return [&v](const auto& i){ return v[i]; };
}

template<class Rng>
auto less_rng = std::less<range_value_t<Rng>>();

CPP_template(class Rng, class F = decltype(less_rng<Rng>)) (requires range<Rng>)
auto indirect_comparison_lambda (const Rng& v, const F& f = less_rng<Rng>) {
    return [&v,&f](const auto& a, const auto& b) { return f(v[a],v[b]); };
}


std::vector<double> forestQuantiles(const std::vector<double> &origObs,
                                    const std::vector<double> &origWeights,
                                    const std::vector<double> &quantiles)
{
    size_t n = origObs.size();
    std::vector<double> quant(quantiles.size());
    const auto& ord = views::iota(0_z,n) 
        | to_vector 
        | actions::sort(indirect_comparison_lambda(origObs));

    const auto& obs = ord | views::transform(indirect_lambda(origObs));
    const auto& weights = ord | views::transform(indirect_lambda(origWeights));

    std::vector<double> cumweights = weights | views::partial_sum | to<std::vector>();
    double lastcum = cumweights[n-1];
    cumweights |= actions::transform([&lastcum](const auto& d){ return d/lastcum; });
    for(auto qc = 0; qc < quantiles.size(); qc++) {
        auto wc = ranges::count_if(cumweights,[&quantiles,qc](auto v){ return v < quantiles[qc]; });
        if (wc <= 1) {
            quant[qc] = obs[0];
        }
        else {
            auto quantmax = obs[wc];
            auto quantmin = obs[wc-1];
            auto weightmax = cumweights[wc];
            auto weightmin = cumweights[wc-1];
            double factor;
            if (weightmax-weightmin < 1e-10)
                factor = 0.5;
            else
                factor = (quantiles[qc] - weightmin)/(weightmax-weightmin);
            quant[qc] = quantmin + factor * (quantmax - quantmin); 
        }
    }
    return quant;
}

std::vector<std::vector<double>> forestQuantiles_b(const std::vector<double> &obs,
                                                   const std::vector<std::vector<double>> &weights,
                                                   const std::vector<double> &asked)
{
    std::vector<std::vector<double>> quants(weights.size());
    ThreadPool::ParallelFor<size_t>(0, weights.size(), [&](auto i) {
        quants[i] = forestQuantiles(obs, weights[i], asked);
    });
    return quants;
}

double median(std::vector<double> vec)
{
    size_t size = vec.size();
    if (size == 0) return 0;
    else {
        sort(vec.begin(), vec.end());
        size_t mid = size/2;
        return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
        }
}
