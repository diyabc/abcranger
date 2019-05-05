#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <boost/random.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/weighted_tail_quantile.hpp>


using namespace boost::accumulators;

TEST_CASE("Quantiles from weighted params") {
    // tolerance in %
    double epsilon = 0.01;

std::size_t n = 100000; // number of MC steps
std::size_t c =  20000; // cache size

double mu1 = 1.0;
double mu2 = -1.0;
boost::lagged_fibonacci607 rng;
boost::normal_distribution<> mean_sigma1(mu1,1);
boost::normal_distribution<> mean_sigma2(mu2,1);
boost::variate_generator<boost::lagged_fibonacci607&, boost::normal_distribution<> > normal1(rng, mean_sigma1);
boost::variate_generator<boost::lagged_fibonacci607&, boost::normal_distribution<> > normal2(rng, mean_sigma2);

accumulator_set<double, stats<tag::weighted_tail_quantile<right> >, double>accumulator_set<double, stats<tag::weighted_tail_quantile<right> >, double>
    acc1(right_tail_cache_size = c);

accumulator_set<double, stats<tag::weighted_tail_quantile<left> >, double>
    acc2(left_tail_cache_size = c);

for (std::size_t i = 0; i < n; ++i)
{
    double sample1 = normal1();
    double sample2 = normal2();
    acc1(sample1, weight = std::exp(-mu1 * (sample1 - 0.5 * mu1)));
    acc2(sample2, weight = std::exp(-mu2 * (sample2 - 0.5 * mu2)));
}
CHECK( quantile(acc1, quantile_probability = 0.975) ==  Approx(1.959963).epsilon(epsilon));
CHECK( quantile(acc1, quantile_probability = 0.999) ==  Approx(3.090232).epsilon(epsilon));
CHECK( quantile(acc2, quantile_probability  = 0.025) == Approx(-1.959963).epsilon(epsilon) );
CHECK( quantile(acc2, quantile_probability  = 0.001) == Approx(-3.090232).epsilon(epsilon) );
}