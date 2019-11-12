#pragma once
#include <range/v3/all.hpp>
using namespace ranges;

// Cut/pasted from R and cleaned to pure C/C++
double psmirnov2x(double x, size_t mo, size_t no)
{
    double md, nd, q, w;
    size_t i, j,m,n;
    m = mo;
    n = no;
	
    if(m > n) {
		i = n; n = m; m = i;
    }
    md = static_cast<double>(m);
    nd = static_cast<double>(n);
    /*
	 q has 0.5/mn added to ensure that rounding error doesn't
	 turn an equality into an inequality, eg abs(1/2-4/5)>3/10 
	 
	 */
    q = (0.5 + floor(x * md * nd - 1e-7)) / (md * nd);
    std::vector<double> u(n + 1);
	
    for(j = 0; j <= n; j++) {
		u[j] = ((j / nd) > q) ? 0 : 1;
    }
    for(i = 1; i <= m; i++) {
		w = static_cast<double>(i) / (static_cast<double>(i + n));
		if((i / md) > q)
			u[0] = 0;
		else
			u[0] = w * u[0];
		for(j = 1; j <= n; j++) {
			if(fabs(i / md - j / nd) > q) 
				u[j] = 0;
			else
				u[j] = w * u[j] + u[j - 1];
		}
    }
    return u[n];
}

// simple application from R code of ks.test
// w <- c(x, y)
// z <- cumsum(ifelse(order(w) <= n.x, 1 / n.x, - 1 / n.y))
// STATISTIC <- max(abs(z))
template<class T>
double KSTest(std::vector<T> x, std::vector<T> y) {
    double xn = 1.0/static_cast<double>(x.size());
    double yn = 1.0/static_cast<double>(y.size());
    std::vector<std::pair<size_t,T>> w = views::concat(x,y) 
        | views::enumerate
        | to<std::vector>();
    w |= actions::sort([](auto a, auto b){ return a.second < b.second; });
    auto D = w
        | views::transform([xs=x.size(),xn,yn](auto wii){ return wii.first < xs? xn : -yn; })
        | views::partial_sum
        | views::transform([](auto d){ return std::abs(d); });
    return ranges::max(D);
}

/// "Vieille" version C++11
// #include <vector>
// #include <algorithm>
// #include <numeric>

// template<class T>
// double KSTest(std::vector<T> x, std::vector<T> y) {
//     std::vector<T> w(x.size() + y.size());
//     double xn = 1.0/static_cast<double>(x.size());
//     double yn = 1.0/static_cast<double>(y.size());
//     auto middle = std::next(w.begin(),x.size());
//     std::vector<size_t> wi(w.size());
//     std::iota(wi.begin(),wi.end(),0);
//     std::copy(x.begin(),x.end(),w.begin());
//     std::copy(y.begin(),y.end(),middle);
//     std::sort(wi.begin(),wi.end(),[&w](size_t a, size_t b){ return w[a] > w[b]; });
//     std::vector<double> cumsum(w.size());
//     std::transform(wi.begin(),wi.end(),cumsum.begin(),[xs=x.size(),xn,yn](size_t wii){ return wii < xs? xn : -yn; });
//     std::partial_sum(cumsum.begin(),cumsum.end(),cumsum.begin());
//     return std::abs(*std::max_element(cumsum.begin(),cumsum.end(),[](double a, double b) { return std::abs(a) < std::abs(b); }));
// }
