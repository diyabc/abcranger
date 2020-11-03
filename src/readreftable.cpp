#define _REGEX_MAX_STACK_COUNT 5000L
#include <string>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "readreftable.hpp"
#include "tqdm.hpp"
#include <range/v3/all.hpp>

#include <stdio.h>
#ifdef __APPLE__
    #include <sys/uio.h>
#elif defined(_MSC_VER)
    #include <io.h>
#else
    #include <sys/io.h>
#endif

#if defined(_MSC_VER)
    #include <boost/regex.hpp>
    using namespace boost;
#else
    #include <regex>
    using namespace std;
#endif

// using namespace boost;
// using namespace ranges;

template<class A, class B> 
B readAndCast(std::ifstream& f) {
    A t;
    f.read(reinterpret_cast<char *>(&t),sizeof(A));
    return static_cast<B>(t);
} 

Reftable<MatrixXd> readreftable(std::string headerpath, std::string reftablepath, size_t N, bool quiet, std::string groups_opt) {
    ///////////////////////////////////////// read headers
    if (!quiet) std::cout << "///////////////////////////////////////// read headers" << std::endl;

    std::ifstream headerStream(headerpath, std::ios::in);
    if (headerStream.fail()){
        std::cout << "No header file, exiting" << std::endl;
        exit(1);
    }
    headerStream >> std::noskipws;
    const std::string hS(std::istream_iterator<char>{headerStream}, {});
    
    const regex scen_re(R"#(\bscenario\s+(\d+)\s+.*?\n((?:(?!(?:scenario|\n)).*?\n)+))#");
    sregex_token_iterator itscen(std::begin(hS), std::end(hS), scen_re, {1,2});
    //    print(itscen, {});

    auto nscenh = std::distance(itscen, {})/2;
    std::vector<std::string> scendesc(nscenh);
    sregex_token_iterator endregexp;
    
    auto it = itscen;
    while (it != endregexp) {
        const std::string num(*it++);
        const std::string desc(*it++);
        scendesc[stoi(num)-1] = desc;
    }
    //std::cout << scendesc[0];
    const regex nparamtot(R"#(historical parameters priors \((\d+)\D)#");
    smatch base_match;
    regex_search(std::begin(hS), std::end(hS),base_match,nparamtot);
    auto nparamtoth = stoi(base_match[1]);
//    std::cout << nparamtoth << std::endl;
    std::string reparamlistrestr = R"#(\bhistorical parameters priors.*?\n((?:\w+\W[^\n]*?\n){)#" + std::to_string(nparamtoth) + "})";
    const regex reparamlist(reparamlistrestr);
    smatch base_match2;
    regex_search(std::begin(hS), std::end(hS),base_match2,reparamlist);
    //std::cout << base_match[1] << std::endl;
    const std::string paramlistmatch = base_match2[1];
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[((?:\d|\.)+?)\W*,((?:\d|\.)+?)(?:,.+?)?\]?\n)#");
    sregex_token_iterator reparamit(std::begin(paramlistmatch),std::end(paramlistmatch), reparam, {1,2,3});
    it = reparamit;
    size_t reali = 1;
    std::map<std::string,size_t> paramdesc;
    while (it != endregexp) {
        const std::string param = *it++;
        const double mini = stod(*it++);
        const double maxi = stod(*it++);
        if (maxi != 0.0) {
            if ((maxi-mini)/maxi > 0.000001) {
                paramdesc[param] = reali;
                reali++;
            }
        }
    }
    size_t realparamtot = reali - 1;
    std::vector<std::vector<size_t>> parambyscenh(nscenh);
    const regex splitre(R"#(\W)#");
    for(auto i = 0; i < nscenh; i++) {
        sregex_token_iterator it(std::begin(scendesc[i]),std::end(scendesc[i]),splitre,-1);
        for(; it != endregexp; ++it) {
            const std::string term = *it;
            if (paramdesc.count(term) > 0) {
                const size_t nterm = paramdesc[term];
                if (find(std::begin(parambyscenh[i]),std::end(parambyscenh[i]),nterm)  == std::end(parambyscenh[i]))
                    parambyscenh[i].push_back(nterm);
            } 
        }
    }
    // for (auto scen : parambyscenh) {
    //     for(auto p : scen) std::cout << p << " ";
    //     std::cout << std::endl;       
    // }
    smatch base_match3;
    const regex restatsname(R"#(\n\s*\nscenario\s+)#");
    regex_search(std::begin(hS), std::end(hS),base_match3,restatsname);
    const std::string allstatsname = base_match3.suffix();
    const regex splitre2(R"#(\s+)#");
    std::vector<std::string> allcolspre;
    for(sregex_token_iterator it(std::begin(allstatsname),std::end(allstatsname),splitre2,-1); it != endregexp; it++)
        allcolspre.push_back(*it);

    if (!quiet) std::cout << "read headers done." << std::endl;
    headerStream.close();
     ///////////////////////////////////////// read reftable
    if (!quiet) std::cout << "///////////////////////////////////////// read reftable" << std::endl;


    std::ifstream reftableStream(reftablepath, std::ios::in|std::ios::binary);
    if (reftableStream.fail()){
        std::cout << "No Reftable, exiting" << std::endl;
        exit(1);
    }
    size_t realnrec = readAndCast<int,size_t>(reftableStream);
    // reftableStream.read(reinterpret_cast<char *>(realnrec_i),sizeof(realnr));
    size_t nrec = N > 0 ? std::min(realnrec,N) : realnrec;
    size_t nscen = readAndCast<int,size_t>(reftableStream);
    std::vector<size_t> nrecscen(nscen);
    for(auto& r : nrecscen) r = readAndCast<int,size_t>(reftableStream);
    std::vector<size_t> nparam(nscen);
    for(auto& r : nparam) r = readAndCast<int,size_t>(reftableStream);
    size_t nstat = readAndCast<int,size_t>(reftableStream);
    std::vector<std::string> params_names { std::begin(allcolspre), std::begin(allcolspre) + (allcolspre.size() - nstat) };
    std::vector<std::string> stats_names  { std::begin(allcolspre)+ (allcolspre.size() - nstat),  std::end(allcolspre) };
    std::vector<double> scenarios(nrec);
    size_t nmutparams = params_names.size() - realparamtot;
    MatrixXd params = MatrixXd::Constant(nrec,params_names.size(),NAN);
    // for(auto r : statsname) std::cout << r << std::endl;
    MatrixXd stats = MatrixXd::Constant(nrec,nstat,NAN);
    // DataDouble data(stats,statsname,nrec,nstat + 1);
    // bool hasError;
    std::vector<int> groups;
    if (groups_opt.length() > 0) {
        groups = std::vector<int>(nscen,0);
        auto groupstr = groups_opt
            | ranges::views::split(';')
            | ranges::views::transform([](auto&& s) { return s 
                | ranges::views::split(',') 
                | ranges::views::transform([](auto&& si){ return std::stoi(si | ranges::to<std::string>); })
                | ranges::to<std::vector>; 
                    }
                )
            | ranges::views::enumerate
            | ranges::to<std::vector>;
        for(auto&& s: groupstr) 
            for(auto&& si: s.second)
                groups[si-1] = static_cast<double>(s.first + 1);
        

        std::vector<size_t> new_nrecscen = std::vector<size_t>(groupstr.size());
        for(auto i = 0; i < nscen; i++) 
            if (groups[i] > 0) new_nrecscen[groups[i] - 1] += nrecscen[i];
        nrecscen = new_nrecscen;
        // myread.scenarios |= actions::transform([&groups](auto&& si){ return groups[si-1]; });        
    }
    size_t rcount = 0;
    tqdm bar;
    for(auto i = 0; i < nrec; i++) {
        bool to_skip = false;
        if (!quiet) {
            if (isatty(fileno(stdin))) 
                bar.progress(i,nrec);
            else 
                if ((i + 1) % 500 == 0)
                     std::cout << "read " << (i + 1) << "/" << nrec << std::endl;
        }
        size_t scen = readAndCast<int,size_t>(reftableStream);
        if (groups.size() > 0) {
            if (groups[scen-1] == 0) to_skip = true;
            else scenarios[rcount] = static_cast<double>(groups[scen-1]);
        } else scenarios[rcount] = static_cast<double>(scen);
//        reftableStream.read(reinterpret_cast<char *>(&scen),4);

        // data.set(nstat,i,static_cast<double>(scen),hasError);
        scen--;
        std::vector<float> lparam(nparam[scen]);
        for(auto& r: lparam) {
            reftableStream.read(reinterpret_cast<char *>(&r),sizeof(r));
        }
        if (!to_skip) {
            for(auto j = 0; j < parambyscenh[scen].size(); j++)
                params(rcount,parambyscenh[scen][j] - 1) = lparam[j];
            for(auto j = nparam[scen] - nmutparams; j < nparam[scen]; j++)
                params(rcount,j) = lparam[j];
        }
        for(auto j = 0; j < nstat; j++) {
            float r;
            reftableStream.read(reinterpret_cast<char *>(&r),4);
            // col * num_rows + row
            if (!to_skip) stats(rcount,j) = r;
            // data.set(j,i,r,hasError);
        }
        if (!to_skip) rcount++;
    }
    if (rcount < nrec) {
        stats.conservativeResize(rcount,NoChange);
        params.conservativeResize(rcount,NoChange);
        scenarios.resize(rcount);
    }
    reftableStream.close();
    if (!quiet) std::cout << std::endl << "read reftable done." << std::endl;
    Reftable reftable(rcount,nrecscen, nparam, params_names, stats_names, stats,params, scenarios);
    return reftable;
}

#define READSCEN_BUFFER_SIZE 1000
Reftable<MatrixXd> readreftable_scen(std::string headerpath, std::string reftablepath, size_t sel_scen, size_t N, bool quiet) {
    ///////////////////////////////////////// read headers
    if (!quiet) std::cout << "///////////////////////////////////////// read headers" << std::endl;
    
    std::ifstream headerStream(headerpath,std::ios::in);
    if (headerStream.fail()){
        std::cout << "No header file, exiting" << std::endl;
        exit(1);
    }
    headerStream >> std::noskipws;
    const std::string hS(std::istream_iterator<char>{headerStream}, {});
    
    const regex scen_re(R"#(\bscenario\s+(\d+)\s+.*?\n((?:(?!(?:scenario|\n)).*?\n)+))#");
    sregex_token_iterator itscen(std::begin(hS), end(hS), scen_re, {1,2});
    //    print(itscen, {});

    auto nscenh = distance(itscen, {})/2;
    std::vector<std::string> scendesc(nscenh);
    sregex_token_iterator endregexp;
    
    auto it = itscen;
    while (it != endregexp) {
        const std::string num(*it++);
        const std::string desc(*it++);
        scendesc[stoi(num)-1] = desc;
    }
    //std::cout << scendesc[0];
    const regex nparamtot(R"#(historical parameters priors \((\d+)\D)#");
    smatch base_match;
    regex_search(std::begin(hS), end(hS),base_match,nparamtot);
    auto nparamtoth = stoi(base_match[1]);
//    std::cout << nparamtoth << std::endl;
    std::string reparamlistrestr = R"#(\bhistorical parameters priors.*?\n((?:\w+\W[^\n]*?\n){)#" + std::to_string(nparamtoth) + "})";
    const regex reparamlist(reparamlistrestr);
    smatch base_match2;
    regex_search(std::begin(hS), end(hS),base_match2,reparamlist);
    //std::cout << base_match[1] << std::endl;
    const std::string paramlistmatch = base_match2[1];
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[((?:\d|\.)+?)\W*,((?:\d|\.)+?)(?:,.+?)?\]?\n)#");
    sregex_token_iterator reparamit(std::begin(paramlistmatch),std::end(paramlistmatch), reparam, {1,2,3});
    it = reparamit;
    size_t reali = 1;
    std::map<std::string,size_t> paramdesc;
    while (it != endregexp) {
        const std::string param = *it++;
        const double mini = stod(*it++);
        const double maxi = stod(*it++);
        if (maxi != 0.0) {
            if ((maxi-mini)/maxi > 0.000001) {
                paramdesc[param] = reali;
                reali++;
            }
        }
    }
    size_t realparamtot = reali - 1;
    std::vector<std::vector<size_t>> parambyscenh(nscenh);
    const regex splitre(R"#(\W)#");
    for(auto i = 0; i < nscenh; i++) {
        sregex_token_iterator it(std::begin(scendesc[i]),std::end(scendesc[i]),splitre,-1);
        for(; it != endregexp; ++it) {
            const std::string term = *it;
            if (paramdesc.count(term) > 0) {
                const size_t nterm = paramdesc[term];
                if (find(std::begin(parambyscenh[i]),std::end(parambyscenh[i]),nterm)  == std::end(parambyscenh[i]))
                    parambyscenh[i].push_back(nterm);
            } 
        }
    }
    // for (auto scen : parambyscenh) {
    //     for(auto p : scen) std::cout << p << " ";
    //     std::cout << std::endl;       
    // }
    smatch base_match3;
    const regex restatsname(R"#(\n\s*\nscenario\s+)#");
    regex_search(std::begin(hS), end(hS),base_match3,restatsname);
    const std::string allstatsname = base_match3.suffix();
    const regex splitre2(R"#(\s+)#");
    std::vector<std::string> allcolspre;
    for(sregex_token_iterator it(std::begin(allstatsname),std::end(allstatsname),splitre2,-1); it != endregexp; it++)
        allcolspre.push_back(*it);

    if (!quiet) std::cout << "read headers done." << std::endl;
    headerStream.close();
     ///////////////////////////////////////// read reftable
    if (!quiet) std::cout << "///////////////////////////////////////// read reftable" << std::endl;


    std::ifstream reftableStream(reftablepath,std::ios::in|std::ios::binary);
    if (reftableStream.fail()){
        std::cout << "No Reftable, exiting" << std::endl;
        exit(1);
    }
    size_t realnrec = readAndCast<int,size_t>(reftableStream);
    // reftableStream.read(reinterpret_cast<char *>(realnrec_i),sizeof(realnr));
    if (N == 0) N = realnrec;
    size_t nrec = realnrec;
    size_t nscen = readAndCast<int,size_t>(reftableStream);
    std::vector<size_t> nrecscen(nscen);
    for(auto& r : nrecscen) r = readAndCast<int,size_t>(reftableStream);
    std::vector<size_t> nparam(nscen);
    for(auto& r : nparam) r = readAndCast<int,size_t>(reftableStream);
    size_t nstat = readAndCast<int,size_t>(reftableStream);
    std::vector<std::string> params_names { std::begin(allcolspre), std::begin(allcolspre) + (allcolspre.size() - nstat) };
    std::vector<std::string> stats_names  { std::begin(allcolspre)+ (allcolspre.size() - nstat),  std::end(allcolspre) };
    std::vector<double> scenarios(nrec);
    size_t nmutparams = params_names.size() - realparamtot;
    MatrixXd params = MatrixXd::Constant(0,params_names.size(),NAN);
    // for(auto r : statsname) std::cout << r << std::endl;
    MatrixXd stats = MatrixXd::Constant(0,nstat,NAN);
    // DataDouble data(stats,statsname,nrec,nstat + 1);
    // bool hasError;
    size_t ncount = 0;
    tqdm bar;
    for(auto i = 0; (i < nrec) && (ncount < N); i++) {
        if (!quiet) {
            if (isatty(fileno(stdin))) 
                bar.progress(i,nrec);
            else 
                if ((i + 1) % 500 == 0)
                     std::cout << "read " << (i + 1) << "/" << nrec << std::endl;
        }
        size_t scen = readAndCast<int,size_t>(reftableStream);
//        reftableStream.read(reinterpret_cast<char *>(&scen),4);
        bool matched = (scen == sel_scen);
        if (matched && ncount == stats.rows()) {
            stats.conservativeResize(stats.rows() + READSCEN_BUFFER_SIZE,NoChange);
            params.conservativeResize(params.rows() + READSCEN_BUFFER_SIZE,NoChange);
        }
        scenarios[i] = static_cast<double>(scen);
        // data.set(nstat,i,static_cast<double>(scen),hasError);
        scen--;
        std::vector<float> lparam(nparam[scen]);
        for(auto& r: lparam) {
            reftableStream.read(reinterpret_cast<char *>(&r),sizeof(r));
        }
        if (matched) {
            for(auto j = 0; j < parambyscenh[scen].size(); j++)
                params(ncount,parambyscenh[scen][j] - 1) = lparam[j];
            for(auto j = nparam[scen] - nmutparams; j < nparam[scen]; j++)
                params(ncount,j) = lparam[j];
        }
        for(auto j = 0; j < nstat; j++) {
            float r;
            reftableStream.read(reinterpret_cast<char *>(&r),4);
            // col * num_rows + row
            if (matched) stats(ncount,j) = r;
            // data.set(j,i,r,hasError);
        }
        if (matched) ncount++;
    }

    reftableStream.close();
    if (!quiet) std::cout << std::endl << "read reftable done. " << ncount << " samples of total " << nrec << ", " << N << " asked and " << nrecscen[sel_scen-1] << " avalaible." << std::endl;
    if (!quiet && (N > nrecscen[sel_scen-1])) std::cout << "Warning : asked for more samples than available." << std::endl;
    stats.conservativeResize(ncount,NoChange);
    params.conservativeResize(ncount,NoChange);
   std::vector<size_t> uniqrec  = { nrecscen[sel_scen-1] };
    Reftable reftable(ncount,uniqrec, nparam, params_names, stats_names, stats,params, scenarios);
    return reftable;
}