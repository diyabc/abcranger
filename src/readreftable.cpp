#define _REGEX_MAX_STACK_COUNT 5000
// #if defined(_MSC_VER)
// #include <boost/regex.hpp>
// using namespace boost;
// #else
#include <regex>
// #endif
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

// using namespace boost;
// using namespace ranges;

template<class A, class B> 
B readAndCast(ifstream& f) {
    A t;
    f.read(reinterpret_cast<char *>(&t),sizeof(A));
    return static_cast<B>(t);
} 

using namespace std;

Reftable<MatrixXd> readreftable(string headerpath, string reftablepath, size_t N, bool quiet, std::string groups_opt) {
    ///////////////////////////////////////// read headers
    if (!quiet) cout << "///////////////////////////////////////// read headers" << endl;

    ifstream headerStream(headerpath,ios::in);
    if (headerStream.fail()){
        cout << "No header file, exiting" << endl;
        exit(1);
    }
    headerStream >> noskipws;
    const std::string hS(istream_iterator<char>{headerStream}, {});
    
    const regex scen_re(R"#(\bscenario\s+(\d+)\s+.*?\n((?:(?!(?:scenario|\n)).*?\n)+))#");
    sregex_token_iterator itscen(begin(hS), end(hS), scen_re, {1,2});
    //    print(itscen, {});

    auto nscenh = distance(itscen, {})/2;
    vector<string> scendesc(nscenh);
    sregex_token_iterator endregexp;
    
    auto it = itscen;
    while (it != endregexp) {
        const string num(*it++);
        const string desc(*it++);
        scendesc[stoi(num)-1] = desc;
    }
    //cout << scendesc[0];
    const regex nparamtot(R"#(historical parameters priors \((\d+)\D)#");
    smatch base_match;
    regex_search(begin(hS), end(hS),base_match,nparamtot);
    auto nparamtoth = stoi(base_match[1]);
//    cout << nparamtoth << endl;
    string reparamlistrestr = R"#(\bhistorical parameters priors.*?\n((?:\w+\W[^\n]*?\n){)#" + to_string(nparamtoth) + "})";
    const regex reparamlist(reparamlistrestr);
    smatch base_match2;
    regex_search(begin(hS), end(hS),base_match2,reparamlist);
    //cout << base_match[1] << endl;
    const string paramlistmatch = base_match2[1];
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[((?:\d|\.)+?)\W*,((?:\d|\.)+?)(?:,.+?)?\]?\n)#");
    sregex_token_iterator reparamit(begin(paramlistmatch),end(paramlistmatch), reparam, {1,2,3});
    it = reparamit;
    size_t reali = 1;
    map<string,size_t> paramdesc;
    while (it != endregexp) {
        const string param = *it++;
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
    vector<vector<size_t>> parambyscenh(nscenh);
    const regex splitre(R"#(\W)#");
    for(auto i = 0; i < nscenh; i++) {
        sregex_token_iterator it(begin(scendesc[i]),end(scendesc[i]),splitre,-1);
        for(; it != endregexp; ++it) {
            const string term = *it;
            if (paramdesc.count(term) > 0) {
                const size_t nterm = paramdesc[term];
                if (find(parambyscenh[i].begin(),parambyscenh[i].end(),nterm)  == parambyscenh[i].end())
                    parambyscenh[i].push_back(nterm);
            } 
        }
    }
    // for (auto scen : parambyscenh) {
    //     for(auto p : scen) cout << p << " ";
    //     cout << endl;       
    // }
    smatch base_match3;
    const regex restatsname(R"#(\n\s*\nscenario\s+)#");
    regex_search(begin(hS), end(hS),base_match3,restatsname);
    const string allstatsname = base_match3.suffix();
    const regex splitre2(R"#(\s+)#");
    vector<string> allcolspre;
    for(sregex_token_iterator it(allstatsname.begin(),allstatsname.end(),splitre2,-1); it != endregexp; it++)
        allcolspre.push_back(*it);

    if (!quiet) cout << "read headers done." << endl;
    headerStream.close();
     ///////////////////////////////////////// read reftable
    if (!quiet) cout << "///////////////////////////////////////// read reftable" << endl;


    ifstream reftableStream(reftablepath,ios::in|ios::binary);
    if (reftableStream.fail()){
        cout << "No Reftable, exiting" << endl;
        exit(1);
    }
    size_t realnrec = readAndCast<int,size_t>(reftableStream);
    // reftableStream.read(reinterpret_cast<char *>(realnrec_i),sizeof(realnr));
    size_t nrec = N > 0 ? min(realnrec,N) : realnrec;
    size_t nscen = readAndCast<int,size_t>(reftableStream);
    vector<size_t> nrecscen(nscen);
    for(auto& r : nrecscen) r = readAndCast<int,size_t>(reftableStream);
    vector<size_t> nparam(nscen);
    for(auto& r : nparam) r = readAndCast<int,size_t>(reftableStream);
    size_t nstat = readAndCast<int,size_t>(reftableStream);
    vector<string> params_names { allcolspre.begin(), allcolspre.begin() + (allcolspre.size() - nstat) };
    vector<string> stats_names  { allcolspre.begin()+ (allcolspre.size() - nstat),  allcolspre.end() };
    vector<double> scenarios(nrec);
    size_t nmutparams = params_names.size() - realparamtot;
    MatrixXd params = MatrixXd::Constant(nrec,params_names.size(),NAN);
    // for(auto r : statsname) cout << r << endl;
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
                | ranges::views::transform([](auto&& si){ return si | ranges::to<std::string>; }); 
                    }
                )
            | ranges::views::enumerate
            | ranges::to<std::vector>;
        for(auto&& s: groupstr) 
            for(auto&& si: s.second)
                groups[std::stoi(si)-1] = static_cast<double>(s.first + 1);
        

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
                cout << "read " << (i + 1) << "/" << nrec << std::endl;
        }
        size_t scen = readAndCast<int,size_t>(reftableStream);
        if (groups.size() > 0) {
            if (groups[scen-1] == 0) to_skip = true;
            else scenarios[rcount] = static_cast<double>(groups[scen-1]);
        } else scenarios[rcount] = static_cast<double>(scen);
//        reftableStream.read(reinterpret_cast<char *>(&scen),4);

        // data.set(nstat,i,static_cast<double>(scen),hasError);
        scen--;
        vector<float> lparam(nparam[scen]);
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
    if (!quiet) cout << endl << "read reftable done." << endl;
    Reftable reftable(rcount,nrecscen, nparam, params_names, stats_names, stats,params, scenarios);
    return reftable;
}

#define READSCEN_BUFFER_SIZE 1000
Reftable<MatrixXd> readreftable_scen(string headerpath, string reftablepath, size_t sel_scen, size_t N, bool quiet) {
    ///////////////////////////////////////// read headers
    if (!quiet) cout << "///////////////////////////////////////// read headers" << endl;
    
    ifstream headerStream(headerpath,ios::in);
    if (headerStream.fail()){
        cout << "No header file, exiting" << endl;
        exit(1);
    }
    headerStream >> noskipws;
    const std::string hS(istream_iterator<char>{headerStream}, {});
    
    const regex scen_re(R"#(\bscenario\s+(\d+)\s+.*?\n((?:(?!(?:scenario|\n)).*?\n)+))#");
    sregex_token_iterator itscen(begin(hS), end(hS), scen_re, {1,2});
    //    print(itscen, {});

    auto nscenh = distance(itscen, {})/2;
    vector<string> scendesc(nscenh);
    sregex_token_iterator endregexp;
    
    auto it = itscen;
    while (it != endregexp) {
        const string num(*it++);
        const string desc(*it++);
        scendesc[stoi(num)-1] = desc;
    }
    //cout << scendesc[0];
    const regex nparamtot(R"#(historical parameters priors \((\d+)\D)#");
    smatch base_match;
    regex_search(begin(hS), end(hS),base_match,nparamtot);
    auto nparamtoth = stoi(base_match[1]);
//    cout << nparamtoth << endl;
    string reparamlistrestr = R"#(\bhistorical parameters priors.*?\n((?:\w+\W[^\n]*?\n){)#" + to_string(nparamtoth) + "})";
    const regex reparamlist(reparamlistrestr);
    smatch base_match2;
    regex_search(begin(hS), end(hS),base_match2,reparamlist);
    //cout << base_match[1] << endl;
    const string paramlistmatch = base_match2[1];
    const regex reparam(R"#((\w+)\W+\w\W+\w\w\[((?:\d|\.)+?)\W*,((?:\d|\.)+?)(?:,.+?)?\]?\n)#");
    sregex_token_iterator reparamit(begin(paramlistmatch),end(paramlistmatch), reparam, {1,2,3});
    it = reparamit;
    size_t reali = 1;
    map<string,size_t> paramdesc;
    while (it != endregexp) {
        const string param = *it++;
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
    vector<vector<size_t>> parambyscenh(nscenh);
    const regex splitre(R"#(\W)#");
    for(auto i = 0; i < nscenh; i++) {
        sregex_token_iterator it(begin(scendesc[i]),end(scendesc[i]),splitre,-1);
        for(; it != endregexp; ++it) {
            const string term = *it;
            if (paramdesc.count(term) > 0) {
                const size_t nterm = paramdesc[term];
                if (find(parambyscenh[i].begin(),parambyscenh[i].end(),nterm)  == parambyscenh[i].end())
                    parambyscenh[i].push_back(nterm);
            } 
        }
    }
    // for (auto scen : parambyscenh) {
    //     for(auto p : scen) cout << p << " ";
    //     cout << endl;       
    // }
    smatch base_match3;
    const regex restatsname(R"#(\n\s*\nscenario\s+)#");
    regex_search(begin(hS), end(hS),base_match3,restatsname);
    const string allstatsname = base_match3.suffix();
    const regex splitre2(R"#(\s+)#");
    vector<string> allcolspre;
    for(sregex_token_iterator it(allstatsname.begin(),allstatsname.end(),splitre2,-1); it != endregexp; it++)
        allcolspre.push_back(*it);

    if (!quiet) cout << "read headers done." << endl;
    headerStream.close();
     ///////////////////////////////////////// read reftable
    if (!quiet) cout << "///////////////////////////////////////// read reftable" << endl;


    ifstream reftableStream(reftablepath,ios::in|ios::binary);
    if (reftableStream.fail()){
        cout << "No Reftable, exiting" << endl;
        exit(1);
    }
    size_t realnrec = readAndCast<int,size_t>(reftableStream);
    // reftableStream.read(reinterpret_cast<char *>(realnrec_i),sizeof(realnr));
    if (N == 0) N = realnrec;
    size_t nrec = realnrec;
    size_t nscen = readAndCast<int,size_t>(reftableStream);
    vector<size_t> nrecscen(nscen);
    for(auto& r : nrecscen) r = readAndCast<int,size_t>(reftableStream);
    vector<size_t> nparam(nscen);
    for(auto& r : nparam) r = readAndCast<int,size_t>(reftableStream);
    size_t nstat = readAndCast<int,size_t>(reftableStream);
    vector<string> params_names { allcolspre.begin(), allcolspre.begin() + (allcolspre.size() - nstat) };
    vector<string> stats_names  { allcolspre.begin()+ (allcolspre.size() - nstat),  allcolspre.end() };
    vector<double> scenarios(nrec);
    size_t nmutparams = params_names.size() - realparamtot;
    MatrixXd params = MatrixXd::Constant(0,params_names.size(),NAN);
    // for(auto r : statsname) cout << r << endl;
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
                cout << "read " << (i + 1) << "/" << nrec << std::endl;
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
        vector<float> lparam(nparam[scen]);
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
    if (!quiet) cout << endl << "read reftable done. " << ncount << " samples of total " << nrec << ", " << N << " asked and " << nrecscen[sel_scen-1] << " avalaible." << endl;
    if (!quiet && (N > nrecscen[sel_scen-1])) cout << "Warning : asked for more samples than available." << endl;
    stats.conservativeResize(ncount,NoChange);
    params.conservativeResize(ncount,NoChange);
   std::vector<size_t> uniqrec  = { nrecscen[sel_scen-1] };
    Reftable reftable(ncount,uniqrec, nparam, params_names, stats_names, stats,params, scenarios);
    return reftable;
}