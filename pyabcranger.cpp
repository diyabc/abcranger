#include "cxxopts.hpp"
#include "EstimParam.hpp"
#include "ModelChoice.hpp"
#include "forestQuantiles.hpp"
#include "cxxopts.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <list>

namespace py = pybind11;

// py::scoped_ostream_redirect output;

const cxxopts::ParseResult parseopt(std::string stropts) {
    std::istringstream ss("abcranger " + stropts);
    std::string arg;
    std::list<std::string> ls;
    std::vector<char*> v;
    while (ss >> arg)
    {
        ls.push_back(arg); 
        v.push_back(const_cast<char*>(ls.back().c_str()));
    }
    v.push_back(0);  // need terminating null pointer

    size_t nref;
    cxxopts::Options options("", " - ABC Random Forest - Model choice or parameter estimation command line options");

    options
        .positional_help("[optional args]")
        .show_positional_help();

    options.add_options()
        ("o,output","Prefix output (modelchoice_out or estimparam_out by default)",cxxopts::value<std::string>())
        ("n,nref","Number of samples, 0 means all",cxxopts::value<size_t>()->default_value("0"))
        ("m,minnodesize","Minimal node size. 0 means 1 for classification or 5 for regression",cxxopts::value<size_t>()->default_value("0"))
        ("t,ntree","Number of trees",cxxopts::value<size_t>()->default_value("500"))
        ("j,threads","Number of threads, 0 means all",cxxopts::value<size_t>()->default_value("0"))
        ("s,seed","Seed, generated by default",cxxopts::value<size_t>()->default_value("0"))
        ("c,noisecolumns","Number of noise columns",cxxopts::value<size_t>()->default_value("5"))
        ("nolinear","Disable LDA for model choice or PLS for parameter estimation")
        ("plsmaxvar","Percentage of maximum explained Y-variance for retaining pls axis",cxxopts::value<double>()->default_value("0.9"))
        ("chosenscen","Chosen scenario (mandatory for parameter estimation)", cxxopts::value<size_t>())
        ("noob","number of oob testing samples (mandatory for parameter estimation)",cxxopts::value<size_t>())
        ("parameter","name of the parameter of interest (mandatory for parameter estimation)",cxxopts::value<std::string>())
        ("g,groups","Groups of models",cxxopts::value<std::string>())
        ("help", "Print help")
        ;
    int argc = (int)v.size()-1;
    char **argv = v.data();

    auto res = options.parse(argc,argv);
    return res;
}

ModelChoiceResults ModelChoice_fun_py(Reftable &reftable,
                                   std::vector<double> statobs,
                                   std::string options,
                                   bool quiet = false) {
    py::scoped_ostream_redirect stream(
        std::cout,                               // std::ostream&
        py::module::import("sys").attr("stdout") // Python output
    );
    py::gil_scoped_release release;
    return ModelChoice_fun(reftable,statobs,parseopt(options),quiet);
}

EstimParamResults EstimParam_fun_py(Reftable &reftable,
                                   std::vector<double> statobs,
                                   std::string options,
                                   bool quiet = false,
                                   bool weights = false) {
    py::scoped_ostream_redirect stream(
        std::cout,                               // std::ostream&
        py::module::import("sys").attr("stdout") // Python output
    );
    py::gil_scoped_release release;
    return EstimParam_fun(reftable,statobs,parseopt(options),quiet,weights);
}

using namespace Eigen;

PYBIND11_MODULE(pyabcranger, m) { 
    py::class_<Reftable>(m,"reftable")
        .def(py::init<int,
                       std::vector<size_t>,
                       std::vector<size_t>,
                       std::vector<string>,
                       std::vector<string>,
                       MatrixXd,
                       MatrixXd,
                       std::vector<double>>());
    py::class_<ModelChoiceResults>(m,"modelchoice_results")
        .def_readwrite("confusion_matrix",&ModelChoiceResults::confusion_matrix)
        .def_readwrite("variable_importance",&ModelChoiceResults::variable_importance)
        .def_readwrite("ntree_oob_error",&ModelChoiceResults::ntree_oob_error)
        .def_readwrite("predicted_model",&ModelChoiceResults::predicted_model)
        .def_readwrite("votes",&ModelChoiceResults::votes)
        .def_readwrite("post_proba",&ModelChoiceResults::post_proba);
    py::class_<EstimParamResults>(m,"estimparam_results")
        .def_readwrite("plsvar",&EstimParamResults::plsvar)
        .def_readwrite("plsweights",&EstimParamResults::plsweights)
        .def_readwrite("variable_importance",&EstimParamResults::variable_importance)
        .def_readwrite("ntree_oob_error",&EstimParamResults::ntree_oob_error)
        .def_readwrite("values_weights",&EstimParamResults::values_weights)
        .def_readwrite("oob_map",&EstimParamResults::oob_map)
        .def_readwrite("oob_weights",&EstimParamResults::oob_weights)
        .def_readwrite("point_estimates",&EstimParamResults::point_estimates)
        .def_readwrite("errors",&EstimParamResults::errors);

    m.def("modelchoice", &ModelChoice_fun_py);
    m.def("estimparam", &EstimParam_fun_py);
    m.def("forestQuantiles", &forestQuantiles);
}