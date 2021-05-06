#include <Rcpp.h>

void foo_cpp_function(SEXP a, SEXP b) {
  Rcpp::Rcout << "Hello" << std::endl;
  int* a_ptr = INTEGER(a);
  int* b_ptr = INTEGER(b);
  PROTECT(a);
  PROTECT(b);
  Rcpp::Rcout << "a:" << *a_ptr << " b:" << *b_ptr << std::endl;
  UNPROTECT(2);
}

RCPP_MODULE(foomodule) {
   Rcpp::function("foo_r_function", &foo_cpp_function) ;
}