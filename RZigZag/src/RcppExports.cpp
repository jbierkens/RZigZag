// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// ZigZagLogistic
List ZigZagLogistic(const Eigen::MatrixXd dataX, const Eigen::VectorXi dataY, int n_iterations, const NumericVector x0, const double finalTime, const bool subsampling, const bool controlvariates, const int n_samples, const int n_batches, const bool computeCovariance, const bool upperbound, const NumericVector v0, const NumericVector x_ref);
RcppExport SEXP _RZigZag_ZigZagLogistic(SEXP dataXSEXP, SEXP dataYSEXP, SEXP n_iterationsSEXP, SEXP x0SEXP, SEXP finalTimeSEXP, SEXP subsamplingSEXP, SEXP controlvariatesSEXP, SEXP n_samplesSEXP, SEXP n_batchesSEXP, SEXP computeCovarianceSEXP, SEXP upperboundSEXP, SEXP v0SEXP, SEXP x_refSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type dataX(dataXSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXi >::type dataY(dataYSEXP);
    Rcpp::traits::input_parameter< int >::type n_iterations(n_iterationsSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type x0(x0SEXP);
    Rcpp::traits::input_parameter< const double >::type finalTime(finalTimeSEXP);
    Rcpp::traits::input_parameter< const bool >::type subsampling(subsamplingSEXP);
    Rcpp::traits::input_parameter< const bool >::type controlvariates(controlvariatesSEXP);
    Rcpp::traits::input_parameter< const int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< const int >::type n_batches(n_batchesSEXP);
    Rcpp::traits::input_parameter< const bool >::type computeCovariance(computeCovarianceSEXP);
    Rcpp::traits::input_parameter< const bool >::type upperbound(upperboundSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type v0(v0SEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type x_ref(x_refSEXP);
    rcpp_result_gen = Rcpp::wrap(ZigZagLogistic(dataX, dataY, n_iterations, x0, finalTime, subsampling, controlvariates, n_samples, n_batches, computeCovariance, upperbound, v0, x_ref));
    return rcpp_result_gen;
END_RCPP
}
// ZigZagGaussian
List ZigZagGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iterations, const Eigen::VectorXd x0, const double finalTime, const int n_samples, const int n_batches, bool computeCovariance);
RcppExport SEXP _RZigZag_ZigZagGaussian(SEXP VSEXP, SEXP muSEXP, SEXP n_iterationsSEXP, SEXP x0SEXP, SEXP finalTimeSEXP, SEXP n_samplesSEXP, SEXP n_batchesSEXP, SEXP computeCovarianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type V(VSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< int >::type n_iterations(n_iterationsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type x0(x0SEXP);
    Rcpp::traits::input_parameter< const double >::type finalTime(finalTimeSEXP);
    Rcpp::traits::input_parameter< const int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< const int >::type n_batches(n_batchesSEXP);
    Rcpp::traits::input_parameter< bool >::type computeCovariance(computeCovarianceSEXP);
    rcpp_result_gen = Rcpp::wrap(ZigZagGaussian(V, mu, n_iterations, x0, finalTime, n_samples, n_batches, computeCovariance));
    return rcpp_result_gen;
END_RCPP
}
// BPSGaussian
List BPSGaussian(const Eigen::MatrixXd V, const Eigen::VectorXd mu, int n_iterations, const Eigen::VectorXd x0, const double finalTime, const double refresh_rate, const bool unit_velocity, const int n_samples, const int n_batches, bool computeCovariance);
RcppExport SEXP _RZigZag_BPSGaussian(SEXP VSEXP, SEXP muSEXP, SEXP n_iterationsSEXP, SEXP x0SEXP, SEXP finalTimeSEXP, SEXP refresh_rateSEXP, SEXP unit_velocitySEXP, SEXP n_samplesSEXP, SEXP n_batchesSEXP, SEXP computeCovarianceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd >::type V(VSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type mu(muSEXP);
    Rcpp::traits::input_parameter< int >::type n_iterations(n_iterationsSEXP);
    Rcpp::traits::input_parameter< const Eigen::VectorXd >::type x0(x0SEXP);
    Rcpp::traits::input_parameter< const double >::type finalTime(finalTimeSEXP);
    Rcpp::traits::input_parameter< const double >::type refresh_rate(refresh_rateSEXP);
    Rcpp::traits::input_parameter< const bool >::type unit_velocity(unit_velocitySEXP);
    Rcpp::traits::input_parameter< const int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< const int >::type n_batches(n_batchesSEXP);
    Rcpp::traits::input_parameter< bool >::type computeCovariance(computeCovarianceSEXP);
    rcpp_result_gen = Rcpp::wrap(BPSGaussian(V, mu, n_iterations, x0, finalTime, refresh_rate, unit_velocity, n_samples, n_batches, computeCovariance));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RZigZag_ZigZagLogistic", (DL_FUNC) &_RZigZag_ZigZagLogistic, 13},
    {"_RZigZag_ZigZagGaussian", (DL_FUNC) &_RZigZag_ZigZagGaussian, 8},
    {"_RZigZag_BPSGaussian", (DL_FUNC) &_RZigZag_BPSGaussian, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_RZigZag(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
