#pragma once

#include <numeric>
#include <random>
#include <Eigen/Dense>

static std::random_device randev;
static std::default_random_engine RNG(randev());
static std::uniform_real_distribution<double> runif(0,1);

using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic> Mat2D;
typedef Matrix<double, Dynamic, 1>  Col;
typedef Matrix<double, 1, Dynamic>  Row;
typedef Matrix<size_t, 1, Dynamic>  Rowi;
typedef Matrix<std::complex<double>,Dynamic,Dynamic> Mat2Dc;
typedef Matrix<std::complex<double>, Dynamic, 1>  Colc;

inline Row col_means( Mat2D mat ) { return mat.colwise().sum() / mat.rows(); }

// Mat2D read_matrix_file(std::string filename, char sep); 
using std::complex;
using std::vector;

Row col_stdev( Mat2D mat, Row means ) {
    Row stdevs = Row::Zero(mat.cols());
    const double N = static_cast<double>(mat.rows());
    if ( N < 2 ) return stdevs;

    const double N_inv = 1.0/(N-1); // N-1 for unbiased sample variance
    for (int i=0; i<mat.cols(); i++) {
        stdevs[i] = sqrt( (mat.col(i).array()-means[i]).square().sum() * N_inv );
    }
    return stdevs;
}


double dominant_eigenvalue( EigenSolver<Mat2Dc> es ){
    Colc  ev = es.eigenvalues();
    double m = 0;

    for (int i = 0; i<ev.size(); i++) {
        if (imag(ev[i]) == 0) {
            if (abs(ev[i]) > m) m = abs(ev[i]);
        }
    }
    return m;
}


Colc dominant_eigenvector( EigenSolver<Mat2D> es ){
    Colc eig_val = es.eigenvalues();
    double m = 0;
    int idx = 0;

    for (int i = 0; i<eig_val.size(); i++) {
        if (imag(eig_val[i]) == 0) {
            if (abs(eig_val[i]) > m) {
                m = abs(eig_val[i]);
                idx = i;
            }
        }
    }
    return es.eigenvectors().col(idx);
}

Mat2D colwise_z_scores( const Mat2D& mat ) {
    // Standardize values by column, i.e. convert to Z-scores
    Row means = col_means( mat );
    Row stdev = col_stdev( mat, means );
    Mat2D zmat = Mat2D::Zero(mat.rows(), mat.cols());
    for (int r = 0; r<mat.rows(); r++) { zmat.row(r) = (mat.row(r) - means).cwiseQuotient(stdev); }
    return zmat;
}

vector<size_t> ordered(Col const& values) {
    vector<size_t> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));

    std::sort(
        begin(indices), end(indices),
        [&](auto a, auto b) { return values[a] < values[b]; }
    );
    return indices;
}



double normalcdf(double z){
    const double c1 = 0.196854;
    const double c2 = 0.115194;
    const double c3 = 0.000344;
    const double c4 = 0.019527;
    double p;
    if (z < 0) {
        z = -z;
        p = 1 - 0.5 / pow(1 + c1*z + c2*z*z + c3*z*z*z + c4*z*z*z*z,4);
    } else {
        p = 0.5 / pow(1 + c1*z + c2*z*z + c3*z*z*z + c4*z*z*z*z,4);
    }
    double probn = 1.0 - p;
    return probn;
}

double wilcoxon(const Col err_1, const Col err_2) {
    size_t n = err_1.rows();
    Col del = err_1.cwiseAbs() - err_2.cwiseAbs();
    Rowi sdel;
    sdel.setZero(del.size());
    //Matrix<int, Dynamic, 1> sdel = del.unaryExpr(std::ptr_fun(_sgn)); // can't get this to work
    for (int i=0; i<del.size(); i++)  sdel(i) = (0 < del(i)) - (del(i) < 0); // get the sign of each element
    Col adel = del.cwiseAbs();
    // 's' gives the original positions (indices) of the sorted values
    vector<size_t> s = ordered(adel); 
    float d = 0;
    for (int i=0; i<n; i++) d += (i+1)*sdel(s[i]);
    double t  = n*(n+1)/2.0;
    double v  = (t-d)/2.0;
    double ev = t/2.0;
    double sv = sqrt((double) n*(n+1)*(2*n+1)/24.0);
    double z = (v-ev)/sv;
    double probw = 1.0 - normalcdf(z);

    return probw;
}
/*
double normal_pdf(double x, double mu, double var) {
    long double PI = 3.1415926535897932384;
    return exp(-pow(x-mu,2) / (2.0*var)) / sqrt(2*PI*var);
}

double normal_cdf(double x, double mu, double var) {
    x = (x-mu)/sqrt(var);
    // Abramowitz & Stegun (1964) approximation
    long double b0 = 0.2316419;
    double b1 = 0.319381530;
    double b2 = -0.356563782;
    double b3 = 1.781477937;
    double b4 = -1.821255978;
    double b5 = 1.330274429;
    if (x >= 0.0) {
        long double t = 1.0/(1.0+b0*x);
        return 1.0 - normal_pdf(x, 0, 1)*(b1*t + b2*pow(t,2) + b3*pow(t,3) + b4*pow(t,4) + b5*pow(t,5));
    } else {
        long double t = 1.0/(1.0-b0*x);
        return normal_pdf(x, 0, 1)*(b1*t + b2*pow(t,2) + b3*pow(t,3) + b4*pow(t,4) + b5*pow(t,5));
    }
}*/

void rand_nchoosek(size_t N, vector<int>& sample) {
    if (sample.size() == 0 ) return;
    size_t k = sample.size();       // k is specified by size of requested vector

    assert( k <= N );

    size_t top = N-k;
    double Nreal = (double) N;

    int newidx=0;
    int lastidx=0;
    int i=0;

    while ( k >= 2 ) {
        double V = runif(RNG);
        //double V = mtrand->rand();
        int S=0;
        double quot = top/Nreal;
        while( quot > V ) {
            S++; top-=1; Nreal-=1;
            quot =(quot*top)/Nreal;
        }
        //skip over the next S records and select the following one for the sample
        newidx = lastidx + S;
        sample[i]=newidx; lastidx = newidx+1; i++;
        Nreal -= 1.0; k -= 1;
    }

    if ( k == 1 ) {
        // the following line had (newidx+1) instead of lastidx before, which
        // produced incorrect results when N == 1; this, I believe, is correct
        std::uniform_int_distribution<int> randInt(0,(int) Nreal - 1);
        sample[i++] = lastidx + randInt(RNG); // truncated float on [0, Nreal]
    }
}


typedef enum { KERNEL_TYPE1, KERNEL_TYPE2 } REGRESSION_ALGORITHM;
typedef enum { LOO, LSO } VALIDATION_METHOD;
typedef enum { PRESS, MSEP, RMSEP } VALIDATION_OUTPUT;

class PLS_Model {
  public:
    Mat2Dc P, W, R, Q, T;
    size_t A;
    REGRESSION_ALGORITHM algorithm;
    void initialize(size_t num_predictors, size_t num_responses, size_t num_components) {
        A = num_components; 
        P.setZero(num_predictors, num_components);
        W.setZero(num_predictors, num_components);
        R.setZero(num_predictors, num_components);
        Q.setZero(num_responses, num_components);
        // T will be initialized if needed
        return;
    }

    //"Modified kernel algorithms 1 and 2"
    //from Dayal and MacGregor (1997) "Improved PLS Algorithms" J. of Chemometrics. 11,73-85.
    void plsr(const Mat2D X, const Mat2D Y, REGRESSION_ALGORITHM _algorithm) {
        algorithm = _algorithm;
        size_t M = Y.cols(); // Number of response variables == columns in Y

        if (algorithm == KERNEL_TYPE1) T.setZero(X.rows(), A);

        Mat2D XY = X.transpose() * Y;
        Mat2D XX;
        if (algorithm == KERNEL_TYPE2) XX = X.transpose() * X;

        for (auto i=0; i<A; i++) {
            Colc w, p, q, r, t; 
            complex<double> tt;
            if (M==1) {
                w = XY.cast<complex<double> >();
            } else {
                EigenSolver<Mat2D> es( (XY.transpose() * XY) );
                q = dominant_eigenvector(es);
                w = (XY*q);
            }

            w /= sqrt((w.transpose()*w)(0,0)); // use normalize function from eigen?
            r=w;
            for (auto j=0; j<=i-1; j++) {
                r -= (P.col(j).transpose()*w)(0,0)*R.col(j);
            }
            if (algorithm == KERNEL_TYPE1) {
                t = X*r;
                tt = (t.transpose()*t)(0,0);
                p.noalias() = (X.transpose()*t);
            } else if (algorithm == KERNEL_TYPE2) {
                tt = (r.transpose()*XX*r)(0,0);
                p.noalias() = (r.transpose()*XX).transpose();
            }
            p /= tt;
            q.noalias() = (r.transpose()*XY).transpose(); q /= tt;
            XY -= ((p*q.transpose())*tt).real(); // is casting this to 'real' always safe?
            W.col(i)=w;
            P.col(i)=p;
            Q.col(i)=q;
            R.col(i)=r;
            if (algorithm == KERNEL_TYPE1) T.col(i) = t;
        }
        if (algorithm == KERNEL_TYPE2) T = X*R; // not part of the algorithm; allows users to retrieve scores
        return; 
    }

    Mat2Dc scores() { return scores(A); }
    Mat2Dc scores(size_t comp) { 
        assert (comp <= A);
        assert (comp > 0);
        return T.leftCols(comp);
    }

    Mat2Dc loadingsX() { return loadingsX(A); }
    Mat2Dc loadingsX(size_t comp) { 
        assert (comp <= A);
        assert (comp > 0);
        return P.leftCols(comp);
    }

    Mat2Dc loadingsY() { return loadingsY(A); }
    Mat2Dc loadingsY(size_t comp) { 
        assert (comp <= A);
        assert (comp > 0);
        return Q.leftCols(comp);
    }
 
    // compute the regression coefficients (aka 'beta')
    Mat2Dc coefficients() { return coefficients(A); }
    Mat2Dc coefficients(size_t comp) {
        assert (comp <= A);
        assert (comp > 0);
        return R.leftCols(comp)*Q.leftCols(comp).transpose();
    }

    // predicted Y values, given X values and pls model
    Mat2D fitted_values(const Mat2D& X) { return fitted_values(X, A); }
    Mat2D fitted_values(const Mat2D& X, size_t comp) {
        return X*coefficients(comp).real();
    }

    // unexplained portion of Y values
    Mat2D residuals(const Mat2D& X, const Mat2D& Y) { return residuals(X, Y, A); }
    Mat2D residuals(const Mat2D& X, const Mat2D& Y, size_t comp) {
        return Y - fitted_values(X, comp);
    }
    
    // Sum of squared errors
    Row SSE(const Mat2D& X, const Mat2D& Y) { return this->SSE(X, Y, A); }
    Row SSE(const Mat2D& X, const Mat2D& Y, size_t comp) {
        return residuals(X, Y, comp).colwise().squaredNorm();
    }

    // Total sum of squares
    Row SST(const Mat2D& Y) { 
        Row sst(Y.cols());
        for (auto c = 0; c < Y.cols(); c++) {
            sst(c) = (Y.col(c).array() - (Y.col(c).sum()/Y.rows())).square().sum();
        }
        return sst;
    }

    // fraction of explainable variance
    Row explained_variance(const Mat2D& X, const Mat2D& Y) { return explained_variance(X, Y, A); }
    Row explained_variance(const Mat2D& X, const Mat2D& Y, size_t comp) {
        assert (comp <= A);
        assert (comp > 0);
    //    cerr << "ev: " << this->SSE(X, Y, comp).cwiseQuotient( SST(Y) ) << endl;
        return (1.0 - this->SSE(X, Y, comp).cwiseQuotient( SST(Y) ).array()).matrix(); 
    }

    std::vector<Mat2D> _loo_cv_residual_matrix(const Mat2D& X, const Mat2D& Y) { 
        Mat2D Xv = X.bottomRows(X.rows()-1);
        Mat2D Yv = Y.bottomRows(Y.rows()-1);
       
        // vector of error matrices(rows=Y.rows(), cols=Y.cols())
        // col = Y category #, row = obs #, tier = component
        std::vector<Mat2D> Ev(this->A, Mat2D::Zero(X.rows(), Y.cols()));

        PLS_Model plsm_v;
        plsm_v.initialize(Xv.cols(), Yv.cols(), this->A);
        for (auto i = 0; i < X.rows(); i++) {
            plsm_v.plsr(Xv, Yv, this->algorithm);
            for (auto j = 0; j < this->A; j++) {
                Row res = plsm_v.residuals(X.row(i), Y.row(i), j+1).row(0); // convert j to number of components
                for (auto k = 0; k < res.size(); k++) Ev[j](i,k) = res(k);
            }
            if ( i < Xv.rows() ) {
                // we haven't run out of rows to swap out yet
                Xv.row(i) = X.row(i); 
                Yv.row(i) = Y.row(i); 
            }
        }
        return Ev;
    }


    // leave-one-out validation of model (i.e., are we overfitting?)
    Mat2D loo_validation(const Mat2D& X, const Mat2D& Y, VALIDATION_OUTPUT out_type) { 
        std::vector<Mat2D> Ev = _loo_cv_residual_matrix(X,Y);
        Mat2D SSEv = Mat2D::Zero(Y.cols(), this->A);

        for (auto j = 0; j < this->A; j++) {
            Mat2D res = Ev[j];
            Mat2D SE  = res.cwiseProduct(res);
            // rows in SSEv correspond to different parameters
            // Collapse the squared errors so that we're summing over all predicted rows
            // then transpose, so that rows now represent different parameters
            SSEv.col(j) += SE.colwise().sum().transpose();
        }
        if ( out_type == PRESS ) {
            return SSEv;
        } else {
            SSEv /= static_cast<double>(X.rows());
            if ( out_type == MSEP ) { 
                return SSEv;
            } else {
                // RMSEP
                return SSEv.cwiseSqrt();
            }
        }
    }


    std::vector<Mat2D> _lso_cv_residual_matrix(const Mat2D& X, const Mat2D& Y, const float test_fraction, const size_t num_trials) { 
        const size_t N = X.rows();
        const size_t test_size = (int) (test_fraction * N + 0.5);
        const size_t train_size = N - test_size;
        std::vector<Mat2D> Ev(this->A, Mat2D::Zero(num_trials*test_size, Y.cols()));
        vector<int> sample(train_size);
        Mat2D Xv(train_size, X.cols()); // values we're training on
        Mat2D Yv(train_size, Y.cols());
        Mat2D Xp(test_size, X.cols());  // values we're predicting
        Mat2D Yp(test_size, Y.cols());

        PLS_Model plsm_v;
        plsm_v.initialize(Xv.cols(), Yv.cols(), this->A);
        for (auto rep = 0; rep < num_trials; ++rep) {
            rand_nchoosek(N, sample);
            size_t j=0;
            size_t k=0;
            for (auto i=0; i<N; ++i) {
                if( sample[j] == i ) { // in training set
                    Xv.row(j) = X.row(i);
                    Yv.row(j) = Y.row(i);
                    j++; 
                } else {               // in testing set
                    Xp.row(k) = X.row(i);
                    Yp.row(k) = Y.row(i);
                    k++; 
                }
            }

            plsm_v.plsr(Xv, Yv, this->algorithm);
            for (auto j = 0; j < this->A; j++) {
                Mat2D res = plsm_v.residuals(Xp, Yp, j+1); // convert j to number of components
                Ev[j].middleRows(rep*test_size, test_size) = res; // write to submatrix; middleRows(startRow, numRows)
            }
        }

        return Ev;
    }


    // leave-some-out validation of model (i.e., are we overfitting?)
    Mat2D lso_validation(const Mat2D& X, const Mat2D& Y, VALIDATION_OUTPUT out_type, float test_fraction, size_t num_trials) { 
        const std::vector<Mat2D> Ev = _lso_cv_residual_matrix(X, Y, test_fraction, num_trials);
        assert(Ev.size() > 0);
        const size_t num_residuals = Ev[0].rows();
        Mat2D SSEv = Mat2D::Zero(Y.cols(), this->A);

        for (auto j = 0; j < this->A; j++) {
            Mat2D res = Ev[j];
            // square all of the residuals
            Mat2D SE  = res.cwiseProduct(res);
            // rows in SSEv correspond to different parameters
            // Collapse the squared errors so that we're summing over all predicted rows
            // then transpose, so that rows now represent different parameters
            SSEv.col(j) += SE.colwise().sum().transpose();
        }
        if ( out_type == PRESS ) {
            return SSEv;
        } else {
            SSEv /= static_cast<double>(num_residuals);
            if ( out_type == MSEP ) { 
                return SSEv;
            } else {
                // RMSEP
                return SSEv.cwiseSqrt();
            }
        }
    }
    

    Rowi loo_optimal_num_components(const Mat2D& X, const Mat2D& Y) { 
        const size_t dummy = 0;
        return _optimal_num_components(X, Y, LOO, dummy, dummy);
    }


    Rowi lso_optimal_num_components(const Mat2D& X, const Mat2D& Y, const float test_fraction, const size_t num_trials) { 
        return _optimal_num_components(X, Y, LSO, test_fraction, num_trials);
    }


    Rowi _optimal_num_components(const Mat2D& X, const Mat2D& Y, const VALIDATION_METHOD vmethod, const float test_fraction, const size_t num_trials) { 
        // tier = component #, col = Y category, row = obs #
        std::vector<Mat2D> errors;
        if (vmethod == LOO) {
            errors = _loo_cv_residual_matrix(X,Y);
        } else if (vmethod == LSO) {
            errors = _lso_cv_residual_matrix(X, Y, test_fraction, num_trials);
        }
        Mat2D press = Mat2D::Zero(Y.cols(), this->A);
        Rowi min_press_idx = Rowi::Zero(Y.cols());
        Row  min_press_val(Y.cols());
        Rowi best_comp(Y.cols());
        
        // Determine PRESS values
        for (auto j = 0; j < this->A; j++) {
            Mat2D resmat = errors[j];
            for (auto i = 0; i < X.rows(); i++) {
                Row res = resmat.row(i);
                press.col(j) += res.cwiseProduct(res).transpose();
            }
        }
        
        min_press_val = press.col(0);
        // Find the component number that minimizes PRESS for each Y category
        for (auto i=0; i<press.rows(); i++) {              // for each Y category
            for (auto j = 0; j < this->A; j++) {
                if (press(i,j) < min_press_val(i)) {
                    min_press_val(i) = press(i,j);
                    min_press_idx(i) = j;
                }
            }
        }

        best_comp = min_press_idx.array() + 1;
        // Find the min number of components that is not significantly
        // different from the min PRESS at alpha = 0.1 for each Y category
        const double ALPHA = 0.1;
        for (auto i=0; i<press.rows(); i++) {              // for each Y category
            for (auto j=0; j<min_press_idx(i); j++) {      // for each smaller number of components
                Col err1 = errors[min_press_idx(i)].col(i);
                Col err2 = errors[j].col(i);
                if (wilcoxon(err1, err2) > ALPHA) {
                    best_comp(i) = j+1; // +1 to convert from index to component number
                    break;
                }
            }
        }

        return best_comp;
    }

};
