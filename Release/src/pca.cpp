#define RELEASE
#include <iostream>
#ifdef DEBUG
  #include <iterator>
#endif
#include<eigen3/Eigen/SVD>
#include "../include/pca.hpp"

using namespace std;
using namespace Eigen;

int Pca::Calculate(vector<double> &x, 
              const unsigned int &nrows, 
              const unsigned int &ncols, 
              const bool is_corr, 
              const bool is_center, 
              const bool is_scale) {
  _ncols = ncols;
  _nrows = nrows;
  _is_corr = is_corr;
  _is_center = is_center;
  _is_scale = is_scale;
  if (x.size()!= _nrows*_ncols) {
    return -1;
  }
  if ((1 == _ncols) || (1 == nrows)) {
    return -1;
  }
  // Convert vector to Eigen 2-dimensional matrix
  //Map<MatrixXd> _xXd(x.data(), _nrows, _ncols);
  _xXd.resize(_nrows, _ncols);
  for (unsigned int i = 0; i < _nrows; ++i) {
    for (unsigned int j = 0; j < _ncols; ++j) {
      _xXd(i, j) = x[j + i*_ncols];
    }
  }
  // Mean and standard deviation for each column
  VectorXd mean_vector(_ncols);
  mean_vector = _xXd.colwise().mean();
  VectorXd sd_vector(_ncols);
  unsigned int zero_sd_num = 0;
  double denom = static_cast<double>((_nrows > 1)? _nrows - 1: 1);
  for (unsigned int i = 0; i < _ncols; ++i) {
     VectorXd curr_col  = VectorXd::Constant(_nrows, mean_vector(i)); // mean(x) for column x
     curr_col = _xXd.col(i) - curr_col; // x - mean(x)
     curr_col = curr_col.array().square(); // (x-mean(x))^2  
     sd_vector(i) = sqrt((curr_col.sum())/denom);
     if (0 == sd_vector(i)) {
      zero_sd_num++;
     }
  }
  // If colums with sd == 0 are too many,
  // don't continue calculations
  if (1 > _ncols-zero_sd_num) {
    return -1;
  }
  // Delete columns where sd == 0
  MatrixXd tmp(_nrows, _ncols-zero_sd_num);
  VectorXd tmp_mean_vector(_ncols-zero_sd_num);
  unsigned int curr_col_num = 0;
  for (unsigned int i = 0; i < _ncols; ++i) {
    if (0 != sd_vector(i)) {
      tmp.col(curr_col_num) = _xXd.col(i);
      tmp_mean_vector(curr_col_num) = mean_vector(i);
      curr_col_num++;
    } else {
      _eliminated_columns.push_back(i);
    }
  }
  _ncols -= zero_sd_num;
  _xXd = tmp;
  mean_vector = tmp_mean_vector;
  tmp.resize(0, 0); tmp_mean_vector.resize(0);
  // Shift to zero
  if (true == _is_center) {
    for (unsigned int i = 0; i < _ncols; ++i) {
      _xXd.col(i) -= VectorXd::Constant(_nrows, mean_vector(i));
    }
  }
  // Scale to unit variance
  if ( (false == _is_corr) || (true == _is_scale)) {
    for (unsigned int i = 0; i < _ncols; ++i) {
     _xXd.col(i) /= sqrt(_xXd.col(i).array().square().sum()/denom);
    }
  }
  #ifdef DEBUG
    cout << "\nScaled matrix:\n";
    cout << _xXd << endl;
    cout << "\nMean before scaling:\n" << mean_vector.transpose();
    cout << "\nStandard deviation before scaling:\n" << sd_vector.transpose();
  #endif
  // When _nrows < _ncols then svd will be used.
  // If corr is true and _nrows > _ncols then will be used correlation matrix
  // (TODO): What about covariance?
  if ( (_nrows < _ncols) || (false == _is_corr)) { // Singular Value Decomposition is on
    _method = "svd";
    JacobiSVD<MatrixXd> svd(_xXd, ComputeThinV);
    VectorXd eigen_singular_values = svd.singularValues();
    VectorXd tmp_vec = eigen_singular_values.array().square();
    double tmp_sum = tmp_vec.sum();
    tmp_vec /= tmp_sum;
    // PC's standard deviation and
    // PC's proportion of variance
    _kaiser = 0;
    unsigned int lim = (_nrows < _ncols)? _nrows : _ncols;
    for (unsigned int i = 0; i < lim; ++i) {
      _sd.push_back(eigen_singular_values(i)/sqrt(denom));
      if (_sd[i] >= 1) {
        _kaiser = i + 1;
      }
      _prop_of_var.push_back(tmp_vec(i));
    }
    #ifdef DEBUG
      cout << "\n\nStandard deviations for PCs:\n";
      copy(_sd.begin(), _sd.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\nKaiser criterion: PC #" << _kaiser << endl;
    #endif
    tmp_vec.resize(0);
    // PC's cumulative proportion
    _thresh95 = 1; _thresh99 = 1;
    _cum_prop.push_back(_prop_of_var[0]); 
    for (unsigned int i = 1; i < _prop_of_var.size(); ++i) {
      _cum_prop.push_back(_cum_prop[i-1]+_prop_of_var[i]);
      if (_cum_prop[i] <= 0.95) {
        _thresh95 = i+1;
      }
      if (_cum_prop[i] <= 0.99) {
        _thresh99 = i+1;
      }
    }
    #ifdef DEBUG
      cout << "\nCumulative proportion:\n";
      copy(_cum_prop.begin(), _cum_prop.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\nThresh95 criterion: PC #" << _thresh95 << endl;
    cout << "\n\nThresh99 criterion: PC #" << _thresh99 << endl;
    #endif
    // Scores
    MatrixXd eigen_scores = _xXd * svd.matrixV();
    #ifdef DEBUG
      cout << "\n\nRotated values (scores):\n" << eigen_scores;
    #endif
    _scores.reserve(lim*lim);
    for (unsigned int i = 0; i < lim; ++i) {
      for (unsigned int j = 0; j < lim; ++j) {
        _scores.push_back(eigen_scores(i, j));
      }
    }
    eigen_scores.resize(0, 0);
    #ifdef DEBUG
      cout << "\n\nScores in vector:\n";
      copy(_scores.begin(), _scores.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n";  
    #endif
  } else { // COR OR COV MATRICES ARE HERE
    _method = "cor";
    // Calculate covariance matrix
    MatrixXd eigen_cov; // = MatrixXd::Zero(_ncols, _ncols);
    VectorXd sds;
    // (TODO) Should be weighted cov matrix, even if is_center == false
    eigen_cov = (1.0 /(_nrows/*-1*/)) * _xXd.transpose() * _xXd;
    sds = eigen_cov.diagonal().array().sqrt();
    MatrixXd outer_sds = sds * sds.transpose();
    eigen_cov = eigen_cov.array() / outer_sds.array();
    outer_sds.resize(0, 0);
    // ?If data matrix is scaled, covariance matrix is equal to correlation matrix
    #ifdef DEBUG
      cout << eigen_cov << endl;
    #endif
    EigenSolver<MatrixXd> edc(eigen_cov);
    VectorXd eigen_eigenvalues = edc.eigenvalues().real();
    #ifdef DEBUG
      cout << endl << eigen_eigenvalues.transpose() << endl;
    #endif
    MatrixXd eigen_eigenvectors = edc.eigenvectors().real();	
    #ifdef DEBUG
      cout << endl << eigen_eigenvectors << endl;
    #endif
    // The eigenvalues and eigenvectors are not sorted in any particular order.
    // So, we should sort them
    typedef pair<double, int> eigen_pair;
    vector<eigen_pair> ep;	
    for (unsigned int i = 0 ; i < _ncols; ++i) {
	    ep.push_back(make_pair(eigen_eigenvalues(i), i));
    }
    sort(ep.begin(), ep.end()); // Ascending order by default
    // Sort them all in descending order
    MatrixXd eigen_eigenvectors_sorted = MatrixXd::Zero(eigen_eigenvectors.rows(), eigen_eigenvectors.cols());
    VectorXd eigen_eigenvalues_sorted = VectorXd::Zero(_ncols);
    int colnum = 0;
    int i = ep.size()-1;
    for (; i > -1; i--) {
      eigen_eigenvalues_sorted(colnum) = ep[i].first;
      eigen_eigenvectors_sorted.col(colnum++) += eigen_eigenvectors.col(ep[i].second);
    }
    #ifdef DEBUG
      cout << endl << eigen_eigenvalues_sorted.transpose() << endl;
      cout << endl << eigen_eigenvectors_sorted << endl;
    #endif  
    // We don't need not sorted arrays anymore
    eigen_eigenvalues.resize(0);
    eigen_eigenvectors.resize(0, 0);
    
    _sd.clear(); _prop_of_var.clear(); _kaiser = 0;
    double tmp_sum = eigen_eigenvalues_sorted.sum();
    for (unsigned int i = 0; i < _ncols; ++i) {
      _sd.push_back(sqrt(eigen_eigenvalues_sorted(i)));
      if (_sd[i] >= 1) {
        _kaiser = i + 1;
      }
      _prop_of_var.push_back(eigen_eigenvalues_sorted(i)/tmp_sum);
    }
    #ifdef DEBUG
      cout << "\nStandard deviations for PCs:\n";
      copy(_sd.begin(), _sd.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\nProportion of variance:\n";
      copy(_prop_of_var.begin(), _prop_of_var.end(), std::ostream_iterator<double>(std::cout," ")); 
      cout << "\nKaiser criterion: PC #" << _kaiser << endl;
    #endif
    // PC's cumulative proportion
    _cum_prop.clear(); _thresh95 = 1; _thresh99 = 1;
    _cum_prop.push_back(_prop_of_var[0]);
    for (unsigned int i = 1; i < _prop_of_var.size(); ++i) {
      _cum_prop.push_back(_cum_prop[i-1]+_prop_of_var[i]);
      if (_cum_prop[i] <= 0.95) {
        _thresh95 = i+1;
      }
      if (_cum_prop[i] <= 0.99) {
        _thresh99 = i+1;
      }
    }  
    #ifdef DEBUG
      cout << "\n\nCumulative proportions:\n";
      copy(_cum_prop.begin(), _cum_prop.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\n95% threshold: PC #" << _thresh95 << endl;
    cout << "\n\n99% threshold: PC #" << _thresh99 << endl;
    #endif
    // Scores for PCA with correlation matrix
    // Scale before calculating new values
    for (unsigned int i = 0; i < _ncols; ++i) {
     _xXd.col(i) /= sds(i);
    }
    sds.resize(0);
    MatrixXd eigen_scores = _xXd * eigen_eigenvectors_sorted;
    #ifdef DEBUG
      cout << "\n\nRotated values (scores):\n" << eigen_scores;
    #endif
    _scores.clear();
    _scores.reserve(_ncols*_nrows);
    for (unsigned int i = 0; i < _nrows; ++i) {
      for (unsigned int j = 0; j < _ncols; ++j) {
        _scores.push_back(eigen_scores(i, j));
      }
    }
    eigen_scores.resize(0, 0);
    #ifdef DEBUG
      cout << "\n\nScores in vector:\n";
      copy(_scores.begin(), _scores.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\n";  
    #endif
  }

  return 0;
}

int Pca::Calculate(Eigen::MatrixXd mat, const bool is_corr , const bool is_center, const bool is_scale)
{
   _ncols = mat.cols();
  _nrows = mat.rows();
  _is_corr = is_corr;
  _is_center = is_center;
  _is_scale = is_scale;

  if ((1 == _ncols) || (1 == _nrows)) {
    return -1;
  }
  _xXd = mat;
  
  // Mean and standard deviation for each column
  VectorXd mean_vector(_ncols);
  mean_vector = _xXd.colwise().mean();
  VectorXd sd_vector(_ncols);
  unsigned int zero_sd_num = 0;
  double denom = static_cast<double>((_nrows > 1)? _nrows - 1: 1);
  for (unsigned int i = 0; i < _ncols; ++i) {
     VectorXd curr_col  = VectorXd::Constant(_nrows, mean_vector(i)); // mean(x) for column x
     curr_col = _xXd.col(i) - curr_col; // x - mean(x)
     curr_col = curr_col.array().square(); // (x-mean(x))^2  
     sd_vector(i) = sqrt((curr_col.sum())/denom);
     if (0 == sd_vector(i)) {
      zero_sd_num++;
     }
  }
  // If colums with sd == 0 are too many,
  // don't continue calculations
  if (1 > _ncols-zero_sd_num) {
    return -1;
  }
  // Delete columns where sd == 0
  MatrixXd tmp(_nrows, _ncols-zero_sd_num);
  VectorXd tmp_mean_vector(_ncols-zero_sd_num);
  unsigned int curr_col_num = 0;
  for (unsigned int i = 0; i < _ncols; ++i) {
    if (0 != sd_vector(i)) {
      tmp.col(curr_col_num) = _xXd.col(i);
      tmp_mean_vector(curr_col_num) = mean_vector(i);
      curr_col_num++;
    } else {
      _eliminated_columns.push_back(i);
    }
  }
  _ncols -= zero_sd_num;
  _xXd = tmp;
  mean_vector = tmp_mean_vector;
  tmp.resize(0, 0); tmp_mean_vector.resize(0);
  // Shift to zero
  if (true == _is_center) {
    for (unsigned int i = 0; i < _ncols; ++i) {
      _xXd.col(i) -= VectorXd::Constant(_nrows, mean_vector(i));
    }
  }
  // Scale to unit variance
  if ( (false == _is_corr) || (true == _is_scale)) {
    for (unsigned int i = 0; i < _ncols; ++i) {
     _xXd.col(i) /= sqrt(_xXd.col(i).array().square().sum()/denom);
    }
  }
  #ifdef DEBUG
    cout << "\nScaled matrix:\n";
    cout << _xXd << endl;
    cout << "\nMean before scaling:\n" << mean_vector.transpose();
    cout << "\nStandard deviation before scaling:\n" << sd_vector.transpose();
  #endif
  // When _nrows < _ncols then svd will be used.
  // If corr is true and _nrows > _ncols then will be used correlation matrix
  // (TODO): What about covariance?
  if ( (_nrows < _ncols) || (false == _is_corr)) { // Singular Value Decomposition is on
    _method = "svd";
    JacobiSVD<MatrixXd> svd(_xXd, ComputeThinV);
    VectorXd eigen_singular_values = svd.singularValues();
    VectorXd tmp_vec = eigen_singular_values.array().square();
    double tmp_sum = tmp_vec.sum();
    tmp_vec /= tmp_sum;
    // PC's standard deviation and
    // PC's proportion of variance
    _kaiser = 0;
    unsigned int lim = (_nrows < _ncols)? _nrows : _ncols;
    for (unsigned int i = 0; i < lim; ++i) {
      _sd.push_back(eigen_singular_values(i)/sqrt(denom));
      if (_sd[i] >= 1) {
        _kaiser = i + 1;
      }
      _prop_of_var.push_back(tmp_vec(i));
    }
    #ifdef DEBUG
      cout << "\n\nStandard deviations for PCs:\n";
      copy(_sd.begin(), _sd.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\nKaiser criterion: PC #" << _kaiser << endl;
    #endif
    tmp_vec.resize(0);
    // PC's cumulative proportion
    _thresh95 = 1; _thresh99 = 1;
    _cum_prop.push_back(_prop_of_var[0]); 
    for (unsigned int i = 1; i < _prop_of_var.size(); ++i) {
      _cum_prop.push_back(_cum_prop[i-1]+_prop_of_var[i]);
      if (_cum_prop[i] <= 0.95) {
        _thresh95 = i+1;
      }
      if (_cum_prop[i] <= 0.99) {
        _thresh99 = i+1;
      }
    }
    #ifdef DEBUG
      cout << "\nCumulative proportion:\n";
      copy(_cum_prop.begin(), _cum_prop.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\nThresh95 criterion: PC #" << _thresh95 << endl;
    cout << "\n\nThresh99 criterion: PC #" << _thresh99 << endl;
    #endif
    // Scores
    MatrixXd eigen_scores = _xXd * svd.matrixV();
    #ifdef DEBUG
      cout << "\n\nRotated values (scores):\n" << eigen_scores;
    #endif
    _scores.reserve(lim*lim);
    for (unsigned int i = 0; i < lim; ++i) {
      for (unsigned int j = 0; j < lim; ++j) {
        _scores.push_back(eigen_scores(i, j));
      }
    }
    eigen_scores.resize(0, 0);
    #ifdef DEBUG
      cout << "\n\nScores in vector:\n";
      copy(_scores.begin(), _scores.end(),std::ostream_iterator<double>(std::cout," "));  
      cout << "\n";  
    #endif
  } else { // COR OR COV MATRICES ARE HERE
    _method = "cor";
    // Calculate covariance matrix
    MatrixXd eigen_cov; // = MatrixXd::Zero(_ncols, _ncols);
    VectorXd sds;
    // (TODO) Should be weighted cov matrix, even if is_center == false
    eigen_cov = (1.0 /(_nrows/*-1*/)) * _xXd.transpose() * _xXd;
    sds = eigen_cov.diagonal().array().sqrt();
    MatrixXd outer_sds = sds * sds.transpose();
    eigen_cov = eigen_cov.array() / outer_sds.array();
    outer_sds.resize(0, 0);
    // ?If data matrix is scaled, covariance matrix is equal to correlation matrix
    #ifdef DEBUG
      cout << eigen_cov << endl;
    #endif
    EigenSolver<MatrixXd> edc(eigen_cov);
    VectorXd eigen_eigenvalues = edc.eigenvalues().real();
    #ifdef DEBUG
      cout << endl << eigen_eigenvalues.transpose() << endl;
    #endif
    MatrixXd eigen_eigenvectors = edc.eigenvectors().real();	
    #ifdef DEBUG
      cout << endl << eigen_eigenvectors << endl;
    #endif
    // The eigenvalues and eigenvectors are not sorted in any particular order.
    // So, we should sort them
    typedef pair<double, int> eigen_pair;
    vector<eigen_pair> ep;	
    for (unsigned int i = 0 ; i < _ncols; ++i) {
	    ep.push_back(make_pair(eigen_eigenvalues(i), i));
    }
    sort(ep.begin(), ep.end()); // Ascending order by default
    // Sort them all in descending order
    MatrixXd eigen_eigenvectors_sorted = MatrixXd::Zero(eigen_eigenvectors.rows(), eigen_eigenvectors.cols());
    VectorXd eigen_eigenvalues_sorted = VectorXd::Zero(_ncols);
    int colnum = 0;
    int i = ep.size()-1;
    for (; i > -1; i--) {
      eigen_eigenvalues_sorted(colnum) = ep[i].first;
      eigen_eigenvectors_sorted.col(colnum++) += eigen_eigenvectors.col(ep[i].second);
    }
    #ifdef DEBUG
      cout << endl << eigen_eigenvalues_sorted.transpose() << endl;
      cout << endl << eigen_eigenvectors_sorted << endl;
    #endif  
    // We don't need not sorted arrays anymore
    eigen_eigenvalues.resize(0);
    eigen_eigenvectors.resize(0, 0);
    
    _sd.clear(); _prop_of_var.clear(); _kaiser = 0;
    double tmp_sum = eigen_eigenvalues_sorted.sum();
    for (unsigned int i = 0; i < _ncols; ++i) {
      _sd.push_back(sqrt(eigen_eigenvalues_sorted(i)));
      if (_sd[i] >=1.0) {
        _kaiser = i + 1;
      }
      _prop_of_var.push_back(eigen_eigenvalues_sorted(i)/tmp_sum);
    }
    #ifdef DEBUG
      cout << "\nStandard deviations for PCs:\n";
      copy(_sd.begin(), _sd.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\nProportion of variance:\n";
      copy(_prop_of_var.begin(), _prop_of_var.end(), std::ostream_iterator<double>(std::cout," ")); 
      cout << "\nKaiser criterion: PC #" << _kaiser << endl;
    #endif
    // PC's cumulative proportion
    _cum_prop.clear(); _thresh95 = 1; _thresh99 = 1;
    _cum_prop.push_back(_prop_of_var[0]);
    for (unsigned int i = 1; i < _prop_of_var.size(); ++i) {
      _cum_prop.push_back(_cum_prop[i-1]+_prop_of_var[i]);
      if (_cum_prop[i] <= 0.95) {
        _thresh95 = i+1;
      }
      if (_cum_prop[i] <= 0.99) {
        _thresh99 = i+1;
      }
    }  
    #ifdef DEBUG
      cout << "\n\nCumulative proportions:\n";
      copy(_cum_prop.begin(), _cum_prop.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\n\n95% threshold: PC #" << _thresh95 << endl;
    cout << "\n\n99% threshold: PC #" << _thresh99 << endl;
    #endif
    // Scores for PCA with correlation matrix
    // Scale before calculating new values
    for (unsigned int i = 0; i < _ncols; ++i) {
     _xXd.col(i) /= sds(i);
    }
    sds.resize(0);
    MatrixXd eigen_scores = _xXd * eigen_eigenvectors_sorted;
    #ifdef DEBUG
      cout << "\n\nRotated values (scores):\n" << eigen_scores;
    #endif
    _scores.clear();
    _scores.reserve(_ncols*_nrows);
    for (unsigned int i = 0; i < _nrows; ++i) {
      for (unsigned int j = 0; j < _ncols; ++j) {
        _scores.push_back(eigen_scores(i, j));
      }
    }
    eigen_scores.resize(0, 0);
    #ifdef DEBUG
      cout << "\n\nScores in vector:\n";
      copy(_scores.begin(), _scores.end(), std::ostream_iterator<double>(std::cout," "));  
      cout << "\n";  
    #endif
  }

  return 0;  
}
std::vector<double> Pca::sd(void) { return _sd; };    
std::vector<double> Pca::prop_of_var(void) {return _prop_of_var; };
std::vector<double> Pca::cum_prop(void) { return _cum_prop; };
std::vector<double> Pca::scores(void) { return _scores; };
std::vector<unsigned int> Pca::eliminated_columns(void) { return _eliminated_columns; }
string Pca::method(void) { return _method; }
unsigned int Pca::kaiser(void) { return _kaiser; };
unsigned int Pca::thresh95(void) { return _thresh95; };
unsigned int Pca::thresh99(void) { return _thresh99; };
unsigned int Pca::ncols(void) { return _ncols; }
unsigned int Pca::nrows(void) { return _nrows; }
bool Pca::is_scale(void) {  return _is_scale; }
bool Pca::is_center(void) { return _is_center; }
Pca::Pca(void) {
  _nrows      = 0;
  _ncols      = 0;
  // Variables will be scaled by default
  _is_center  = true;
  _is_scale   = true;  
  // By default will be used singular value decomposition
  _method   = "svd";
  _is_corr  = false;

  _kaiser   = 0;
  _thresh95 = 1;
  _thresh99 = 1;
}
Pca::~Pca(void) { 
  _xXd.resize(0, 0);
  _x.clear();
}
