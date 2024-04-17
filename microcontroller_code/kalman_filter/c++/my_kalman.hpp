#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>

class KF {
public:
  KF(const Eigen::VectorXd& initial_state);
  // ~KF() = default;

  std::string toString() const;

  void predict(double dt);
  void update(const Eigen::VectorXd& meas_value, const Eigen::MatrixXd& meas_variance, const Eigen::MatrixXd& meas_func);

  Eigen::VectorXd getStateVector();
  Eigen::MatrixXd getStateCovarianceSubmatrix(int row_start, int row_end, int col_start, int col_end);

  const Eigen::MatrixXd& cov() const { return _P; }
  const Eigen::VectorXd& mean() const { return _x; }

private:
  Eigen::VectorXd _x;  // State estimate
  Eigen::MatrixXd _P;  // Covariance matrix
  double _var_ax;
  double _var_ay;
  double _var_az;
  Eigen::MatrixXd _Q;  // Process noise covariance
};

#endif /* KALMAN_FILTER_H */
