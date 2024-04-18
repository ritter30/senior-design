#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

class KF {
public:
  KF(const VectorXd& initial_state) : _x(initial_state), _P(MatrixXd::Identity(9, 9)) {
    // Acceleration variances (assuming constant)
    _var_ax = 0.0000931225;
    _var_ay = 0.0000808201;
    _var_az = 0.0001729225;
  }

  std::string toString() const {
    return "x: " + std::to_string(_x.rows()) + "x" + std::to_string(_x.cols()) +
           "\nP: " + std::to_string(_P.rows()) + "x" + std::to_string(_P.cols()) +
           "\nQ: " + std::to_string(_Q.rows()) + "x" + std::to_string(_Q.cols());
  }

  void predict(double dt) {
    double dt2 = dt * dt;

    MatrixXd F = MatrixXd::Zero(9, 9);
    F.block<3, 3>(0, 0) << 1, dt, 0.5 * dt2,
                           0, 1, dt,
                           0, 0, 1;

    VectorXd new_x = F * _x;

    MatrixXd G(3, 1);
    G << 0.5 * dt2, dt, 1;

    _Q = MatrixXd::Zero(9, 9);
    _Q.block<3, 3>(0, 0) = G * G.transpose() * _var_ax;
    _Q.block<3, 3>(3, 3) = G * G.transpose() * _var_ay;
    _Q.block<3, 3>(6, 6) = G * G.transpose() * _var_az;

    MatrixXd new_P = F * _P * F.transpose() + _Q;

    _P = new_P;
    _x = new_x;
  }

  void update(const VectorXd& meas_value, const MatrixXd& meas_variance, const MatrixXd& meas_func) {
    MatrixXd H = meas_func;
    VectorXd z = meas_value;
    MatrixXd R = meas_variance;

    VectorXd y = z - H * _x;
    MatrixXd S = H * _P * H.transpose() + R;
    MatrixXd K = _P * H.transpose() * S.inverse();

    VectorXd new_x = _x + K * y;
    MatrixXd new_P = (MatrixXd::Identity(9, 9) - K * H) * _P;

    _P = new_P;
    _x = new_x;
  }

  const MatrixXd& cov() const { return _P; }
  const VectorXd& mean() const { return _x; }chat

  VectorXd getStateVector() const { return _x; }

  MatrixXd getStateCovarianceSubmatrix(int row_start, int row_end, int col_start, int col_end) const {
    return _P.block(row_start, row_end - row_start + 1, col_start, col_end - col_start + 1);
  }

private:
  VectorXd _x;  // State estimate
  MatrixXd _P;  // Covariance matrix
  double _var_ax;
  double _var_ay;
  double _var_az;
  MatrixXd _Q;  // Process noise covariance
};