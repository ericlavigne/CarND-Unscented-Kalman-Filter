#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // Set to true after receiving first measurement
  is_initialized_ = false;

  // How many measurements processed so far?
  iterations = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Pre-allocate state and covariance matrices that will become meaningful after first measurement
  x_ = VectorXd(5);
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // Typical bicycle speed is 5.5 m/s. Braking to stop takes about a second, so max 5.5 m/s2.
  // Use half of that for std dev of 2.7 m/s2.
  std_a_ = 1.4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // Bicycle can complete a full circle in about 5 seconds, so max turn of 2PI/5s or 1.2 rad/s.
  // Bicycle can switch directions from such a sharp turn in about 2 seconds, so max of 0.6 rad/s2.
  // Use half of that for std dev of 0.3 m/s2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.4;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* Weights of sigma points
  weights = VectorXd(2*n_aug_+1);
  weights.fill(0.5/(n_aug_+lambda_));
  weights(0) = lambda_/(lambda_+n_aug_);

  // Pre-allocate matrix that will only be used locally, just to avoid re-allocation for each round
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_+1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if(!is_initialized_) {
    x_.fill(0.0);
    P_.fill(0.0);
    time_us_ = meas_package.timestamp_;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double ro = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      double ro_dot = meas_package.raw_measurements_[2];

      // Convert radar from polar to cartesian coordinates
      double px = ro * sin(theta);
      double py = ro * cos(theta);

      // Low estimates of velocity, assuming no motion perpendicular to radar beam.
      double v = ro_dot;

      // Ballpark estimate of direction, assuming movement straight toward or away from sensor.
      double psi = theta;

      // No information on turning, so assume no turning.
      double psi_dot = 0.0;

      x_ << px, py, v, psi, psi_dot;

      double spatial_uncertainty = max(std_radr_, std_radphi_ * ro);

      P_(0,0) = spatial_uncertainty * spatial_uncertainty;
      P_(1,1) = spatial_uncertainty * spatial_uncertainty;
      P_(2,2) = 2.7 * 2.7; // Half of bicycle max speed
      P_(3,3) = 0.25 * 3.14 * 3.14; // No idea. Limit uncertainty to quarter of circle to avoid extreme non-linearity.
      P_(4,4) = 0.6 * 0.6; // Half of bicycle's maximum turning speed, because we have no idea.

      cout << "Initializing with radar: " << endl << x_ << endl << "Pmag = " << P_.norm() << endl;

    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {

      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];

      x_ << px, py, 0, 0, 0;

      P_(0,0) = std_laspx_;
      P_(1,1) = std_laspy_;
      P_(2,2) = 2.7 * 2.7; // Half of bicycle max speed
      P_(3,3) = 0.25 * 3.14 * 3.14; // No idea. Limit uncertainty to quarter of circle to avoid extreme non-linearity.
      P_(4,4) = 0.6 * 0.6; // Half of bicycle's maximum turning speed, because we have no idea.

      cout << "Initializing with lidar: " << endl << x_ << endl;
      cout << "Pmag = " << P_.norm() << endl;

    } else {
      cout << "Unrecognized sensor type" << endl;
      throw "Unrecognized sensor type";
    }

    is_initialized_ = true;
    iterations++;
    return;
  }
  double delta_t = (meas_package.timestamp_ - time_us_) * 0.000001;
  cout << "Performing prediction with deltat of " << delta_t << endl;
  Prediction(delta_t);
  cout << "After prediction:" << endl << x_ << endl;
  cout << "Pmag = " << P_.norm() << endl;
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "Radar measurement is:" << endl << meas_package.raw_measurements_ << endl;
    if(use_radar_) {
      UpdateRadar(meas_package);
    }
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    cout << "Lidar measurement is:" << endl << meas_package.raw_measurements_ << endl;
    if(use_laser_) {
      UpdateLidar(meas_package);
    }
  } else {
    cout << "Unrecognized sensor type" << endl;
    throw "Unrecognized sensor type";
  }
  cout << "After measurement:" << endl << x_ << endl;
  cout << "Pmag = " << P_.norm() << endl;
  time_us_ = meas_package.timestamp_;
  iterations++;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Generate sigma points with augmentation to represent process noise
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
  P_aug.fill(0.0);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  x_aug.head(n_x_) = x_;
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug(n_x_,n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1,n_x_+1) = std_yawdd_ * std_yawdd_;
  MatrixXd A_aug = P_aug.llt().matrixL();
  MatrixXd B_aug = A_aug * sqrt(lambda_+n_aug_);
  // Ensure yaw deviations < 45 degrees (pi/4) and yawd deviations < 0.3
  for(int i = 0; i < n_aug_; i++) {
    if(B_aug(3,i) > M_PI/4) B_aug(3,i) = M_PI/4;
    if(B_aug(3,i) < -M_PI/4) B_aug(3,i) = -M_PI/4;
    if(B_aug(4,i) > 0.3) B_aug(4,i) = 0.3;
    if(B_aug(4,i) < -M_PI/4) B_aug(4,i) = -0.3;
  }
  Xsig_aug.block(0,0,n_aug_,1) = x_aug;
  Xsig_aug.block(0,1,n_aug_,n_aug_) = B_aug.colwise() + x_aug;
  Xsig_aug.block(0,n_aug_+1,n_aug_,n_aug_) = (B_aug.colwise() - x_aug) * -1.0;

  cout << endl << "Sigma point prediction..." << endl << endl
       << "Xsig_aug" << endl << Xsig_aug << endl << endl;

  // Predict new sigma points
  for(int i=0; i < 2*n_aug_+1; i++) {
    // Extract values for readability
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    // Predicted state values
    double px_pred, py_pred;
    if(fabs(v < 0.001)) {
      px_pred = px;
      py_pred = py;
    } else if(fabs(yawd) > 0.001) {
      px_pred = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_pred = py + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    } else {
      px_pred = px + v*delta_t*cos(yaw);
      py_pred = py + v*delta_t*sin(yaw);
    }
    double v_pred = v;
    double yaw_pred = yaw + yawd*delta_t;
    double yawd_pred = yawd;
    // Add noise
    px_pred += 0.5*nu_a*delta_t*delta_t*cos(yaw);
    py_pred += 0.5*nu_a*delta_t*delta_t*sin(yaw);
    v_pred += nu_a*delta_t;
    yaw_pred += 0.5*nu_yawdd*delta_t*delta_t;
    yawd_pred += nu_yawdd*delta_t;
    // Prevent extreme values of yaw and yawd
    while(yaw_pred > M_PI) yaw_pred -= 2. * M_PI;
    while(yaw_pred < -M_PI) yaw_pred += 2. * M_PI;
    if(yawd_pred > 2.0) yawd_pred = 2.0;
    if(yawd_pred < -2.0) yawd_pred = -2.0;
    // Store predicted sigma point in matrix
    Xsig_pred_(0,i) = px_pred;
    Xsig_pred_(1,i) = py_pred;
    Xsig_pred_(2,i) = v_pred;
    Xsig_pred_(3,i) = yaw_pred;
    Xsig_pred_(4,i) = yawd_pred;
  }
  cout << "Xsig_pred" << endl << Xsig_pred_ << endl << endl;
  // shift yaw by yaw_offset, and shift back after calculating x and P.
  // trying to keep yaw away from discontinuity at +- pi.
  double yaw_offset = - Xsig_pred_(3,0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Xsig_pred_(3,i) = Xsig_pred_(3,i) + yaw_offset;
    while(Xsig_pred_(3,i) > M_PI) Xsig_pred_(3,i) -= 2. * M_PI;
    while(Xsig_pred_(3,i) < -M_PI) Xsig_pred_(3,i) += 2. * M_PI;
  }
  // Calculate predicted state matrix
  x_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    x_ += weights(i) * Xsig_pred_.col(i);
  }
  // Prevent extreme values of v and yawd
  if(x_(2) > 10.0) x_(2) = 10.0;
  if(x_(4) > 2.0) x_(4) = 2.0;
  if(x_(4) < -2.0) x_(4) = -2.0;
  // Calculate predicted state covariance matrix
  P_.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    P_ += weights(i) * diff * diff.transpose();
  }
  // Undo offset
  x_(3) = x_(3) - yaw_offset;
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Xsig_pred_(3,i) = Xsig_pred_(3,i) - yaw_offset;
    while(Xsig_pred_(3,i) > M_PI) Xsig_pred_(3,i) -= 2. * M_PI;
    while(Xsig_pred_(3,i) < -M_PI) Xsig_pred_(3,i) += 2. * M_PI;
  }
  // Prevent negative v
  if(x_(2) < -0.01) {
    x_(2) = x_(2) * -1.0;
    x_(3) = x_(3) + M_PI;
  }
  // Prevent extreme values of yaw
  while(x_(3) > M_PI) x_(3) -= 2. * M_PI;
  while(x_(3) < -M_PI) x_(3) += 2. * M_PI;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Dimensions of measurement space
  int n_z = 2;
  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z,2*n_aug_+1);
  // Transform sigma points into measurement space
  for(int i = 0; i < 2*n_aug_+1; i++) {
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }
  // Calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    z_pred += weights(i) * Zsig.col(i);
  }
  // Measurement covariance
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    S += weights(i) * diff * diff.transpose();
  }
  S(0,0) += std_laspx_*std_laspx_;
  S(1,1) += std_laspy_*std_laspy_;
  // Cross-correlation
  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights(i) * x_diff * z_diff.transpose();
  }
  // Kalman gain
  MatrixXd K = Tc * S.inverse();
  // Update state mean and covariance
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  VectorXd x_diff = K * z_diff;
  cout << "Lidar x_diff: " << endl << x_diff << endl;
  //if(iterations > 10) {
    // After 10 iterations, when model has stabilized, don't let one measurement have too much impact.
    //if(x_diff(0) > 0.1) x_diff(0) = 0.1;
    //if(x_diff(0) < -0.1) x_diff(0) = -0.1;
    //if(x_diff(1) > 0.1) x_diff(1) = 0.1;
    //if(x_diff(1) < -0.1) x_diff(1) = -0.1;
    //if(x_diff(2) > 0.1) x_diff(2) = 0.2;
    //if(x_diff(2) < -0.1) x_diff(2) = -0.2;
    //if(x_diff(3) > 0.2) x_diff(3) = 0.2;
    //if(x_diff(3) < -0.2) x_diff(3) = -0.2;
    //if(x_diff(4) > 0.1) x_diff(4) = 0.1;
    //if(x_diff(4) < -0.1) x_diff(4) = -0.1;
  //}
  x_ += x_diff;
  P_ -= K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Dimensions of measurement space
  int n_z = 3;
  // Sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z,2*n_aug_+1);
  // Transform sigma points into measurement space
  for(int i = 0; i < 2*n_aug_+1; i++) {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double rho = sqrt(px*px + py*py);
    Zsig(0,i) = rho;
    Zsig(1,i) = atan2(py,px);
    Zsig(2,i) = v*(px*cos(yaw) + py*sin(yaw)) / rho;
  }
  // Calculate mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    z_pred += weights(i) * Zsig.col(i);
  }
  // Measurement covariance
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;
    while(diff(1) > M_PI) diff(1) -= 2. * M_PI;
    while(diff(1) < -M_PI) diff(1) += 2. * M_PI;
    S += weights(i) * diff * diff.transpose();
  }
  S(0,0) += std_radr_*std_radr_;
  S(1,1) += std_radphi_*std_radphi_;
  S(2,2) += std_radrd_*std_radrd_;
  // Cross-correlation
  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
    Tc += weights(i) * x_diff * z_diff.transpose();
  }
  // Kalman gain
  MatrixXd K = Tc * S.inverse();
  // Update state mean and covariance
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  VectorXd x_diff = K * z_diff;
  cout << "Radar x_diff: " << endl << x_diff << endl;
  //if(iterations > 10) {
    // After 10 iterations, when model has stabilized, don't let one measurement have too much impact.
    //if(x_diff(0) > 0.1) x_diff(0) = 0.1;
    //if(x_diff(0) < -0.1) x_diff(0) = -0.1;
    //if(x_diff(1) > 0.1) x_diff(1) = 0.1;
    //if(x_diff(1) < -0.1) x_diff(1) = -0.1;
    //if(x_diff(2) > 0.1) x_diff(2) = 0.1;
    //if(x_diff(2) < -0.1) x_diff(2) = -0.1;
    //if(x_diff(3) > 0.1) x_diff(3) = 0.1;
    //if(x_diff(3) < -0.1) x_diff(3) = -0.1;
    //if(x_diff(4) > 0.1) x_diff(4) = 0.1;
    //if(x_diff(4) < -0.1) x_diff(4) = -0.1;
  //}
  x_ += x_diff;
  P_ -= K * S * K.transpose();
}
