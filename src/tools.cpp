#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  if (estimations.size() == 0) {
    return rmse;
  }
  if (estimations.size() != ground_truth.size()) {
    throw "estimations and ground_truth are different length";
  }
  //cout << endl << "Calculating RMSE with " << estimations.size() << " residuals." << endl << endl;
  for(int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    VectorXd residual_squared = residual.array() * residual.array();
    rmse += residual_squared;
    //cout << "  Residual " << residual(0) << " = estimation " << estimations[i](0) << " - truth " << ground_truth[i](0) << endl;
    //cout << "  Residual^2 " << residual_squared(0) << endl;
    //cout << "  Total " << rmse(0) << endl << endl;
    if(true && i > 5 && (residual_squared(0) > 2 || residual_squared(1) > 2 || residual_squared(2) > 30 || residual_squared(3) > 30)) {
      cout << "Crazy Residual! " << endl;
      cout << "estimation" << endl << estimations[i] << endl;
      cout << "truth" << endl << ground_truth[i] << endl;
      throw "Crazy Residual";
    }
  }

  cout << "estimation" << endl << estimations[estimations.size() - 1] << endl;
  cout << "truth" << endl << ground_truth[estimations.size() - 1] << endl;

  rmse /= estimations.size();
  //cout << "Mean " << endl << rmse << endl;
  rmse = rmse.array().sqrt();
  //cout << "RMSE " << endl << rmse << endl << endl;
  return rmse;
}