#include "kalman.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <GeographicLib/UTMUPS.hpp>


std::vector<double> readSerialData() {
    std::vector<double> data;
    // code for parsing serial data
    return data;
}

int main() {
    int state_dim = 9;
    int measurement_dim = 3;

    double lat = 40.425869, lon = -86.908066;  // West Lafayette, IN
    int zone;
    bool northp;
    double x, y;
    GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, x, y);

    std::cout << "Zone: " << zone << " " << (northp ? "North" : "South") << "\n";
    std::cout << "Easting: " << x << "\n";
    std::cout << "Northing: " << y << "\n";

    KalmanFilter kf(state_dim, measurement_dim);

    while (true) {
        std::vector<double> data = readSerialData();

        Eigen::VectorXd measurement(measurement_dim);
        for (int i = 0; i < measurement_dim; i++) {
            measurement(i) = data[i];
        }

        Eigen::MatrixXd H(measurement_dim, state_dim);
        Eigen::MatrixXd R(measurement_dim, measurement_dim);

        if (data[0] == "GPS") {
            H << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0;

            R << 10, 0, 0,
                 0, 10, 0,
                 0, 0, 10;

        } else if (data[0] == "IMU") {
            H << 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1;

            R << 1000, 0, 0,
                 0, 1000, 0,
                 0, 0, 1000;
        }

        kf.setH(H);
        kf.setR(R);

        // update filter with measurement
        kf.update(measurement);

        Eigen::VectorXd state = kf.getState()

        std::ofstream file("state.csv", std::ios::app);

        if (!file) {
            std::cerr << "Unable to open file";
            return 1;
        }

        for (int i = 0; i < state.size(); ++i) {
            file << state(i);
            if (i != state.size() - 1) {
                file << ", ";
            }
        }
        file << "\n";
        file.close();
    }

    return 0;
}