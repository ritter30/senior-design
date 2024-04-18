#include "my_kalman.hpp"
#include <Eigen/Dense>
#include <proj.h>
#include <fstream>
#include <list>
#include <string>

Eigen::Vector3d ConvertToUTM(const Eigen::VectorXd&);

int main() {

    std::ofstream file("kf_state.csv");

    // Define the initial state
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    Eigen::VectorXd z_n = Eigen::VectorXd::Zero(3);

    // Create a KF instance
    KF kf(initial_state);

    // Define H matrix for GPS
    Eigen::MatrixXd H(3, 9);
    H << 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0;

    // Define R matrix for GPS
    Eigen::MatrixXd R(3, 3);
    R << 10, 0, 0,
        0, 10, 0,
        0, 0, 10;

    float dt = 1;

    // Simulate the Kalman filter operation
    while (true) {
        // Get a measurement
        Eigen::VectorXd z_n = Eigen::VectorXd::Zero(3);
        // Eigen::VectorXd z_n = getMeasurement();

        std::string data[3] = {"GPS", "40.42", "-86.92"};

        if (data[0] == "GPS") {
            H << 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0;

            R << 10, 0, 0,
                 0, 10, 0,
                 0, 0, 10;

            dt = 1;

        } else if (data[0] == "IMU") {
            H << 0, 0, 1, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1;

            R << 1000, 0, 0,
                 0, 1000, 0,
                 0, 0, 1000;

            dt = 0.01;
        }

        // Eigen::VectorXd z_utm = ConvertToUTM(z_n);

        // Predict the next state
        kf.predict(dt);

        // Update the state based on the measurement
        // kf.update(z_utm, R, H);

        Eigen::VectorXd state = kf.getStateVector();
        for (int i = 0; i < state.size(); ++i) {
            file << state[i];
            if (i != state.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    return 0;
}

// Assuming z_n is an Eigen::VectorXd
Eigen::Vector3d ConvertToUTM(const Eigen::VectorXd& z_n) {
PJ_CONTEXT* context = proj_context_create();
if (!context) {
    // Handle error: proj_context_create failed
    return Eigen::Vector3d::Zero();
}

PJ* proj = proj_create_crs_to_crs(
    context, 
    "EPSG:4326",
    "+proj=utm +zone=16 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
    NULL);
if (!proj) {
    proj_context_destroy(context);
    // Handle error: proj_create_crs_from_string failed
    return Eigen::Vector3d::Zero();
}

// PJ_COORD src_coord, dst_coord;
// src_coord.enu.e = z_n[0];
// src_coord.enu.n = z_n[1];
// src_coord.enu.u = z_n[2];

double lon = -86.92;
double lat = 40.42;
double alt = 0;

int ret = proj_trans_generic(proj, PJ_FWD, 
                             &lon, sizeof(lon), 1,
                             &lat, sizeof(lat), 1,
                             &alt, sizeof(alt), 1,
                             NULL, sizeof(double), 0);

proj_destroy(proj);
proj_context_destroy(context);

if (ret != 0) {
    // Handle error: proj_transform failed
    return Eigen::Vector3d::Zero();
}

// Eigen::Vector3d z_utm(dst_coord.enu.e, dst_coord.enu.n, dst_coord.enu.u);

printf("UTM coordinates: (%f, %f)\n", lon, lat);

return Eigen::Vector3d::Zero(3);
}