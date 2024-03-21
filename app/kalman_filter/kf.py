# %%
import numpy as np

class KF:
    def __init__(self, initial_state: np.ndarray) -> None:
        
        r"""
        Initialization Step

        Parameters:
            * initial_state (\hat{x}): The initial State Estimate Vector
            * initial_cov (P_{0,0}): The initial Estimate Covariance Matrix
        """
        
        # mean of state GRV
        self._x = initial_state

        # covariance of state GRV
        self._P = np.eye(9)

        # acceleration variance
        self._var_ax = 0.0000931225
        self._var_ay = 0.0000808201
        self._var_az = 0.0001729225

    def predict(self, dt: float) -> None:
        r"""
        The Prediction of next state

        Parameters:
            * dt: \delta{t} to be used between predictions

        Equations:
            * x = F x
            * P = F P F^t + G G^t \sigma_{accel}^2 
        """
        
        dt2 = dt*dt

        kinematic_mat = np.array([[1, dt, 0.5*dt2],        # Linear equaiton for 'x'
                                  [0, 1, dt],              # Linear equation for 'x dot'
                                  [0, 0, 1]])
        
        F = np.zeros((9,9))

        F[:3,:3] = kinematic_mat
        F[3:6,3:6] = kinematic_mat
        F[6:9,6:9] = kinematic_mat
        
        new_x = F.dot(self._x)

        G = np.array([[0.5*dt2, dt, 1]]).T

        self._Q = np.zeros((9,9))
        self._Q[:3,:3] = G.dot(G.T) * self._var_ax
        self._Q[3:6,3:6] = G.dot(G.T) * self._var_ay
        self._Q[6:9,6:9] = G.dot(G.T) * self._var_az

        # print(self._Q)

        new_P = F.dot(self._P).dot(F.T) + self._Q

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: np.ndarray, 
                     meas_variance: np.ndarray,
                     meas_func: np.ndarray):
        r"""
        Measurement Update step

        Parameters:
            * meas_value (z_n): The Measurements Vector
            * meas_variance (R_n): Measurement Covariance Matrix
            * meas_func (H) : Measurement function

        Equations:
            * y = z - H x
            * S = H P H^t + R
            * K = P H^t S^-1
            * x = x + K y
            * P = (I - k H) * P
        """

        H = meas_func                  ## Important for sensor fusion

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(9) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    accel = np.loadtxt('/Users/pal/Desktop/senior_design/code/app/data/noise_accel_data.csv', delimiter=",")

    DT = 0.01
    DT2 = DT * DT
    MEAS_EVERY_STEPS = 100
    TIME = [DT * i for i in range(accel.shape[0])]
    initial_x = np.array([[0,0,0,0,0,0,0,0,0]]).T

    mus = []
    covs = []
    x_accel = np.array([])
    x_pos = np.array([])

    real_x_pos = 0.0
    imu_variance_x = 93.1225e-6     # The calibration variance measured when sitting still
    real_v = 0.0

    R_imu = np.array([
        [1000, 0, 0],
        [0, 1000, 0],
        [0, 0, 1000]
    ])
    
    H_imu = np.zeros((3,9))

    H_imu[0,2] = 1
    H_imu[1,5] = 1
    H_imu[2,8] = 1

    kf = KF(initial_state=initial_x)

    for step in range(accel.shape[0]):
        covs.append(kf.cov)
        mus.append(kf.mean)

        meas_accel = accel[step]

        real_v = 0.0
        real_x_pos = 0.0

        x_accel = np.append(x_accel, meas_accel)
        x_pos = np.append(x_pos, real_x_pos)  

        kf.predict(dt=DT)
        if step != 0 and step % MEAS_EVERY_STEPS == 0:
            kf.update(meas_value=meas_accel, meas_variance=R_imu, meas_func=H_imu)
            # print(np.var(x_accel))

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16,9))

    ax[0].set_title('X Position')
    ax[0].plot(TIME, [mu[0] for mu in mus], 'g')
    ax[0].plot(TIME, x_pos, 'b')
    ax[0].fill_between(
        TIME,
        [mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)],
        [mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)],
        facecolor='r',
        alpha=0.2
        )
    ax[0].legend(['Predicted X', 'Real X', 'Error'])

    ax[1].set_title('X Velocity')
    ax[1].plot(TIME, [mu[1] for mu in mus], 'g')
    ax[1].plot(TIME, [real_v for i in range(accel.shape[0])], 'b')
    ax[1].fill_between(
        TIME,
        [mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)],
        [mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)],
        facecolor='r',
        alpha=0.2
        )
    ax[1].legend(['Predicted X', 'Real X', 'Error'])

    plt.show()
# %%
