# %%
import numpy as np

class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float, 
                       accel_variance: float) -> None:
        
        """
        Initialization Step

        Parameters:
            initial_x: Initial position
            initial_v: Initial velocity 
            accel_variance: IMU variance
            **
            Need to change for parameters:
                * initial_state (\hat{x}): The initial State Estimate Vector
                * initial_cov (P_{0,0}): The initial Estimate Covariance Matrix
            **
        """
        
        # mean of state GRV
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance

        # covariance of state GRV
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        """
        The Prediction of next state

        Parameters:
            dt: Delta t to be used between predictions

        Equations:
            # x = F x
            # P = F P F^t + G G^t a 
        """
        

        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)

        G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        """
        Measurement Update step

        Parameters:
            meas_value (z_n): The Measurements Vector
            meas_variance (R_n): Measurement Covariance Matrix

        Equations:
            # y = z - H x
            # S = H P H^t + R
            # K = P H^t S^-1
            # x = x + K y
            # P = (I - k H) * P
        """

        H = np.array([1,0]).reshape((1,2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

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

    accel = np.loadtxt('/Users/pal/Downloads/downstairs_accel_data_raw.csv', delimiter=',')

    DT = 0.01
    DT2 = DT * DT
    MEAS_EVERY_STEPS = 100
    TIME = [DT * i for i in range(accel.shape[0])]
    initial_v = DT * accel[0,0]

    mus = []
    covs = []
    x_accel = np.array([])
    x_pos = np.array([])

    real_x_pos = 0.0
    imu_variance_x = 93.1225e-6     # The calibration variance measured when sitting still
    real_v = 0.0
    
    kf = KF(initial_x=0.0, initial_v=initial_v, accel_variance=imu_variance_x)

    for step in range(accel.shape[0]):
        covs.append(kf.cov)
        mus.append(kf.mean)

        meas_x_accel = accel[step,0]
        # real_v = real_v + DT * meas_x_accel
        # real_x_pos = real_x_pos + DT * real_v + DT**2 * meas_x_accel

        real_v = 0.0
        real_x_pos = 0.0

        x_accel = np.append(x_accel, meas_x_accel) 
        x_pos = np.append(x_pos, real_x_pos)     

        kf.predict(dt=DT)
        if step != 0 and step % MEAS_EVERY_STEPS == 0:
            kf.update(meas_value=meas_x_accel, meas_variance=np.var(x_accel))
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
