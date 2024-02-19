# %%
import numpy as np

class KF:
    def __init__(self, initial_x: float, 
                       initial_v: float, 
                       accel_variance: float) -> None:
        
        # mean of state GRV
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance

        # covariance of state GRV
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P F^t + G G^t a

        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)

        G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P H^t + R
        # K = P H^t S^-1
        # x = x + K y
        # P = (I - k H) * P

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

    real_x = 0.0
    meas_variance = 93.1225e-6
    real_v = 0.0
    
    kf = KF(initial_x=0.0, initial_v=initial_v, accel_variance=meas_variance)

    for step in range(accel.shape[0]):
        covs.append(kf.cov)
        mus.append(kf.mean)

        meas_x = accel[step,0]
        real_x = real_x + DT * real_v

        kf.predict(dt=DT)
        if step != 0 and step % MEAS_EVERY_STEPS == 0:
            kf.update(meas_value=meas_x, meas_variance=0.5)

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(16,9))

    ax[0].set_title('X Position')
    ax[0].plot(TIME, [mu[0] for mu in mus], 'g')
    ax[0].plot(TIME, [real_x for i in range(accel.shape[0])], 'b')
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
