import numpy as np
import mehanika_robota.mehanika.kinematika as kin
import mehanika_robota.mehanika.kretanje_krutog_tela as kkt

L1 = np.float64(0.1)
L2 = np.float64(0.2)
L3 = np.float64(0.1)

S1 = np.array([0, 0, 1, 0,      0, 0], dtype=float)
S2 = np.array([0, 0, 0, 0,      1, 0], dtype=float)
S3 = np.array([0, 0, 1, 0, -L1-L2, 0], dtype=float)

M = kkt.SE3_sastavi(np.eye(3), [L1 + L2 + L3, 0, 0])

theta = np.array([0.0, 0.1, -3*np.pi/2], dtype=float)

Td = (
    kkt.exp_vek_ugao(S1, theta[0])
    @ kkt.exp_vek_ugao(S2, theta[1])
    @ kkt.exp_vek_ugao(S3, theta[2])
    @ M
)

T1 = Td @ kkt.inv(M)

r1 = np.array([      0,   0, 0.1], dtype=float)
r3 = np.array([L1 + L2,   0,   0], dtype=float)
p3 = np.array([      0, 0.1,   0], dtype=float)

phi2 = kin.pardos_gotor3(
    S2,
    r3,
    r1,
    np.linalg.norm(kkt.SE3_proizvod_3D(T1, r3) - r1)
)

if isinstance(phi2, tuple):
    Phi = np.array([[0, phi2[0], 0],
                    [0, phi2[1], 0]], dtype=np.float64)
else:
    Phi = np.array([[0, phi2, 0]], dtype=np.float64)

er = np.zeros(Phi.shape[0], dtype=float)
ep = np.zeros(Phi.shape[0], dtype=float)
Ts = np.zeros((Phi.shape[0], 4, 4), dtype=float)

for i in range(Phi.shape[0]):
    Phi[i, 0] = kin.paden_kahan1(
        S1,
        Phi[i, 1]*S2[3:] + r3,
        kkt.SE3_proizvod_3D(T1, r3)
    )
    
    Phi[i, 2] = kin.paden_kahan1(
        S3,
        p3,
        kkt.SE3_proizvod_3D(
            kkt.exp_vek_ugao(S2, -Phi[i, 1])
            @ kkt.exp_vek_ugao(S1, -Phi[i, 0])
            @ T1,
            p3
        )
    )

    Ts[i] = (
        kkt.exp_vek_ugao(S1, Phi[i, 0])
        @ kkt.exp_vek_ugao(S2, Phi[i, 1])
        @ kkt.exp_vek_ugao(S3, Phi[i, 2])
        @ M
    )
    
    Rd, pd = kkt.SE3_rastavi(Td)
    Rs, ps = kkt.SE3_rastavi(Ts[i])

    er[i] = np.linalg.norm(Rd - Rs)
    ep[i] = np.linalg.norm(pd - ps)

np.set_printoptions(precision=16, suppress=True)

print(f'Solution = ({Phi})')
print(f'Orientational error = {er.mean()}')
print(f'Positional error = {ep.mean()}')