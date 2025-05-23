import numpy as np
import mehanika_robota.mehanika.kinematika as kin
import mehanika_robota.mehanika.mat_prostor as mp

L1 = np.float64(0.3)
L2 = np.float64(0.3)
L3 = np.float64(0.1)

S1 = np.array([0, 0, 1, 0, 0, 0], dtype=float)
S2 = np.array([0, 0, 0, 0, 0, 1], dtype=float)
S3 = np.array([0, 0, 0, 0, 1, 0], dtype=float)

M = mp.SE3_sastavi(np.eye(3), [0, L3, L1 + L2])
    
theta = np.array([np.pi, 0.05, 0.1], dtype=float)

Td = (
    mp.exp_vek_ugao(S1, theta[0])
    @ mp.exp_vek_ugao(S2, theta[1])
    @ mp.exp_vek_ugao(S3, theta[2])
    @ M
)

T1 = Td @ mp.inv(M)

r1 = np.array([0,   0, 0.1], dtype=float)
p1 = np.array([0, 0.1,   0], dtype=float)

Phi = np.zeros(3)

phi32_negative = kin.pardos_gotor2(
    S3,
    S2,
    r1,
    mp.SE3_proizvod_3D(mp.inv(T1), r1)
)

Phi[2], Phi[1] = -phi32_negative[0], -phi32_negative[1]

Phi[0] = kin.paden_kahan1(
    S1,
    p1,
    mp.SE3_proizvod_3D(
        T1
        @ mp.exp_vek_ugao(S3, -Phi[2])
        @ mp.exp_vek_ugao(S2, -Phi[1]),
        p1
    )
)

Ts = (
    mp.exp_vek_ugao(S1, Phi[0])
    @ mp.exp_vek_ugao(S2, Phi[1])
    @ mp.exp_vek_ugao(S3, Phi[2])
    @ M
)

Rd, pd = mp.SE3_rastavi(Td)
Rs, ps = mp.SE3_rastavi(Ts)

er = np.linalg.norm(Rd - Rs)
ep = np.linalg.norm(pd - ps)

np.set_printoptions(precision=16, suppress=True)

print(f'Solution = ({Phi})')
print(f'Orientational error = {er}')
print(f'Positional error = {ep}')