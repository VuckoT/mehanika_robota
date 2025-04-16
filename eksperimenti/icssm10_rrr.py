import numpy as np
import mehanika_robota.mehanika.kinematika as kin
import mehanika_robota.mehanika.mat_prostor as mp

L1 = np.float64(0.6)
L2 = np.float64(0.4)
L3 = np.float64(0.2)

S1 = np.array([0, 0, 1, 0,      0, 0], dtype=float)
S2 = np.array([0, 0, 1, 0,    -L1, 0], dtype=float)
S3 = np.array([0, 0, 1, 0, -L1-L2, 0], dtype=float)

M = mp.SE3_sastavi(np.eye(3), [L1 + L2 + L3, 0, 0])

theta = np.array([-np.pi/6, np.pi/2, -np.pi/3], dtype=float)

Td = (
    mp.exp_vek_ugao(S1, theta[0])
    @ mp.exp_vek_ugao(S2, theta[1])
    @ mp.exp_vek_ugao(S3, theta[2])
    @ M
)

T1 = Td @ mp.inv(M)

r3 = np.array([L1 + L2,   0, 0], dtype=float)
p3 = np.array([      0, 0.1, 0], dtype=float)

phi12 = kin.paden_kahan2(S1, S2, r3, mp.SE3_proizvod_3D(T1, r3))

if isinstance(phi12, tuple):
    Phi = np.array([[*phi12[0], 0],
                    [*phi12[1], 0]], dtype=float)
else:
    Phi = np.array([[*phi12, 0]], dtype=float)

er = np.zeros(Phi.shape[0], dtype=float)
ep = np.zeros(Phi.shape[0], dtype=float)
Ts = np.zeros((Phi.shape[0], 4, 4), dtype=float)

for i in range(Phi.shape[0]):
    Phi[i, 2] = kin.paden_kahan1(S3, p3, mp.SE3_proizvod_3D(
        mp.exp_vek_ugao(S2, -Phi[i, 1])
        @ mp.exp_vek_ugao(S1, -Phi[i, 0])
        @ T1,
        p3
    ))
    
    Ts[i] = (
        mp.exp_vek_ugao(S1, Phi[i, 0])
        @ mp.exp_vek_ugao(S2, Phi[i, 1])
        @ mp.exp_vek_ugao(S3, Phi[i, 2])
        @ M
    )
    
    Rd, pd = mp.SE3_rastavi(Td)
    Rs, ps = mp.SE3_rastavi(Ts[i])

    er[i] = np.linalg.norm(Rd - Rs)
    ep[i] = np.linalg.norm(pd - ps)

np.set_printoptions(precision=16, suppress=True)

print(f'Solution = ({Phi})')
print(f'Orientational error = {er.mean()}')
print(f'Positional error = {ep.mean()}')