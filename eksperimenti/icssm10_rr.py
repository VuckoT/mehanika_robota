import numpy as np
import mehanika_robota.mehanika.kinematika as kin
import mehanika_robota.mehanika.mat_prostor as mp

L1 = np.float64(0.2)
L2 = np.float64(0.2)

S1 = np.array([0, 0, 1, 0,   0, 0], dtype=float)
S2 = np.array([0, 0, 1, 0, -L1, 0], dtype=float)

M = mp.SE3_sastavi(np.eye(3), [L1 + L2, 0, 0])

theta = np.array([np.pi/6, -np.pi/3], dtype=float)

Td = mp.exp_vek_ugao(S1, theta[0]) @ mp.exp_vek_ugao(S2, theta[1]) @ M
T1 = Td @ mp.inv(M)

r2 = np.array([L1,  0, 0], dtype=float)
p2 = np.array([ 0, 0.1, 0], dtype=float)

Phi = np.zeros(2)

Phi[0] = kin.paden_kahan1(S1, r2, mp.SE3_proizvod_3D(T1, r2))
Phi[1] = kin.paden_kahan1(S2, p2, mp.SE3_proizvod_3D(
    mp.exp_vek_ugao(S1, -Phi[0])@T1,
    p2
))

Ts = mp.exp_vek_ugao(S1, Phi[0]) @ mp.exp_vek_ugao(S2, Phi[1]) @ M

Rd, pd = mp.SE3_rastavi(Td)
Rs, ps = mp.SE3_rastavi(Ts)

er = np.linalg.norm(Rd - Rs)
ep = np.linalg.norm(pd - ps)

np.set_printoptions(precision=16, suppress=True)

print(f'Solution = ({Phi})')
print(f'Orientational error = {er}')
print(f'Positional error = {ep}')