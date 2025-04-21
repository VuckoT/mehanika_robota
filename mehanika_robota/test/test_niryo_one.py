"""Modul za testiranje objekata iz modula mat_prostor
"""

"""
*** BIBLIOTEKE ***
"""
from mehanika_robota.roboti.niryo_one import niryo_one as n_one
import numpy as np
import pytest

"""
*** TESTOVI ***
"""
def test__unutar_opsega_aktuiranja() -> None:
    f = n_one._unutar_opsega_aktuiranja

    assert f(np.full(6, 0.0))
    assert f([0, -1, 1, 3, -0.5, 1.5])

    assert not f(np.full(6, 10))
    assert not f([1e6, -1, 1, 3, -0.5, 1.5])
    assert not f([0, 1e6, 1, 3, -0.5, 1.5])
    assert not f([0, -1, 1e6, 3, -0.5, 1.5])
    assert not f([0, -1, 1, 1e6, -0.5, 1.5])
    assert not f([0, -1, 1, 3, 1e6, 1.5])
    assert not f([0, -1, 1, 3, -0.5, 1e6])

def test_dir_kin() -> None:
    f = n_one.dir_kin
    
    # Pravilna upotreba
    assert np.allclose(
        f(np.deg2rad([-10, 10, -10, 23, 3, -50])),
        [[ 0.97990712676,  0.19413301394,  0.04576456974,  0.20610273805],
         [-0.19354881550,  0.87010431565,  0.45327401856, -0.03465441537],
         [ 0.04817550173, -0.45302411942,  0.89019563482,  0.41589554919],
         [           0.0,            0.0,            0.0,            1.0]]
    )

    assert np.allclose(
        f(np.deg2rad([-20, 21, -20, 56, 6, -66]), 4.5e-2),
        [[ 0.90380516113,  0.42792598717, 0.00394717768,  0.20113221550],
         [-0.42117771859,  0.88784301435, 0.18532164263, -0.07471586531],
         [ 0.07579947274, -0.16915712037, 0.98266998965,  0.41507223886],
         [           0.0,            0.0,           0.0,            1.0]]
    )

    assert np.allclose(
        f(np.deg2rad([-34, 20, -13, 2, 0, -50]), koord_sistem_prostor=False),
        [[0.82285805221, -0.55502476436,  0.12186934341, -0.22026236843],
         [0.44925617752,  0.50409022930, -0.73760553664,  0.30512350860],
         [0.34795619391,  0.66169521055,  0.66414300829, -0.27499273163],
         [          0.0,            0.0,            0.0,            1.0]]
    )
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f(np.full(6, 1e6))

def test_inv_kin() -> None:
    f = n_one.inv_kin
    
    assert np.allclose(
        f(
            [[ 0, 0, 1, 150e-3],
             [ 0, 1, 0, -150e-3],
             [-1, 0, 0,   88e-3],
             [ 0, 0, 0,       1]],
            0.001,
            0.001
        ),
        ([
            -0.785398163,
            -0.821959920,
            -0.594031841,
                     0.0,
            -0.154804566,
            -0.785398163],)
    )

    assert np.allclose(
        f(
            [[ 0, 0, 1,120e-3],
             [ 0, 1, 0, -40e-3],
             [-1, 0, 0,  80e-3],
             [ 0, 0, 0,      1]],
            0.001,
            0.001,
            2.3e-2
        ),
        ([
            -3.21750554397e-01,
            -6.49547831639e-01,
                -1.03360067053,
                           0.0,
             1.12352175372e-01,
            -3.21750554397e-01
        ],)
    )

    assert np.allclose(
        f(
            n_one.dir_kin(
                np.deg2rad([-10, 10, -10, 23, 3, -50]),
                koord_sistem_prostor=False
            ),
            0.001,
            0.001,
            koord_sistem_prostor=False
        ),
        ([-0.17467816344,
           0.17487084691,
          -0.17490083456,
           0.40724767479, 
           0.05231310843,
          -0.87861482098],
         [-0.15009127257,
           0.17688998239,
          -0.22153025660,
          -2.69079927651,
          -0.10311263021,
           2.21997396141])
    )