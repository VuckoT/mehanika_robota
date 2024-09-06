"""Modul za testiranje objekata iz modula kretanje_krutog_tela
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
from mehanika_robota.roboti.niryo_one import niryo_one as n_one
import pytest

"""
*** TESTOVI ***
"""
def test__unutar_opsega_aktuiranja() -> None:
    f = n_one._unutar_opsega_aktuiranja

    assert f(np.full(6, 0.0))            == True
    assert f([0, -1, 1, 3, -0.5, 1.5])   == True

    assert f(np.full(6, 10))             == False
    assert f([1e6, -1, 1, 3, -0.5, 1.5]) == False
    assert f([0, 1e6, 1, 3, -0.5, 1.5])  == False
    assert f([0, -1, 1e6, 3, -0.5, 1.5]) == False
    assert f([0, -1, 1, 1e6, -0.5, 1.5]) == False
    assert f([0, -1, 1, 3, 1e6, 1.5])    == False
    assert f([0, -1, 1, 3, -0.5, 1e6])   == False

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
        f(np.deg2rad([-34, 20, -13, 2, 0, -50]), False),
        [[0.82285805221, -0.55502476436,  0.12186934341, -0.22026236843],
         [0.44925617752,  0.50409022930, -0.73760553664,  0.30512350860],
         [0.34795619391,  0.66169521055,  0.66414300829, -0.27499273163],
         [           0.0,           0.0,            0.0,            1.0]]
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
            -0.78539816340,
            -0.82195991980,
            -0.59403184149,
                       0.0,
            -0.15480456550,
            -0.78539816340],)
    )
    
    assert np.allclose(
        f(
            n_one.dir_kin(np.deg2rad([-10, 10, -10, 23, 3, -50]), False),
            0.001,
            0.001,
            False
        ),
        ([-0.17467816344,
           0.17487084691,
          -0.17490083456,
           0.40724767479, 
           0.05231310843,
          -0.87861482098],
         [-0.14969008692, 
           0.17720291579,
          -0.22168025901,
          -2.67705350678,
          -0.10310849559,
           2.20607801820])
    )