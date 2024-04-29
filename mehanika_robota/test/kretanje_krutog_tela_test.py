"""Modul za testiranje objekata iz modula kretanje_krutog_tela
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
from .. import kretanje_krutog_tela as kkt
from .._alati import _testiraj_sve
import pytest

"""
*** POMOCNE FUNKCIJE ***
"""
def _uspesan_test_funkcije_poruka(funkcija_naziv: str) -> None:
    print(
        f"TEST FUNKCIJE mr.kkt.{funkcija_naziv}: USPESAN"
    )

"""
*** TESTOVI ***
"""   
def vek_norm_test() -> None:
    f = kkt.vek_norm
                
    # Pravilna upotreba
    assert np.allclose(
        f(np.array([1.0, 2.0, 3.0]), True)[0],
        [0.267261241912, 0.534522483825, 0.801783725737]
    )
    assert np.allclose(
        f(np.array([1.0, 2.0, 3.0]), True)[1],
        3.74165738677
    )
    assert np.allclose(
        f([[1], [2], [3]]),
        [[0.267261241912], [0.534522483825], [0.801783725737]]
    )
      
    assert np.allclose(
        f([(3, ), (4, ), (5, ), (0, )]),
        [[0.424264068712], [0.565685424949], [0.707106781187], [0.0]]
    )
    
    assert np.allclose(
        f([0, 0, 0], True)[0],
        [0.0, 0.0, 0.0]
    )
    
    assert np.allclose(
        f([0, 0, 0], True)[1],
        0.0
    )

    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f([[1, 2], [3, 4]])
        
    with pytest.raises(ValueError):
        f([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            
    _uspesan_test_funkcije_poruka("vek_norm")
    
def inv_test() -> None:
    f = kkt.inv    
    
    # Pravilna upotreba
    assert np.allclose(
        f([[0, 0, 1],
           [1, 0, 0],
           [0, 1, 0]]),
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]]
    )
    assert np.allclose(
        f([[0, 0, 1,   0],
           [1, 0, 0, 1.5],
           [0, 1, 0,   3],
           [0, 0, 0,   1]]),
        [[0.0, 1.0, 0.0, -1.5],
         [0.0, 0.0, 1.0, -3.0],
         [1.0, 0.0, 0.0,  0.0],
         [0.0, 0.0, 0.0,  1.0]]
    )
    
    _uspesan_test_funkcije_poruka("inv")

def lijeva_algebra_od_vek_test() -> None:
    f = kkt.lijeva_algebra_od_vek

    # Pravilna upotreba
    assert np.allclose(
        f([1, 2, 3]),
        [[ 0, -3,  2],
         [ 3,  0, -1],
         [-2,  1,  0]]
    )

    assert np.allclose(
        f([[1.2], [2], [3], [4.3], [5], [6]]),
        [[ 0.0, -3.0,  2.0, 4.3],
         [ 3.0,  0.0, -1.2, 5.0],
         [-2.0,  1.2,  0.0, 6.0],
         [ 0.0,  0.0,  0.0, 0.0]]
    )
    
    _uspesan_test_funkcije_poruka("lijeva_algebra_od_vek")
    
def vek_od_lijeve_algebre_test() -> None:
    f = kkt.vek_od_lijeve_algebre
    
    # Pravilna upotreba
    assert np.allclose(
        f([[ 0, -3,  2],
           [ 3,  0, -1],
           [-2,  1,  0]], True),
        [[1], 
         [2],
         [3]]
    )
    
    assert np.allclose(
        f([[ 0,   -3,    2,   4],
           [ 3,    0, -1.2, 5.4],
           [-2,  1.2,    0,   6],
           [ 0,    0,    0,   0]]),
        [1.2, 2.0, 3.0, 4.0, 5.4, 6.0]
    )
        
    _uspesan_test_funkcije_poruka("vek_od_lijeve_algebre")
    
def v_prostor_normiranje_test() -> None:
    f = kkt.v_prostor_normiranje
    
    # Pravilna upotreba
    assert np.allclose(
        f([[0], [0], [0], [4], [5], [6]], True)[0],
        [[0.0],
         [0.0],
         [0.0],
         [0.45584230584],
         [0.5698028823],
         [0.68376345876]]
    )
    assert np.allclose(
        f([[0], [0], [0], [4], [5], [6]], True)[1],
         8.774964387392123
    )
    assert np.allclose(
        f([4, 5, 6, 4, 5, 6], True)[0],
        [
            0.45584230584,
             0.5698028823,
            0.68376345876,
            0.45584230584,
             0.5698028823,
            0.68376345876
        ]
    )
    assert np.allclose(
        f([4, 5, 6, 4, 5, 6], True)[1],
         8.774964387392123
    )
    assert np.allclose(
        f([4, 5, 6, 4, 5, 6]),
        [
            0.45584230584,
             0.5698028823,
            0.68376345876,
            0.45584230584,
             0.5698028823,
            0.68376345876
        ]
    )
    assert np.allclose(
        f([[4], [5], [6], [1], [1], [1]]),
         [[0.45584230584],
           [0.5698028823],
          [0.68376345876],
          [0.11396057646],
          [0.11396057646],
          [0.11396057646]]
    )
    assert np.allclose(
        f([[4], [5], [6], [1], [1], [1]], True)[1],
         8.774964387392123
    )
    
    _uspesan_test_funkcije_poruka("v_prostor_normiranje")

def exp_test() -> None:
    f = kkt.exp
    
    # Pravilna upotreba
    assert np.allclose(
        f([[ 0, -3,  2],
           [ 3,  0, -1],
           [-2,  1,  0]]),
        [[-0.69492055764,  0.71352099053, 0.08929285886],
         [-0.19200697279, -0.30378504433, 0.93319235382],
         [ 0.69297816774,  0.63134969938, 0.34810747783]]
    )
    
    assert np.allclose(
        f([[ 0, -3,  2, 4],
           [ 3,  0, -1, 5],
           [-2,  1,  0, 6],
           [ 0,  0,  0, 0]]),
        [[-0.69492055764,  0.71352099053, 0.08929285886, 1.63585649718],
         [-0.19200697279, -0.30378504433, 0.93319235382, 5.28901902922],
         [ 0.69297816774,  0.63134969938, 0.34810747783, 6.59536848146],
         [           0.0,            0.0,           0.0,           1.0]]
    )

    _uspesan_test_funkcije_poruka("exp")
    
def SE3_rastavi_test() -> None:
    f = kkt.SE3_rastavi
    
    # Pravilna upotreba
    assert np.allclose(
        f([[1, 0,  0, 0],
           [0, 0, -1, 0],
           [0, 1,  0, 3],
           [0, 0,  0, 1]])[0],
        [[1, 0,  0],
         [0, 0, -1],
         [0, 1,  0]]
    )
    
    assert np.allclose(
        f([[1, 0,  0, 0],
           [0, 0, -1, 0],
           [0, 1,  0, 3],
           [0, 0,  0, 1]])[1],
        [0, 0, 3]
    )
    
    assert np.allclose(
        f([[1, 0,  0, 3],
           [0, 1,  0, 1],
           [0, 0,  1, 0],
           [0, 0,  0, 1]], True)[0],
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    
    assert np.allclose(
        f([[1, 0,  0, 3],
           [0, 1,  0, 1],
           [0, 0,  1, 0],
           [0, 0,  0, 1]], True)[1],
        [[3],
         [1],
         [0]]
    )
    
    _uspesan_test_funkcije_poruka("SE3_rastavi")

def log_test() -> None:
    f = kkt.log

    # Pravilna upotreba
    assert np.allclose(
        f([[0, 0, 1],
           [1, 0, 0],
           [0, 1, 0]]),
        [[           0.0, -1.20919957616,  1.20919957616],
         [ 1.20919957616,            0.0, -1.20919957616],
         [-1.20919957616,  1.20919957616,            0.0]]
    )
    
    assert np.allclose(
        f([[1, 0,  0, 0],
           [0, 0, -1, 0],
           [0, 1,  0, 3],
           [0, 0,  0, 1]]),
        [[0.0,            0.0,            0.0,           0.0],
         [0.0,            0.0, -1.57079632680, 2.35619449019],
         [0.0,  1.57079632680,            0.0, 2.35619449019],
         [0.0,            0.0,            0.0,           0.0]]
    )

    _uspesan_test_funkcije_poruka("log")
    
def SE3_sastavi_test() -> None:
    f = kkt.SE3_sastavi

    # Pravilna upotreba
    assert np.allclose(
        f(
            [[1, 0,  0],
             [0, 0, -1],
             [0, 1,  0]],
            [1, 2, 3]
        ),
        [[1, 0,  0, 1],
         [0, 0, -1, 2],
         [0, 1,  0, 3],
         [0, 0,  0, 1]]
    )
    
    assert np.allclose(
        f(
            [[1, 0,  0],
             [0, 0, -1],
             [0, 1,  0]],
            [[1],
             [2],
             [3]]
        ),
        [[1, 0,  0, 1],
         [0, 0, -1, 2],
         [0, 1,  0, 3],
         [0, 0,  0, 1]]
    )
    
    _uspesan_test_funkcije_poruka("SE3_sastavi")

def Ad_test() -> None:
    f = kkt.Ad

    # Pravilna upotreba
    assert np.allclose(
        f([[1, 0,  0, 0],
           [0, 0, -1, 0],
           [0, 1,  0, 3],
           [0, 0,  0, 1]]),
        [[1, 0,  0, 0, 0,  0],
         [0, 0, -1, 0, 0,  0],
         [0, 1,  0, 0, 0,  0],
         [0, 0,  3, 1, 0,  0],
         [3, 0,  0, 0, 0, -1],
         [0, 0,  0, 0, 1,  0]]
    )

    _uspesan_test_funkcije_poruka("Ad")
    
def osa_zavrtnja_parametarski_test() -> None:
    f = kkt.osa_zavrtnja_parametarski

    # Pravilna upotreba
    assert np.allclose(
        f(
            [[3],
             [0],
             [0]],
            [0, 0, 1],
            2,
            True
        ),
        [[0],
         [0],
         [1],
         [0],
        [-3],
         [2]]
    )
    
    assert np.allclose(
        f(
            [3, 0, 0],
            [[0],
             [0],
             [3]],
            0
        ),
        [0, 0, 1, 0, -3, 0]
    )
    
    assert np.allclose(
        f(
            [0, 0, 0],
            [[0],
             [0],
             [3]],
            0
        ),
        [0, 0, 1, 0, 0, 0]
    )
    
    # Praviilna ali nije preporucena upotreba
    assert np.allclose(
        f(
            [0, 0, 0],
            [[0],
             [0],
             [3]],
            np.inf
        ),
        [0, 0, 0, 0, 0, 1]
    )
    
    assert np.allclose(
        f(
            [0, 0, 3],
            [[0],
             [0],
             [0]],
            0
        ),
        [0, 0, 0, 0, 0, 0]
    )
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f([1, 0, 0], [1, 0, 0], -3)
    
    _uspesan_test_funkcije_poruka("osa_zavrtnja_parametarski")

def parametri_ose_zavrtnja_test() -> None:
    f = kkt.parametri_ose_zavrtnja
    
    # Pravilna upotreba
    assert np.allclose(
        f([1, 0, 0, 0, 0, 1])[:2],
        [[0, -1, 0], [1, 0, 0]]
    )
    assert np.allclose(f([1, 0, 0, 0, 0, 1])[2], 0)
    
    assert np.allclose(
        f([[1], [0], [0], [1], [0], [1]])[:2],
        [[[0], [-1], [0]], [[1], [0], [0]]]
    )
    assert np.allclose(f([[1], [0], [0], [1], [0], [1]])[2], 1)
    
    assert np.allclose(
        f([3, 4, 12, 0, 0, 26])[:2],
        [[0.61538461538, -0.46153846154,           0],
         [0.23076923077,  0.30769230769, 0.92307692308]]
    )
    assert np.allclose(f([3, 4, 12, 0, 0, 26])[2], 1.84615384615)

    
    _uspesan_test_funkcije_poruka("parametri_ose_zavrtnja")

def osa_zavrtnja_lin_v_test() -> None:
    f = kkt.osa_zavrtnja_lin_v

    # Pravilna upotreba
    assert np.allclose(
        f([[3],
           [0],
           [0]]),
        [[0],
         [0],
         [0],
         [1],
         [0],
         [0]]
    )
    
    assert np.allclose(
        f([0, 2, 0]),
        [0, 0, 0, 0, 1, 0]
    )

    _uspesan_test_funkcije_poruka("osa_zavrtnja_lin_v")

def _proj_SO3_test() -> None:
    f = kkt._proj_SO3

    # Pravilna upotreba
    assert np.allclose(
        f(np.array([[   0, -0.982,     0],
                    [1.01,      0,     0],
                    [0.02,      0, 0.991]])),
        [[0.0,           -1.0,            0.0],
         [0.9999500537,   0.0, -0.00999450328],
         [0.00999450328,  0.0,   0.9999500537]]
    )
    
    assert np.allclose(
        f(np.array([[1, 2, 3],
                    [4, 5, 0],
                    [0, 0, 6]])),
        [[-0.50686954644,  0.76165061122, -0.40369742299],
         [ 0.81546665968,  0.57548589858,  0.06188786217],
         [ 0.27945910227, -0.29783271642, -0.91279695617]]
    )

    _uspesan_test_funkcije_poruka("_proj_SO3")
    
def _proj_so3_test() -> None:
    f = kkt._proj_so3

    # Pravilna upotreba
    assert np.allclose(
        f(np.array([[0.03, -0.982,  0.4],
                    [0.99,   0.02,  1.1],
                    [-0.32, -1.02,    0]])),
            [[  0.0, -0.986, 0.36],
             [0.986,    0.0, 1.06],
             [-0.36,  -1.06,  0.0]]
    )

    _uspesan_test_funkcije_poruka("_proj_so3")
    
def _suprotan_znak_provera_test() -> None:
    f = kkt._suprotan_znak_provera
    
    # Pravilna upotreba
    assert f(2.3, -23)
    assert f(2.3, -23)
    assert f(0, 0)
    assert f(0.0000000001, 0.0000000001)
    assert f(0.0000000001, 0.0000000001)
    
    # Nepravilna upotreba
    assert not f(0, 2.3)
    assert not f(2.3, 0)
    assert not f(2.3, 2.3)
    assert not f(-2.3, -2.3)
    
    _uspesan_test_funkcije_poruka("_suprotan_znak_provera")
    
def _proj_lijeva_algebra_provera_test() -> None:
    f = kkt._proj_lijeva_algebra_provera

    # Pravilna upotreba
    assert f(np.zeros((3, 3))) is None
    assert f(np.zeros((4, 4))) is None
    assert f(np.array([[0, -1.0000000001, -0.000000009],
                       [1,             0,           -2],
                       [0,             2,            0]])) is None
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f(np.array([[0, -1, -2],
                           [1,  0,  3],
                           [2,  3,  0]]))
        
    with pytest.raises(ValueError):
        assert f(np.array([[0, -1,  2],
                           [1,  0, -3],
                           [2,  3,  0]]))
        
    with pytest.raises(ValueError):
        assert f(np.array([[0, 1, -2],
                           [1, 0, -3],
                           [2, 3,  0]]))
        
    with pytest.raises(ValueError):
        assert f(np.array([[0, 1,  2],
                           [1, 0, -3],
                           [2, 3,  0]]))

    _uspesan_test_funkcije_poruka("_proj_lijeva_algebra_provera")

def proj_grupa_test() -> None:
    f = kkt.proj_grupa

    # Pravilna upotreba
    assert np.allclose(
        f([[   0, -0.982,     0],
           [1.01,      0,     0],
           [0.02,      0, 0.991]], "SO3"),
        [[0.0,           -1.0,            0.0],
         [0.9999500537,   0.0, -0.00999450328],
         [0.00999450328,  0.0,   0.9999500537]]
    )
    
    assert np.allclose(
        f([[1, 2, 3],
           [4, 5, 0],
           [0, 0, 6]], "SO3"),
        [[-0.50686954644,  0.76165061122, -0.40369742299],
         [ 0.81546665968,  0.57548589858,  0.06188786217],
         [ 0.27945910227, -0.29783271642, -0.91279695617]]
    )

    assert np.allclose(
        f([[1.03, -0.982,      0,  1.2],
           [0.01,   0.02,      0, 3.42],
           [   0,      0, -0.971,    2],
           [0.23,   0.01,      0, 1.01]], "SE3"),
        [[0.72689794992, -0.68674549173, -0.0,  1.2],
         [0.68674549173,  0.72689794992,  0.0, 3.42],
         [          0.0,            0.0,  1.0,  2.0],
         [          0.0,            0.0,  0.0,  1.0]]
    )
    
    assert np.allclose(
        f([[1, 2, 3, 1],
           [4, 5, 0, 2],
           [0, 0, 6, 3],
           [0, 0, 2, 2]], "SE3"),
        [[-0.50686954644,  0.76165061122, -0.40369742299, 1.0],
         [ 0.81546665968,  0.57548589858,  0.06188786217, 2.0],
         [ 0.27945910227, -0.29783271642, -0.91279695617, 3.0],
         [           0.0,            0.0,            0.0, 1.0]]
    )

    assert np.allclose(
        f([[0.03, -0.982,  0.4],
           [0.99,   0.02,  1.1],
           [-0.32, -1.02,    0]], "so3"),
        [[  0.0, -0.986, 0.36],
         [0.986,    0.0, 1.06],
         [-0.36,  -1.06,  0.0]]
    )
    
    assert np.allclose(
        f([[ 0.03, -1.03,  0.5, 2.3],
           [    1, 0.001,  2.1, 1.2],
           [-0.44, -2.02, 0.02,   0],
           [-0.02,  0.02,    0,   0]], "se3"),
        [[  0.0, -1.015, 0.47, 2.3],
         [1.015,    0.0, 2.06, 1.2],
         [-0.47,  -2.06,  0.0, 0.0],
         [  0.0,    0.0,  0.0, 0.0]]
    )
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f(np.eye(3), "asd")
        
    with pytest.raises(ValueError):
        f(np.eye(4), "asd")
        
    with pytest.raises(ValueError):
        f(np.eye(3), 2.3)

    _uspesan_test_funkcije_poruka("proj_grupa")

def Rot_test() -> None:
    f = kkt.Rot

    # Pravilna upotreba
    assert np.allclose(
        f('x', np.pi/4, "SO3"),
        [[1.0,           0.0,            0.0],
         [0.0, 0.70710678119, -0.70710678119],
         [0.0, 0.70710678119,  0.70710678119]]
    )
    
    assert np.allclose(
        f('y', np.pi/4, "SO3"),
        [[ 0.70710678119, 0.0, 0.70710678119],
         [           0.0, 1.0,           0.0],
         [-0.70710678119, 0.0, 0.70710678119]]
    )

    assert np.allclose(
        f('z', np.pi/4, "SO3"),
        [[0.70710678119, -0.70710678119, 0.0],
         [0.70710678119,  0.70710678119, 0.0],
         [          0.0,            0.0, 1.0]]
    )
    
    assert np.allclose(
        f('y', np.pi/2),
        [[ 0.0, 0.0, 1.0, 0.0],
         [ 0.0, 1.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0],
         [ 0.0, 0.0, 0.0, 1.0]]
    )
    
    assert np.allclose(f('x', 0), np.eye(4))
    
    assert np.allclose(f('y', 0), np.eye(4))
    
    assert np.allclose(f('z', 0), np.eye(4))
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f('x', [1, 2])
        
    with pytest.raises(ValueError):
        assert f('s', 2.3)
    
    with pytest.raises(ValueError):
        assert f('x', 2.3, "asd")
    
    _uspesan_test_funkcije_poruka("Rot")

def Trans_test() -> None:
    f = kkt.Trans

    # Pravilna upotreba
    assert np.allclose(
        f([1, 2, 3]),
        [[1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 2.0],
         [0.0, 0.0, 1.0, 3.0],
         [0.0, 0.0, 0.0, 1.0]]
    )

    _uspesan_test_funkcije_poruka("Trans")

def homogeni_vek_test() -> None:
    f = kkt.homogeni_vek
    
    # Pravilna upotreba
    assert np.allclose(
        f([1, 2, 3]),
        [1, 2, 3, 1]
    )

    assert np.allclose(
        f([[1], [2], [3]]),
        [[1], [2], [3], [1]]
    )
    
    _uspesan_test_funkcije_poruka("homogeni_vek")

def SE3_proizvod_3D_test() -> None:
    f = kkt.SE3_proizvod_3D
    
    # Pravilna upotreba
    assert np.allclose(
        f(
            [[0, -1, 0, 1],
             [1,  0, 0, 0],
             [0,  0, 1, 0],
             [0,  0, 0, 1]],
            [1, 2, 3]
        ),
        [-1, 1, 3]
    )
    
    assert np.allclose(
        f(
            [[0, -1, 0, 1],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]],
            [[1],
             [2],
             [3]]
        ),
        [[-1],
         [ 1],
         [ 3]]
    )

    _uspesan_test_funkcije_poruka("SE3_proizvod_3D")

def testiraj_sve() -> None:
    """Testira sve objekte iz modula i obavestava o uspesnosti testa
    """
    _testiraj_sve(globals())