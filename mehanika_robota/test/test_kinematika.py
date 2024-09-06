"""Modul za testiranje objekata iz modula kinematika
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
import mehanika_robota.mehanika.kinematika as kin
import pytest

"""
*** TESTOVI ***
"""
def test_PadenKahanError() -> None:
    f = kin.PadenKahanError
    
    standardna_poruka = "Paden-Kahanov podproblem nema resenja"
    dodatna_poruka = "Molimo, izvrsite dodatne korekcije"
    for i in range(1, 4):
        with pytest.raises(
            f,
            match=f"{i}. " + standardna_poruka
        ) as poruka_greske:
            raise f(i)
        
        with pytest.raises(
            f,
            match=f"{i}. " + standardna_poruka  + f". {dodatna_poruka}"
        ) as poruka_greske:
            raise f(i, dodatna_poruka)
        
        with pytest.raises(f):
            raise f(float(i))
            
    with pytest.raises(ValueError):
        f(0)
        
    with pytest.raises(ValueError):
        f(4)

    with pytest.raises(ValueError):
        f(0.0)
    
def test_dir_kin() -> None:
    f = kin.dir_kin

    # Pravilna upotreba
    assert np.allclose(
        f(
            [[-1, 0,  0, 0],
             [ 0, 1,  0, 6],
             [ 0, 0, -1, 2],
             [ 0, 0,  0, 1]],
            [[0, 0,    0],
             [0, 0,    0],
             [1, 0,   -1],
             [4, 0,   -6],
             [0, 3,    0],
             [0, 0, -0.1]],
            [np.pi/2, 3, np.pi],
            vek_kolona=True
        ),
        [[0.0, 1.0,  0.0,          -5.0],
         [1.0, 0.0,  0.0,           4.0],
         [0.0, 0.0, -1.0, 1.68584073464],
         [0.0, 0.0,  0.0,           1.0]]
    )

    assert np.allclose(
        f(
            [[-1, 0,  0, 0],
             [ 0, 1,  0, 6],
             [ 0, 0, -1, 2],
             [ 0, 0,  0, 1]],
            [[0, 0, -1, 2, 0,   0],
             [0, 0,  0, 0, 4,   0],
             [0, 0,  1, 0, 0, 0.1]],
            [[np.pi/2], [3], [np.pi]],
            koord_sistem_prostor=False
        ),
        [[0.0, 1.0,  0.0,          -5.0],
         [1.0, 0.0,  0.0,           4.0],
         [0.0, 0.0, -1.0, 1.68584073464],
         [0.0, 0.0,  0.0,           1.0]]
    )
    
    assert np.allclose(
        f(
            [[-1, 0,  0, 0],
             [ 0, 1,  0, 6],
             [ 0, 0, -1, 2],
             [ 0, 0,  0, 1]],
            [0, 0, -1, 2, 0, 0],
            np.pi/2,
        ),
        [[0.0, 1.0,  0.0,  8.0],
         [1.0, 0.0,  0.0, -2.0],
         [0.0, 0.0, -1.0,  2.0],
         [0.0, 0.0,  0.0,  1.0]]
    )
    
    assert np.allclose(
        f(
            [[ 1, 0,  0,    0],
             [ 0, 1,  0,    0],
             [ 0, 0,  1, 0.91],
             [ 0, 0,  0,    1]],
            [[0, 0, 1,    0, 0,     0],
             [0, 5, 0, 4.55, 0,     0],
             [0, 0, 1,    0, 0,     0],
             [0, 1, 0, 0.36, 0, 0.045],
             [0, 0, 1,    0, 0,     0],
             [0, 1, 0, 0.06, 0,     0],
             [0, 0, 1,    0, 0,     0]],
            [0, np.pi/4, 0, -np.pi/4, 0, -np.pi/2, 0],
            koord_sistem_prostor=False
        ),
        [[0.0, 0.0, -1.0, 0.31572853481],
         [0.0, 1.0,  0.0,           0.0],
         [1.0, 0.0,  0.0,  0.6570889245],
         [0.0, 0.0,  0.0,           1.0]]
    )

    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f(
            [[-1, 0,  0, 0],
             [ 0, 1,  0, 6],
             [ 0, 0, -1, 2],
             [ 0, 0,  0, 1]],
            [[0, 0, -1, 2, 0,   0],
             [0, 0,  0, 0, 1,   0],
             [0, 0,  1, 0, 0, 0.1]],
            [np.pi/2, 3]
        )
        
    with pytest.raises(ValueError):
        f(
            [[-1, 0,  0, 0],
             [ 0, 1,  0, 6],
             [ 0, 0, -1, 2],
             [ 0, 0,  0, 1]],
            [[0, 0, -1, 2, 0],
             [0, 0,  0, 0, 1],
             [0, 0,  1, 0, 0]],
            [np.pi / 2, 3, np.pi],
        )

def test_jakobijan() -> None:
    f = kin.jakobijan

    # Pravilna upotreba
    assert np.allclose(
        f(
            [[0,   1,  0,   1],
             [0,   0,  5,   0],
             [1,   0,  0,   0],
             [0,   2,  0, 0.2],
             [0.2, 0, 10, 0.3],
             [0.2, 3,  5, 0.4]],
            [0.2, 1.1, 0.1],
            vek_kolona=True
        ),
        [[0.0, 0.98006657784, -0.09011563789,  0.95749426473],
         [0.0, 0.19866933080,  0.44455439845,  0.28487556542],
         [1.0,           0.0,  0.89120736006, -0.04528405058],
         [0.0, 1.95218638245, -2.21635215690, -0.51161537298],
         [0.2, 0.43654132470, -2.43712572765,  2.77535713396],
         [0.2, 2.96026613384,  3.23573065328,  2.22512443354]]
    )

    assert np.allclose(
        f(
            [[0, 0, 1,   0, 0.2, 0.2],
             [1, 0, 0,   2,   0,   3],
             [0, 1, 0,   0,   2,   1],
             [1, 0, 0, 0.2, 0.3, 0.4]],
            [[1.1],
             [0.1],
             [1.2]],
            koord_sistem_prostor=False
        ),
        [[-0.04528405058,  0.99500416529,            0.0, 1.0],
         [ 0.74359312656,  0.09304864640,  0.36235775448, 0.0],
         [-0.66709715702,  0.03617541268, -0.93203908597, 0.0],
         [ 2.32586047146,  1.66809000495,  0.56410830804, 0.2],
         [-1.44321167182,  2.94561274991,  1.43306521429, 0.3],
         [-2.06639564876,  1.82881722462, -1.58868627853, 0.4]]
    )
    
    assert np.allclose(
        f([9, 0, 0, 0, 0, 9]),
        [[1], [0], [0], [0], [0], [1]]
    )
    
    assert np.allclose(
        f(
            [[1, 0, 0, 0, 0, 1],
             [0, 0, 0, 2, 3, 0]],
          2.3
        ),
        [[1,              0],
         [0,              0],
         [0,              0],
         [0,  0.55470019623],
         [0, -0.55437515962],
         [1,  0.62046424128]]
    )

    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f(
            [[0, 0, -1, 2, 0,   0],
             [0, 0,  0, 0, 1,   0],
             [0, 0,  1, 0, 0, 0.1]],
            [np.pi/2, 3, 4],
        )
        
    with pytest.raises(ValueError):
        f(
            [[0, 0, -1, 2, 0,   0],
             [0, 0,  0, 0, 1,   0],
             [0, 0,  1, 0, 0, 0.1]],
            [np.pi/2],
        )
        
    with pytest.raises(ValueError):
        f(
            [[0, 0, -1, 2, 0,   0],
             [0, 0,  0, 0, 1,   0],
             [0, 0,  1, 0, 0, 0.1]],
            np.pi/2,
        )
        
    with pytest.raises(ValueError):
        f(
            [[0, 0, -1, 2, 0],
             [0, 0,  0, 0, 1],
             [0, 0,  1, 0, 0]],
            [np.pi / 2, 3, np.pi],
            koord_sistem_prostor=False
        )

def test_manip() -> None:
    f = kin.manip
    
    J = [[0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [1, 1, 1, 0, 0, 0],
         [2, 0, 0, 4, 5, 0],
         [0, 2, 3, 0, 5, 0],
         [0, 1, 3, 4, 0, 1]]
    
    # Pravilna upotreba
    assert np.allclose(
        f(J),
        (2.0, 49055.5030446)
    )

    assert np.allclose(
        f(J, "J_omega"),
        (1.73205080757, 3.0)
    )

    assert np.allclose(
        f(J, "J_v"),
        (151.400132100, 4.90676068982)
    )
    
    assert np.allclose(
        f(J, elipsoid_sile=True),
        (0.5, 49055.5030446)
    )

    assert np.allclose(
        f(J, "J_omega", True),
        (0.57735026919, 3.0)
    )
    
    assert np.allclose(
        f(J, "J_v", True),
        (0.00660501405, 4.90676068982)
    )
    assert np.allclose(
        (
            f(J).V*f(J, elipsoid_sile=True).V,
            f(J, "J_omega").V*f(J, "J_omega", True).V,
            f(J, "J_v").V*f(J, "J_v", True).V,    
        ),
        (1, 1, 1)
    )
    
    J3 = [[0, 0, 0],
          [0, 0, 0],
          [1, 1, 1],
          [2, 0, 0],
          [0, 2, 3],
          [0, 1, 3]]

    assert np.allclose(
        f(J3),
        (0.0, np.inf)
    )

    assert np.allclose(
        f(J3, "J_omega"),
        (0.0, np.inf)
    )

    assert np.allclose(
        f(J3, "J_v"),
        (6.0, 56.7601597865)
    )
    
    assert np.allclose(
        f(J3, elipsoid_sile=True),
        (np.inf, np.inf)
    )

    assert np.allclose(
        f(J3, "J_omega", True),
        (np.inf, np.inf)
    )
    
    assert np.allclose(
        f(J3, "J_v", True),
        (0.166666666666, 56.7601597865)
    )
    assert np.allclose(
        f(J3, "J_v").V*f(J3, "J_v", True).V,    
        1
    )
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f(J3, "asd")
    
    with pytest.raises(ValueError):
        assert f(J3*3)

def test_inv_kin() -> None:
    f = kin.inv_kin
    
    # Pravilna upotreba
    M = [[-1, 0,  0, 0],
         [ 0, 1,  0, 6],
         [ 0, 0, -1, 2],
         [ 0, 0,  0, 1]]
    S_lista_prostor = [[0, 0,    0],
                       [0, 0,    0],
                       [1, 0,   -1],
                       [4, 0,   -6],
                       [0, 3,    0],
                       [0, 0, -0.1]]
    teta_lista0 = [1.5, 2.5, 3]
    Tk = [[0, 1,  0,     -5],
          [1, 0,  0,      4],
          [0, 0, -1, 1.6858],
          [0, 0,  0,      1]]

    tol_omega = 0.01
    tol_v = 0.001

    assert np.allclose(
        f(
            M,
            S_lista_prostor,
            teta_lista0,
            Tk,
            tol_omega,
            tol_v,
            vek_kolona=True
        ),
        [1.57073796072, 2.99966378552, 3.14142920811]
    )
    
    S_lista_telo = [[0, 0, -1, 2, 0,   0],
                    [0, 0,  0, 0, 1,   0],
                    [0, 0,  1, 0, 0, 0.1]]

    assert np.allclose(
        f(
            M,
            S_lista_telo,
            teta_lista0,
            Tk,
            tol_omega,
            tol_v,
            koord_sistem_prostor=False
        ),
        [1.57073796072, 2.99966378552, 3.14142920811]
    )
    
    assert np.allclose(
        f(
            M,
            S_lista_prostor,
            [1.57073782966, 2.99966384467, 3.14153419986],
            Tk,
            tol_omega,
            tol_v,
            vek_kolona=True
        ),
        [1.57073782966, 2.99966384467, 3.14153419986]
    )
    
    with pytest.raises(kin.InvKinError):
        assert f(
            M,
            S_lista_prostor,
            teta_lista0,
            Tk,
            tol_omega,
            tol_v,
            1,
            vek_kolona=True
        )
        
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f(
            M,
            S_lista_prostor,
            teta_lista0,
            Tk,
            tol_omega,
            tol_v,
            0,
            vek_kolona=True
        )

def test_paden_kahan1() -> None:
    f = kin.paden_kahan1
    

    assert np.allclose(
        f(
            [0, -1, 0, 0, 0, 0],
            [0, 3, 1],
            [-1, 3, 0]
        ),
        1.57079632679
    )
    
    assert np.allclose(
        f(
            [[0],
             [0],
             [3],
             [3],
             [-9],
             [0]],
            [4, 1, 0],
            [[2],
             [1],
             [0]]
        ),
        3.14159265359
    )

    with pytest.raises(kin.PadenKahanError):
        assert f(
            [0, 1, 0, 0, 0, 0],
            [0, 3, 10],
            [-1, 3, 0]
        )

    with pytest.raises(kin.PadenKahanError):
        assert f(
            [0, 1, 0, 0, 0, 0],
            [0, 3, 1],
            [-1, 5, 0]
        )

    # Nepravilna upotreba
    with pytest.raises(ValueError):
        assert f(
            [[0],
            [0],
            [3],
            [9],
            [3],
            [1]],
            [4, 1, 0],
            [[2],
            [1],
            [0]]
        )

def test_paden_kahan2() -> None:
    f = kin.paden_kahan2
    
    # Pravilna upotreba
    assert np.allclose(
        f(
            [3, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0],
            [[0],
            [0],
            [1]],
            [0, -np.sqrt(3)/2, 1/2]
        ),
        ((1.04719755120, 0.0), (2.61799387799, 3.14159265359))
    )

    assert np.allclose(
        f(
            [[0],
            [0],
            [1],
            [0],
            [0],
            [0]],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ),
        (1.57079632679, 0.0)
    )
    
    assert np.allclose(
        f(
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 1],
            [1, -1, 0]
        ),
        (1.57079632679, 0.0)        
    )
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f(
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 1],
            [1, -1, 0]
        )
        
    with pytest.raises(ValueError):
        f(
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, -1, 1],
            [1, -1, 0]
        )
        
    with pytest.raises(ValueError):
        f(
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, -1, 1],
            [1, -1, 0]
        )
    
    with pytest.raises(kin.PadenKahanError):
        f(
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 1],
            [50, -2, 0]
        )

    with pytest.raises(kin.PadenKahanError):
        f(
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 1.5],
            [1, 0, 1]
        )

def test_paden_kahan3() -> None:
    f = kin.paden_kahan3
    
    # Pravilna upotreba
    assert np.allclose(
        f(
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0],
            [0, 2, 0],
            3
        ),
        0
    )
    
    assert np.allclose(
        f(
            [0, 0, 5, 0, 0, 0],
            [[ 1],
            [-2],
            [0]],
            [3, 3, 3],
            np.sqrt(14)        
        ),
        (2.21429743559, 1.57079632679) 
    )

    assert np.allclose(
        f(
            [0, 0, 1, 0, 0, 0],
            [1, -2, 0],
            [-2, 4, 5],
            np.sqrt(30)
        ),
        3.14159265359
    )

    assert np.allclose(
        f(
            [[4],
            [0],
            [0],
            [0],
            [0],
            [0]],
            [0, -1, 0],
            [[0],
            [2],
            [2]],
            (np.sqrt(9 - 4*np.sqrt(2)))        
        ),
        -2.35619449019
    )
    
    assert np.allclose(
        f(
            [0, 1, 0, 0, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            0
        ),
        1.57079632679
    )
    
    # Nepravilna upotreba
    with pytest.raises(ValueError):
        f(
            [0, 0, 5, 0, 0, 0],
            [1, -2, 0],
            [3, 3, 3],
            -0.1
        )
        
    with pytest.raises(ValueError):
        f(
            [0, 0, 5, 0, 0, 5],
            [1, -2, 0],
            [3, 3, 3],
            np.sqrt(14)
        )

    with pytest.raises(kin.PadenKahanError):    
        f(
            [0, 0, 5, 0, 0, 0],
            [[ 1],
            [-2],
            [0]],
            [3, 3, 3],
            1e6        
        )
    
    with pytest.raises(kin.PadenKahanError):    
        f(
            [[4],
            [0],
            [0],
            [0],
            [0],
            [0]],
            [0, -1, 0],
            [[0],
            [2],
            [2]],
            2e6
        )