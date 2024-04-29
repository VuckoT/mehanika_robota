"""Modul za testiranje objekata iz privatnog modula _alati
"""

"""
*** BIBLIOTEKE ***
"""
from typing import Type
import numpy as np
from .. import _alati
import pytest

"""
*** POMOCNE FUNKCIJE ***
"""
def _uspesan_test_poruka(funkcija_naziv: str) -> None:
    print(f"TEST FUNKCIJE mehanika_robota._alati.{funkcija_naziv}: USPESAN")

"""
*** TESTOVI ***
"""
def _mat_provera_test() -> None:
    f = _alati._mat_provera

    # Pravilan rezultat
    assert f(np.eye(3), (3, 3)) is None
    assert f(np.eye(3), ((3, 3), (4, 4))) is None
    assert f(np.eye(4), ((3, 3), (4, 4))) is None
    assert f(np.zeros((3, 4)), [(3, 5), (4, 3), (3, 4)]) is None
    
    # Nepravilan rezultat
    with pytest.raises(AttributeError):
        f(23, [1, 2])

    with pytest.raises(TypeError):
        f([1, 2], 23)

    with pytest.raises(ValueError):
        f(np.zeros((2, 1)), (1, 2))
        
    with pytest.raises(ValueError):
        f(np.zeros((2, 1)), (0, 2))

    with pytest.raises(ValueError):
        f(np.zeros((2, 1)), (-2, -1))
        
    with pytest.raises(ValueError):
        f(np.zeros((2, 1)), (2, 0))
        
    with pytest.raises(ValueError):
        f(np.zeros((2, 1)), (0, 0))
            
    _uspesan_test_poruka("_mat_provera")

def _vek_provera_test() -> None:
    f = _alati._vek_provera
    
    # Pravilan rezultat    
    assert f(np.array([1, 2, 3]), 3) is None
    assert f(np.array([[1], [2], [3]]), 3) is None
    assert f(np.array([1, 2, 3, 4]), (3, 4)) is None
    assert f(np.array([1, 2, 3, 4]), (4, 3)) is None
    assert f(np.array([[1], [2], [3], [4]]), (3, 4)) is None
    assert f(np.array([[1], [2], [3], [4]]), (4, 3)) is None
    assert f(np.array([[1], [2], [3]]), [3, ]) is None
    
    # Nepravilan rezultat    
    with pytest.raises(ValueError):
        f(np.array([1, 2, 3]), 4)
        
    with pytest.raises(ValueError):
        f(np.array([1, 2, 3]), -4)

    with pytest.raises(ValueError):
        f(np.array([1, 2, 3]), 0)
        
    with pytest.raises(ValueError):
        f(np.array([1, 2, 3]), [-4, -2, 2, 0])
    
    with pytest.raises(ValueError):
        f(np.array([[1], [2], [3]]), 4)
    
    with pytest.raises(AssertionError):
        f(np.array([[1], [2], [3]]), "3")
    
    _uspesan_test_poruka("_vek_provera")
    
def _tol_provera_test() -> None:
    f = _alati._tol_provera

    # Pravilan rezultat    
    assert f(23) is None
    assert f(23.3) is None
    
    # Nepravilan rezultat    
    with pytest.raises(ValueError):
        f(0.0)
        
    with pytest.raises(ValueError):
        f(0)
        
    with pytest.raises(ValueError):
        f(-23)
        
    with pytest.raises(ValueError):
        f(-23.3)
            
    _uspesan_test_poruka("_tol_provera")
    
def testiraj_sve() -> None:
    """Testira sve objekte iz modula i obavestava o uspesnosti testa
    """
    _alati._testiraj_sve(globals())