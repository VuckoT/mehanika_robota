"""Privatni modul za odredjene alate koje se koriste u ostalim modulima
biblioteke
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
from numpy.typing import NDArray
from typing import Sequence

"""
*** PRIVATNE FUNKCIJE ZA DEBAGOVANJE***
"""

def _mat_provera(
    mat: NDArray,
    dim: Sequence[np.int64] | Sequence[Sequence[np.int64]] | NDArray[np.int64],
    mat_naziv: str = "mat"
) -> None:
    # Funkcija proverava da li je neka matrica `mat` dimenzije `dim` ili jedna
    # od dimenzija navedenih u `dim` (npr. da li je mat = np.eye(3) dimenzije
    # [(3, 3), (2, 2)] bi bilo True, jer (3, 3) spada u niz `dim`) i prikazuje
    # gresku sa imenom vektora `mat_naziv` ako nije
    
    if hasattr(dim[0], "__iter__"):
        for dimenzija in dim:
            if np.array_equal(mat.shape, dimenzija):
                return
    else:
        if np.array_equal(mat.shape, dim):
            return

    raise ValueError(
        f"\"{mat_naziv}\" nije pravilnih dimenzija {dim}"
    )    

def _vek_provera(
    vek: NDArray,
    dim: np.int64 | Sequence[np.int64],
    vek_naziv: str = "vek"
):
    # Funkcija proverava da li je neki vektor `vek` dimenzije `dim`x1 ili
    # 1x`dim` i prikazuje gresku sa imenom vektora `vek_naziv` ako nije
    if hasattr(dim, "__iter__"):
        assert not any(hasattr(dimenzija, "__iter__") for dimenzija in dim)
        
        # Kako bi primenili funkciju _mat_provera, moramo dim raspakovati.
        # Npr. ako je dim = (1, 2, 3) nama treba
        # dim_raspakovano = [(1, ), (1, 1), (2, ), (2, 1), (3, ), (3, 1)]
        dim_raspakovano = []
        for dimenzija in dim:
            dim_raspakovano.append((dimenzija, ))
            dim_raspakovano.append((dimenzija, 1))
        
        _mat_provera(vek, dim_raspakovano, vek_naziv)
    else:
        _mat_provera(vek, [(dim, ), (dim, 1)], vek_naziv)

def _tol_provera(
    tol: np.int64 | np.float64,
    tol_naziv: str = "tol"
) -> None:
    # Funkcija proverava da li je neka tolerancija veca od 0 i prikazuje gresku
    # sa imenom tolerancije `tol_naziv` ako nije
    
    if tol <= 0.0:
        raise ValueError(f"Tolerancija \"{tol_naziv}\" nije >0.0")