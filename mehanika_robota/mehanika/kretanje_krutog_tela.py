"""
Kretanje krutog tela
====================
Modul za manipulaciju Lijevim grupama SO(3) i SE(3) kao i njihovim algebrama
so(3) i se(3).

Preporucen nacin uvoza modula je
>>> import mehanika_robota.kretanje_krutog_tela as kkt
"""


"""
*** BIBLIOTEKE ***
"""
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Sequence
from ...mehanika_robota import _alati
from collections import namedtuple

"""
*** PRIVATNE FUNKCIJE ***
"""
def _proj_SO3(R: NDArray) -> NDArray[np.float64]:
    # Projekture matricu `R` najblizem elementu SO(3) grupe
    assert _alati._mat_provera(R, (3, 3), 'R') is None

    U, _, V_transponovano = np.linalg.svd(R)

    proj_R = U @ V_transponovano
    if np.linalg.det(proj_R) < 0.0:
    # U slucaju da projekcija ima negativnu determinantu (levi koordinatni
    # sistem) promenicemo znak determinante (pretvoricemo matricu u desni
    # koordinatni sistem)
    # `mat`, te cemo promeniti znak determinante
        proj_R[:, 2] = -proj_R[:, 2]

    return proj_R

def _proj_so3(mat: NDArray) -> NDArray[np.float64]:
    # Projektuje ulaznu matricu `mat` na najblizi element iz so(3) grupe
    assert _alati._mat_provera(mat, (3, 3)) is None

    return np.array([
        [0.0, (mat[0, 1] - mat[1, 0])/2, (mat[0, 2] - mat[2, 0])/2],
        [(mat[1, 0] - mat[0, 1])/2, 0.0, (mat[1, 2] - mat[2, 1])/2],
        [(mat[2, 0] - mat[0, 2])/2, (mat[2, 1] - mat[1, 2])/2, 0.0]
    ])

def _suprotan_znak_provera(
    a: int | float | np.float64,
    b: int | float | np.float64
) -> bool:
    # Proverava da li su `a` i `b` suprotnog znaka ili jednaki 0
    return (
        np.allclose([a, b], [0.0, 0.0])
        or (
            (a < 0.0 and b > 0.0)
            or (a > 0.0 and b < 0.0)
        )
    )

def _proj_lijeva_algebra_provera(mat: NDArray) -> None:
    # Proverava da li je moguce projektovati `mat` u elemenat so(3) grupe
    assert _alati._mat_provera(mat, [(3, 3), (4, 4)]) is None
    
    if not (
        _suprotan_znak_provera(mat[0, 1], mat[1, 0])
        and _suprotan_znak_provera(mat[0, 2], mat[2, 0])
        and _suprotan_znak_provera(mat[1, 2], mat[2, 1])
    ):
        raise ValueError(
            "Nije moguce projektovati \"mat\" u elemenat "
            f"\"{'so(3)' if mat.shape == (3, 3) else 'se(3)'}\" grupe. Elementi "
            "moraju biti mat[0, 1] == -mat[1, 0], mat[0, 2] == -mat[2, 0] i "
            "mat[1, 2] == -mat[2, 1]"
        )

"""
*** API KLASE ***
"""
ParametriOseZavrtnja = namedtuple(
    "ParametriOseZavrtnja",
    ["vek_ose", "omegaS", "korak_zavrtnja"]
)
"""Klasa koja sadrzi parametre ose zavrtnja
S = (omegaS, -[omegaS]*vek_ose + korak_zavrtnja*omegaS) i sadrzi atribute:

vek_ose : NDArray
    Vektor koji dodiruje tacku ose zavrtnja. Odredjuje se na osnovu Mor-Penroz
    pseudoinversa

omegaS : NDArray
    Vektor ose rotacije
    
korak_zavrtnja : np.float64
    Korak ose zavrtnja
"""


"""
*** API FUNKCIJE ***
"""
def vek_norm(
    vek: Sequence | NDArray,
    vratiti_normu: bool = False
) -> NDArray[np.float64] | Tuple[NDArray[np.float64], np.float64]:
    """Normiranje vektora `vek`

    Parametri
    ---------
    vek : Sequence | NDArray
        Ulazni vektor dimenzije 1xn ili nx1
    
    vratiti_normu : bool
        Odredjuje da li da ukljuci vrednost norme u izlaznu vrednost funkcije
    
    Povratna vrednost
    -----------------
    NDArray[np.float64] | Tuple[NDArray[np.float64], np.float64]
        Normiran vektor `vek` istih dimenzija kao ulazni vektor. Ukoliko je
        `vratiti_normu` True, vraca tuple koji sadrzi normiran vektor i njegova
        norma, inace vraca samo normiran vektor
        
    Greske
    ------
    ValueError
        `vek` nije dimenzije 1xn ili nx1
        
    Primeri
    -------
    >>> vek_norm([1, 2, 3])
    np.array([0.267, 0.535, 0.802])
    >>> vek_norm([[1], [2], [3]], True)
    (np.array([[0.267], [0.535], [0.802]]), np.float64(3.742))
    >>> vek_norm([0, 0, 0], True)
    (np.array([0.0, 0.0, 0.0]), np.float64(0))
    """    
    vek = np.array(vek, dtype=float)

    if vek.ndim <= 2:
        if vek.ndim == 2 and vek.shape[1] > 1:
            raise ValueError("Vektor \"vek\" mora biti dimenzije 1xn ili nx1")
        else:
            norma = np.linalg.norm(vek)
            if np.isclose(norma, 0.0):
                if vratiti_normu:
                    return (np.zeros_like(vek), np.float64(0.0))
                else:
                    return np.zeros_like(vek)
            else:
                return (vek/norma, norma) if vratiti_normu else vek/norma
    else:
        raise ValueError("Vektor \"vek\" mora biti dimenzije 1xn ili nx1")

def inv(mat: Sequence | NDArray) -> NDArray[np.float64]:
    """Odredjuje inverznu matricu koja je deo grupe SO(3) ili SE(3)

    Parametri
    ---------
    mat : Sequence | NDArray
        Matrica koja je deo grupe SO(3) ili SE(3)

    Povratna vrednost
    ----------------
    NDArray[np.float64]
        Inverzna matrica od matrice `mat`
        
    Greske
    ------
    ValueError
        `mat` nije dimenzije 3x3 ili 4x4

    Primeri
    -------
    >>> inv([[0, 0, 1],
             [1, 0, 0],
             [0, 1, 0]])
    np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0]])
    >>> inv([[0, 0, 1,   0],
             [1, 0, 0, 1.5],
             [0, 1, 0,   3],
             [0, 0, 0,   1]])   
    np.array([[0.0, 1.0, 0.0, -1.5],
              [0.0, 0.0, 1.0, -3.0],
              [1.0, 0.0, 0.0,  0.0],
              [0.0, 0.0, 0.0,  1.0]])
    """
    mat = np.array(mat, dtype=float)
    
    _alati._mat_provera(mat, ((3, 3), (4, 4)))
    
    if mat.shape == (4, 4):
        return np.vstack([
            np.hstack([
                mat[:3, :3].T,
                -mat[:3, :3].T @ mat[:3, -1:]
            ]),
            [0.0, 0.0, 0.0, 1.0]
        ])
    else:
        return mat.T

def lijeva_algebra_od_vek(vek: Sequence | NDArray) -> NDArray[np.float64]:
    """Pretvara vektor u element so(3) ili se(3) grupe

    Parametri
    --------
    vek : Sequence | NDArray
        Ulazni vektor dimenzije 1x3, 3x1, 1x6 i 6x1

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Ako je `vek` dimenzije 1x3 ili 3x1, funkcija vraca so(3)
        reprezentaciju vektora, inace (ako je dimenzije 1x6 ili 6x1) vraca
        se(3) reprezentaciju vektora
    
    Greske
    ------
    ValueError
        `vek` nije dimenzije 3x1, 1x3, 6x1 ili 1x6

    Primeri
    -------
    >>> lijeva_algebra_od_vek([1, 2, 3]))
    np.array([[ 0.0, -3.0,  2.0],
              [ 3.0,  0.0, -1.0],
              [-2.0,  1.0,  0.0]])
    >>> lijeva_algebra_od_vek([[1.2],
                                 [2],
                                 [3],
                               [4.3],
                                 [5],
                                 [6]])
    np.array([[ 0.0, -3.0,  2.0, 4.3],
              [ 3.0,  0.0, -1.2, 5.0],
              [-2.0,  1.2,  0.0, 6.0],
              [ 0.0,  0.0,  0.0, 0.0]])
    """    
    vek = np.array(vek, dtype=float)

    _alati._vek_provera(vek, (3, 6))
    
    if vek.ndim == 1:
        so3 = np.array([[    0.0, -vek[2],  vek[1]],
                        [ vek[2],     0.0, -vek[0]],
                        [-vek[1],  vek[0],     0.0]])
    else:
        so3 = np.array([[       0.0, -vek[2, 0],  vek[1, 0]],
                        [ vek[2, 0],        0.0, -vek[0, 0]],
                        [-vek[1, 0],  vek[0, 0],        0.0]])

    if len(vek) == 3:
        return so3
    else:
        return np.vstack([
            np.hstack([so3, vek[3:].reshape(3, 1)]),
            [0.0, 0.0, 0.0, 0.0]
        ])

def vek_od_lijeve_algebre(
    mat: Sequence | NDArray,
    vek_kolona: bool = False
) -> NDArray[np.float64]:
    """Pretvara elemant so(3) ili se(3) grupe u korespodentni vektor

    Parametri
    --------
    mat : Sequence | NDArray
        Ulazna matrice dimenzije 3x3 ili 4x4
    vek_kolona : bool, opcionalno
        Odredjuje oblik povratne vrednosti (automatska vrednost je False).
        Vrednost False znaci da ce izlazni vektor biti vektor red
        (dimenzije 1x3 ili 1x6) dok True znaci da ce izlazni vektor biti
        vektor kolona (dimenzije 3x1 ili 6x1)

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Ako je `mat` dimenzije 3x3, tj. deo so(3) grupe, funkcija vraca
        vektorsku reprezentaciju dimenzije 1x3 ili 3x1 (zavisi od parametra
        `vek_kolona`), inace (ako je dimenzije 4x4, tj. deo se(3) grupe)
        vraca vektorsku reprezentaciju dimenzije 1x6 ili 6x1 (opet, zavisi
        od parametra `vek_kolona`)
        
    Greske
    ------
    ValueError
        `mat` nije dimenzije 3x3 ili 4x4

    Primeri
    -------
    >>> vek_od_lijeve_algebre([[ 0, -3,  2],
                               [ 3,  0, -1],
                               [-2,  1,  0]], True)
    np.array([[1.0], 
              [2.0],
              [3.0]])
    >>> vek_od_lijeve_algebre([[ 0,   -3,    2,   4],
                               [ 3,    0, -1.2, 5.4],
                               [-2,  1.2,    0,   6],
                               [ 0,    0,    0,   0]])
    np.array([1.2, 2.0, 3.0, 4.0, 5.4, 6.0])
    """    
    mat = np.array(mat, dtype=float)
    
    _alati._mat_provera(mat, ((3, 3), (4, 4)))

    omega = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    
    if mat.shape == (3, 3):
        if vek_kolona:
            return omega[:, np.newaxis]
        else:
            return omega
    else:
        if vek_kolona:
            return np.vstack([omega[:, np.newaxis], mat[:3, -1:]])
        else:
            return np.hstack([omega, mat[:3, 3]])

def v_prostor_normiranje(
    v_prostor: Sequence | NDArray,
    vratiti_normu: bool = False
) -> NDArray[np.float64] | Tuple[NDArray[np.float64], np.float64]:
    """Vraca normiran vektor prostornih brzina `v_prostor`, tacnije vraca osu
    zavrtnja. Normu/ugao rotacije takodje moze vratiti u zavisnosti od parametra
    `vratiti_normu`
     
    Parametri
    ---------
    v_prostor : Sequence
        Ulazni vektor prostornih brzina dimenzije 1x6 ili 6x1
    
    vratiti_normu : bool
        Odredjuje da li funkcija da vrati iznos norme vektora prostorne brzine,
        odnosno da li funkcija vraca ugao rotacije ose zavrtnja
    
    Povratna vrednost
    ----------------
    NDArray[np.float64] | Tuple[NDArray[np.float64], np.float64]
        Normiran vektor prostornih brzina `v_prosor`/osu zavrtnja. Ukoliko je
        `vratiti_normu` True, vraca i normu `v_prostor`/ugao rotacije
        korespondentnog ugla rotacije 

    Greske
    ------
    ValueError
        `vek` nije dimenzije 6x1 ili 1x6

    Primeri
    -------
    >>> v_prostor_normiranje([1, 2, 3, 4, 5, 6])
    (np.array([0.267, 0.535, 0.802, 1.069, 1.336, 1.604]), 3.742)
    >>> v_prostor_normiranje([[0],
                              [0],
                              [0],
                              [4],
                              [5],
                              [6]])
    (np.array([[0.0],
               [0.0],
               [0.0],
             [0.456],
             [0.570],
             [0.684]), 8.775)
    >>> v_prostor_normiranje([0, 0, 0, 0, 0, 0], True)
    (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.float64(0))
    """
    v_prostor = np.array(v_prostor, dtype=float)

    _alati._vek_provera(v_prostor, 6, "v_prostor")

    omegaS, teta = vek_norm(v_prostor[:3], True)

    # Nacin konkatenacije vektora zavisi od dimenzije omegaS
    konkatenacija = np.hstack if omegaS.ndim == 1 else np.vstack
    
    if np.isclose(teta, 0.0):
        vS, v_norma = vek_norm(v_prostor[3:], True)
        
        if vratiti_normu:
            return (konkatenacija([omegaS, vS]), v_norma)
        else:
            return konkatenacija([omegaS, vS])
    else:
        if vratiti_normu:
            return (konkatenacija([omegaS, v_prostor[3:]/teta]), teta)
        else:
            return konkatenacija([omegaS, v_prostor[3:]/teta])

def exp(mat: Sequence | NDArray) -> NDArray[np.float64]:
    """Proracunava matricu iz grupe SO(3) ili SE(3) na osnovu Rodrigezove
    formule i Cesls-Moci teoreme respektivno

    Parametri
    ---------
    mat : Sequence | NDArray
        Matrica iz grupe so(3) ili se(3)

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Matricni eksponent na osnovu `mat`- Rodrigezova formula u slucaju da je
        `mat` iz grupe so(3), a Cesls-Moci teorema ako je `mat` iz grupe se(3)

    Greske
    ------
    ValueError
        `mat` nije dimenzije 3x3 ili 4x4

    Primeri
    -------
    >>> exp([[ 0, -3,  2],
             [ 3,  0, -1],
             [-2,  1,  0]])
    np.array([[-0.695,  0.714,  0.089],
              [-0.192, -0.304,  0.933],
              [ 0.693,  0.631 , 0.348]])
    >>> exp([[ 0, -3,  2, 4],
             [ 3,  0, -1, 5],
             [-2,  1,  0, 6],
             [ 0,  0,  0, 0]])
    np.array([[-0.695,  0.714,  0.089,  1.636],
              [-0.192, -0.304,  0.933,  5.289],
              [ 0.693,  0.631,  0.348,  6.595],
              [   0.0,    0.0,    0.0,    1.0]])
    """
    mat = np.array(mat, dtype=float)
    
    _alati._mat_provera(mat, ((3, 3), (4, 4)))
    
    teta = np.linalg.norm(vek_od_lijeve_algebre(mat[:3, :3]))
    
    if np.isclose(teta, 0.0):
        if mat.shape == (3, 3):
            return np.eye(3)
        else:
            return np.vstack([
                np.hstack([np.eye(3),  mat[:3, -1:]]),
                [0.0, 0.0, 0.0, 1.0]
            ])
    else:
        omegaS_mat = mat[:3, :3]/teta
        
        if mat.shape == (3, 3):
            return (
                np.eye(3)
                + np.sin(teta)*omegaS_mat
                + (1 - np.cos(teta))*(omegaS_mat @ omegaS_mat)
            )
        else:
            
            return np.vstack([
                np.hstack([
                    (
                        np.eye(3)
                        + np.sin(teta)*omegaS_mat
                        + (1 - np.cos(teta))*(omegaS_mat @ omegaS_mat)
                    ),
                    np.dot(
                        teta*np.eye(3)
                        + (1 - np.cos(teta))*omegaS_mat
                        + (teta - np.sin(teta))*(omegaS_mat @ omegaS_mat),
                        mat[:3, -1:]/teta
                    )
                ]),
                [0.0, 0.0, 0.0, 1.0]
            ])

def log(mat: Sequence | NDArray) -> NDArray[np.float64]:
    """Odredjuje logaritam matrice `mat` koja je deo grupe SO(3) ili SE(3)

    Parametri
    ---------
    mat : Sequence | NDArray
        Ulazna matrica koja je deo grupe SO(3) ili SE(3)

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Logaritam ulazne matrice koja, u zavisnosti od `mat`, moze biti deo
        grupe omegaS_mat*teta iz so(3) ili S_mat*teta iz se(3) gde je
        omegaS_mat-so(3) reprezentacija jedinicnog vektora rotacije,
        teta-ugao rotacije i S-se(3) reprezentacija ose zavrtnja
    
    Greske
    ------
    ValueError
        `mat` nije dimenzije 3x3 ili 4x4
    
    Primeri
    -------
    >>> log([[0, 0, 1],
             [1, 0, 0],
             [0, 1, 0]])
    np.array([[   0.0, -1.209,  1.209],
              [ 1.209,    0.0, -1.209],
              [-1.209,  1.209,    0.0]])
    >>> log([[1, 0,  0, 0],
             [0, 0, -1, 0],
             [0, 1,  0, 3],
             [0, 0,  0, 1]])
    np.array([[0.0,   0.0,    0.0,    0.0]
              [0.0,   0.0, -1.571,  2.356]
              [0.0, 1.571,    0.0,  2.356]
              [0.0,   0.0,    0.0,   0.0]])

    """
    mat = np.array(mat, dtype=float)
    
    _alati._mat_provera(mat, ((3, 3), (4, 4)))
    
    R = mat[:3, :3]
    
    # Zbog greske pri numerickom proracunu, mozemo dobiti da je cos(teta) malo
    # veci od 1 ili malo manji od -1 i ako to uvrstimo u
    # teta = arccos(cos(teta)) dobijamo np.nan, zato mora na malo drugaciji
    # nacin da se pristupi proracunu ugla teta
    cos_teta = (np.trace(R) - 1)/2
    
    if np.isclose(cos_teta, 1.0):
        teta = 0.0
    elif np.isclose(cos_teta, -1.0):
        teta = np.pi
    else:
        teta = np.arccos(cos_teta)

    if np.isclose(teta, 0.0):
        omegaS_mat = np.eye(3)
        teta = np.float64(0.0)
    elif np.isclose(np.trace(R), -1.0):
        if not np.isclose(1.0 + R[2, 2], 0.0):
            omegaS = (
                (1.0/np.sqrt(2*(1 + R[2, 2])))
                * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            )
        elif not np.isclose(1.0 + R[1, 1], 0.0):
            omegaS = (
                (1.0/np.sqrt(2*(1 + R[1, 1])))
                * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            )
        else:
            omegaS = (
                (1.0/np.sqrt(2*(1 + R[0, 0])))
                * np.array([1 + R[0][0], R[1][0], R[2][0]])
            )
        omegaS_mat = lijeva_algebra_od_vek(omegaS)
        teta = np.pi
    else:
        omegaS_mat = 1/(2 * np.sin(teta))*(R - R.T)
    
    if mat.shape == (3, 3):
        return teta*omegaS_mat
    else:
        if np.isclose(teta, 0.0):
            return np.vstack([
                np.hstack([np.eye(3), mat[:3, -1:]]),
                [0.0, 0.0, 0.0, 0.0]
            ])
        else:
            return np.vstack([
                np.hstack([
                    omegaS_mat,
                    np.dot(
                        np.eye(3)/teta - omegaS_mat/2
                        + (1/teta - 1/(2*np.tan(teta/2)))
                        *(omegaS_mat @ omegaS_mat),
                        mat[:3, -1:]
                    )
                ]),
                [0.0, 0.0, 0.0, 0.0]
            ])*teta

def SE3_rastavi(
    mat: Sequence | NDArray,
    vek_kolona: bool = False
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pretvara homogenu transformacionu matricu `mat` iz grupe SE(3) u tuple
    koji je sacinjen od rotacione (iz grupe SO(3)) i translacione komponente
    (vektor dimenzije 1x3 ili 3x1) od matrice `mat`

    Paramteri
    ---------
    mat : Sequence | NDArray
        Ulazna homogena transformaciona matrica dimenzije 4x4
    vek_kolona : bool, opcionalno
        Odredjuje oblik povratne vrednosti (automatska vrednost je False).
        Vrednost False znaci da ce izlazni vektor biti vektor red
        (dimenzije 1x3) dok True znaci da ce izlazni vektor biti
        vektor kolona (dimenzije 3x1)

    Povratna vrednost
    -----------------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Vraca (R, p) gde je R rotaciona matrica iz grupe SO(3) i p vektor iz
        prostora R^n, izvucene iz matrice `mat`
    
    Greske
    ------
    ValueError
        `mat` nije dimenzije 4x4
    
    Primeri
    -------
    >>> SE3_rastavi([[1, 0,  0, 0],
                     [0, 0, -1, 0],
                     [0, 1,  0, 3],
                     [0, 0,  0, 1]])
    (
        np.array([[1.0, 0.0,  0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0,  0.0]]),
        np.array([0.0, 0.0, 3.0])
    )
    >>> SE3_rastavi([[1, 0,  0, 3],
                     [0, 1,  0, 1],
                     [0, 0,  1, 0],
                     [0, 0,  0, 1]], True)
    (
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]]),
        np.array([[3.0],
                  [1.0],
                  [0.0]])
    )
    """
    mat = np.array(mat, dtype=float)
    
    _alati._mat_provera(mat, (4, 4), 'mat')
    
    return (
        (mat[:3, :3], mat[:3, -1:]) if vek_kolona else
        (mat[:3, :3], mat[:3, 3])
    )

def SE3_sastavi(
    mat: Sequence | NDArray, vek: Sequence | NDArray
) -> NDArray[np.float64]:
    """Pretvara rotacionu matricu `mat` iz grupe SO(3) i pozicioni vektor `vek`
    u homogenu transformacionu matricu iz grupe SE(3)`

    Paramteri
    ---------
    mat : Sequence | NDArray
        Ulazna rotaciona matrica dimenzije 3x3
        
    vek : Sequence | NDArray
        Ulazni pozicioni vektor dimenzije 1x3 ili 3x1

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Vraca homogenu transformacionu matricu iz grupe SE(3) dimenzije 4x4
        u skladu sa parametrima `mat` i `vek`
    
    Greske
    ------
    ValueError
        `mat` nije dimenzije 3x3. `vek` nije dimenzije 1x3 ili 3x1
    
    Primeri
    -------
    >>> SE3_sastavi(
        [[1, 0,  0],
         [0, 0, -1],
         [0, 1,  0]],
        [1, 2, 3]
    )
    np.array([[1.0, 0.0,  0.0, 1.0],
              [0.0, 0.0, -1.0, 2.0],
              [0.0, 1.0,  0.0, 3.0],
              [0.0, 0.0,  0.0, 1.0]])
    """
    mat = np.array(mat, dtype=float)
    vek = np.array(vek, dtype=float)
    
    _alati._mat_provera(mat, (3, 3))
    _alati._vek_provera(vek, 3)
    
    return np.vstack([
        np.hstack([mat, vek[:, np.newaxis] if vek.ndim == 1 else vek]),
        [0.0, 0.0, 0.0, 1.0]
    ])

def Ad(T: Sequence | NDArray) -> NDArray[np.float64]:
    """Adjungovana reprezentacija matrice `T` iz grupe SE(3)

    Parametri
    ---------
    T : Sequence | NDArray
        Ulazna homogena transformaciona matrica iz grupe SE(3) dimenzije 4x4

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Adjungovana reprezentacija matrice `T` cije su dimenzije 6x6
    
    Greske
    ------
    ValueError
        `T` nije dimenzije 4x4
    
    Primeri
    -------
    >>> Ad([[1, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 1,  0, 3],
            [0, 0,  0, 1]])
    np.array([[1.0, 0.0,  0.0, 0.0, 0.0,  0.0],
              [0.0, 0.0, -1.0, 0.0, 0.0,  0.0],
              [0.0, 1.0,  0.0, 0.0, 0.0,  0.0],
              [0.0, 0.0,  3.0, 1.0, 0.0,  0.0],
              [3.0, 0.0,  0.0, 0.0, 0.0, -1.0],
              [0.0, 0.0,  0.0, 0.0, 1.0,  0.0]])
    """
    T = np.array(T, dtype=float)
    
    _alati._mat_provera(T, (4, 4), 'T')
        
    R, p = SE3_rastavi(T)
    return np.vstack([
        np.hstack([R, np.zeros((3, 3))]),
        np.hstack([lijeva_algebra_od_vek(p) @ R, R])
    ])


def osa_zavrtnja_parametarski(
    vek_ose: Sequence | NDArray,
    omegaS: Sequence | NDArray,
    korak_zavrtnja: float | np.float64,
    vek_kolona: bool = False
) -> NDArray[np.float64]:
    """Pretvara parametre `vek_ose`, `omegaS` i `korak_zavrtnja` u vektor ose
    zavrtnja. Za slucaj kada je osa zavrtnja cisto linearno kretanje preporuceno
    je koristiti funkciju `mr.kkt.osa_zavrtnja_lin_v()`. Razlog tome je da ce
    ova funkcija vratiti tacno resenje samo ukoliko vazi da je
    `np.isinf(korak_zavrtnja)`
    
    Parametri
    ---------
    vek_ose : Sequence | NDArray
        Vektor pozicije ose zavrtnja dimenzije 1x3 ili 3x1
    omegaS : Sequence | NDArray
        Vektor orijentacije ose zavrtnja (vektor rotacije oko ose) dimenzije
        1x3 ili 3x1. Ne mora biti normalizovano
    korak_zavrtnja : int | float | np.float64
        Korak zavrtnja
    vek_kolona : bool, opcionalno
        Odredjuje izlazni oblik ose zavrtnja (automatska vrednost je False).
        Ukoliko je `vek_kolona` True onda vraca osu zavrtnja u obliku vektora
        kolona dimenzije 6x1 inace vraca vektor red (kada je `vek_kolona`
        False) cije su dimenzije 1x6
        
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Osa zavrtnja u obliku vektora dimenzije 1x6-kada je `vek_kolona` False
        ili isti vektor u obliku vektora kolone dimenzije 6x1-kada je
        `vek_kolona` True
    
    Greske
    ------
    ValueError
        `vek_ose` ili `omegaS` nije dimenzije 1x3 ili 3x1. `korak_zavrtnja` nije
        >=0
    
    Primeri
    -------
    >>> osa_zavrtnja_parametarski(
        [[3],
         [0],
         [0]],
        [0, 0, 1],
        2,
        True
    )
    np.array([[0.0],
              [0.0],
              [1.0],
              [0.0],
             [-3.0],
              [2.0]])
    >>> osa_zavrtnja_parametarski(
        [3, 0, 0],
        [[0],
         [0],
         [3]],
        0
    )
    np.array([0.0, 0.0, 1.0, 0.0, -3.0, 0.0])
    """
    vek_ose = np.array(vek_ose, dtype=float)
    omegaS = np.array(omegaS, dtype=float)
    
    _alati._vek_provera(vek_ose, 3, "vek_ose")        
    _alati._vek_provera(omegaS, 3, "omegaS")
    
    # U slucaju da je korak blizu nule, zelimo da bude tacno nule zbog
    # komaparacije koja sledi kasnije
    if np.isclose(korak_zavrtnja, 0.0):
        korak_zavrtnja = 0
    
    if korak_zavrtnja < 0:
        raise ValueError("Korak ose zavrtnja \"korak_zavrtnja\" mora biti >0")
            
    if np.isinf(korak_zavrtnja):
        return osa_zavrtnja_lin_v(
            omegaS.reshape((3, 1)) if vek_kolona else omegaS.reshape(3)
        )
            
    # Potrebni su vektori kolone za proracun
    if omegaS.ndim == 2:
        omegaS = omegaS.reshape(3)
        
    if vek_ose.ndim == 2:
        vek_ose = vek_ose.reshape(3)
    
    omegaS = vek_norm(omegaS)

    if vek_kolona:
        return np.vstack([
            omegaS[:, np.newaxis],
            (np.cross(vek_ose, omegaS) + korak_zavrtnja*omegaS)[:, np.newaxis]
        ])
    else:
        return np.hstack(
            [omegaS, np.cross(vek_ose, omegaS) + korak_zavrtnja*omegaS]
        )

def parametri_ose_zavrtnja(
    osa_zavrtnja: Sequence | NDArray
) -> ParametriOseZavrtnja:
    """Odredjuje parametre ose zavrtnja vek_ose, omegaS, korak_zavrtnja
    
    Parametri
    ---------
    osa_zavrtnja : Sequence | NDArray
        Vektor ose zavrtnja dimenzije 1x6 ili 6x1. Ne mora biti normalizovano
        
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        namedtuple klasa koja sadrzi parametre ose zavrtnja i sadrzi atribute:

    vek_ose : NDArray[np.float64]
        Vektor pozicije ose zavrtnja. Odredjuje se na osnovu Mor-Penroz
        pseudoinversa. U slucaju da je osa zavrtnja cista translacija (`korak_zavrtnja ==
        np.inf`), onda se za vektor pozicije ose zavrtnja uzima nula vector. Cak
        i da osa rotacija ne prolazi kroz koordinatni pocetak, nije moguce
        odrediti poziciju ose zavrtnja.

        Vektor koji dodiruje tacku ose zavrtnja. 

    omegaS : NDArray[np.float64]
        Vektor ose rotacije
        
    korak_zavrtnja : np.float64
        Korak ose zavrtnja
    
    Greske
    ------
    ValueError
        "osa_zavrtnja"  nije dimenzije 1x6 ili 6x1
    
    Primeri
    -------
    >>> parametri_ose_zavrtnja([1, 0, 0, 0, 0, 1])
    ParametriOseZavrtnja(
        vek_ose=np.array([0, -1, 0]),
        omegaS=np.array([1, 0, 0]),
        korak_zavrtnja=np.float64(0)
    )
    >>> parametri_ose_zavrtnja([[1],
                                [0],
                                [0],
                                [1],
                                [0],
                                [1])
    ParametriOseZavrtnja(
        vek_ose=np.array([[0], [-1], [0]]),
        omegaS=np.array([[1], [0], [0]]),
        korak_zavrtnja=np.float64(1)
    )
    >>> parametri_ose_zavrtnja([3, 4, 12, 0, 0, 24])
    ParametriOseZavrtnja(
        vek_ose=np.array([0.61538461538, -0.46153846154, 0.0]),
        omegaS=np.array([0.23076923077, 0.30769230769, 0.92307692308]),
        korak_zavrtnja=np.float64(0)
    )
    """
    osa_zavrtnja = np.array(osa_zavrtnja, dtype=float)
    _alati._vek_provera(osa_zavrtnja, 6, "osa_zavrtnja")
    
    # Normirati vektor ose zavrtnja i pretvoriti ga u vektor reda
    osa_zavrtnja = v_prostor_normiranje(osa_zavrtnja)
    
    if osa_zavrtnja.ndim == 1:
        if np.allclose(osa_zavrtnja[:3], np.zeros(3)):
            return ParametriOseZavrtnja(np.zeros(3), osa_zavrtnja[3:], np.inf)
        else:
            return ParametriOseZavrtnja(
                np.linalg.lstsq(
                    -lijeva_algebra_od_vek(osa_zavrtnja[:3]),
                    osa_zavrtnja[3:],
                    None
                )[0],
                osa_zavrtnja[:3],
                np.dot(osa_zavrtnja[:3], osa_zavrtnja[3:])
            )
    else:
        if np.allclose(osa_zavrtnja[:3], np.zeros(3)):
            return ParametriOseZavrtnja(
                np.zeros((3, 1)),
                osa_zavrtnja[3:],
                np.inf
            )
        else:
            return ParametriOseZavrtnja(
                np.linalg.lstsq(
                    -lijeva_algebra_od_vek(osa_zavrtnja[:3]),
                    osa_zavrtnja[3:],
                    None
                )[0],
                osa_zavrtnja[:3],
                np.dot(
                    osa_zavrtnja[:3].reshape(3),
                    osa_zavrtnja[3:].reshape(3)
                )
            )

def osa_zavrtnja_lin_v(v: Sequence | NDArray) -> NDArray[np.float64]:
    """Pretvara linearnu brzinu `v` u vektor ose zavrtnja.
    
    Parametri
    ---------
    v : Sequence | NDArray
        Vektor brzine dimenzije 1x3 ili 3x1. Ne mora biti normalizovano
        
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Osa zavrtnja u obliku vektora reda 1x6 ili vektora kolone 6x1 u 
        zavisnosti od toga da li je `v` vektor red ili kolona.
    
    Greske
    ------
    ValueError
        `v`  nije dimenzije 1x3 ili 3x1
    
    Primeri
    -------
    >>> osa_zavrtnja_lin_v([[3],
                            [0],
                            [0]])
    np.array([[0.0],
              [0.0],
              [0.0],
              [1.0],
              [0.0],
              [0.0]])
    >>> osa_zavrtnja_parametarski([0, 2, 0])
    np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    """
    v = np.array(v, dtype=float)
    
    _alati._vek_provera(v, 3, 'v')
    
    if v.ndim == 1:
        return np.hstack([[0.0, 0.0, 0.0], vek_norm(v)])
    else:
        return np.vstack([[[0.0], [0.0], [0.0]], vek_norm(v)])

def proj_grupa(
    mat: Sequence | NDArray, grupa: Literal["SO3", "SE3", "so3", "se3"]
) -> NDArray[np.float64]:
    """Projektuje matricu najblizem elementu date grupe. Nacin projekcije je
    opisan u odeljku Beleske

    Parametri
    ---------
    mat : Sequence | NDArray
        Ulazna matrica za pojektovanje. Ukoliko je projekcija na grupu so(3) ili
        se(3), `mat `elementi `mat[0, 1]` i `mat[1, 0]`,
        `mat[0, 2]` i `mat[2, 0]`, `mat[1, 2]` i `mat[2, 1]`, moraju biti
        razlicitog znaka. Napomenuti parovi elemenata mogu biti jednaki samo
        u slucaju da su priblizno 0
    grupa : Literal["SO3", "SE3", "so3", "se3"]
        Redom grupe SO(3), SE(3), so(3), se(3) na koje treba projektovati
        matricu `mat`, nacin projekcije je opisan u odeljku Beleske
        
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Projekcija matrice `mat` datoj grupi pomocu `grupa`

    Beleske
    -------
    SO(3) projekcija
        Projekcija se vrsi tako sto se matrica `mat` dekompozicija matrice na
        njene singularne vrednosti i postavljaju se vrednosti dijagonalne matrice
        da budu jednake 1.0, sto
         
         `U, S, V_transponovano = np.linalg.svd(mat)
         S = np.eye(3, dtype=float)
         proj_mat = U @ S @ V_transponovano = U @ V_transponovano`
        
        Vise detalja o singularnoj dekompoziciji je dato u Prilogu E od knjige
        "Modern Robotic: Mechanics, Planning and Control", autori Kevin M. Lynch
        i Frank C. Park. Takodje je dostupno na
        https://hades.mech.northwestern.edu/index.php/Modern_Robotics
            
    SE(3) projekcija
        Rotacioni deo matrice se odredjuje prema prethodno navedenom postupku o
        SO(3) projekciji, translacioni deo se prepisuje, a poslednji red se
        postavlja na `[0, 0, 0, 1]`
    
    so(3) projekcija
        Elementi na glavnoj dijagonali se postavljaju na 0, dok elementi
        iznad i ispod glavne dijagonale uzimaju srednju aritmeticku vrednost.
        Ukoliko znak elemenata iznad i ispod glavne dijagonale nije obrnut,
        proracun nece rezultovati dobroj projekciji
    se(3) projekcija
        Deo matrice `mat[:3, :3]` se odredjuje na osnovu prethodno navedenom
        postupku o so(3) projekciji, brzine (`mat[:3, -1:]`) se prepisuju, a
        poslednji red se postavlja na `[0, 0, 0, 1]`
    
    Greske
    ------
    ValueError
        Dimenzije `mat` i navedena grupa `grupa` se ne poklapaju. `grupa` nije
        \"SO3\", \"SE3\", \"so3\" ili \"se3\". Nemoguca projekcija na grupu
        so(3) ili se(3) zato sto `mat` elementi `mat[0, 1]` i `mat[1, 0]`,
        `mat[0, 2]` i `mat[2, 0]`, `mat[1, 2]` i `mat[2, 1]`, nisu razlicitog
        znaka ili priblizno jednaki 0
    
    Primeri
    -------
    >>> proj_grupa([[   0, -0.982,     0],
                    [1.01,      0,     0],
                    [0.02,      0, 0.991]], "SO3")
    np.array([[0.0, -1.0,   0.0],
              [1.0,  0.0, -0.01],
              [0.01, 0.0,   1.0]])
    >>> proj_grupa([[0.03, -0.982,  0.4],
                    [0.99,   0.02,  1.1],
                    [-0.32, -1.02,    0]]), "so3")
    np.array([[  0.0, -0.986, 0.36],
              [0.986,    0.0, 1.06],
              [-0.36,  -1.06,  0.0]])
    >>> proj_grupa([[1.03, -0.982,      0,  1.2],
                    [0.01,   0.02,      0, 3.42],
                    [   0,      0, -0.971,    2],
                    [0.23,   0.01,      0, 1.01]], "SE3")
    np.array([[0.727, -0.687, 0.0,  1.2],
              [0.687,  0.727, 0.0, 3.42],
              [  0.0,    0.0, 0.0,  2.0],
              [  0.0,    0.0, 0.0,  1.0]])
    >>> proj_grupa([[ 0.03, -1.03,  0.5, 2.3],
                    [    1, 0.001,  2.1, 1.2],
                    [-0.44, -2.02, 0.02,   0],
                    [-0.02,  0.02,    0,   0]], "se3")
    np.array([[  0.0, -1.015, 0.47,  2.3],
              [1.015,    0.0, 2.06,  1.2],
              [-0.47,  -2.06,  0.0,  0.0],
              [  0.0,    0.0,  0.0,  0.0]])
    """
    mat = np.array(mat, dtype=float)
    
    if grupa == "SO3" or grupa == "so3":
        _alati._mat_provera(mat, (3, 3))
    else:
        _alati._mat_provera(mat, (4, 4))

    match grupa:
        case "SO3":
            return _proj_SO3(mat)
        case "SE3":
            return np.vstack([
                np.hstack([
                    _proj_SO3(mat[:3, :3]),
                    mat[:3, -1:]
                ]),
                [0.0, 0.0, 0.0, 1.0]
            ])
        case "so3":
            _proj_lijeva_algebra_provera(mat)
            return _proj_so3(mat)
        case "se3":
            _proj_lijeva_algebra_provera(mat)
            return np.vstack([
                np.hstack([
                    _proj_so3(mat[:3, :3]),
                    mat[:3, -1:]
                ]),
                [0.0, 0.0, 0.0, 0.0]
            ])
        case _:
            raise ValueError(
                "Nepoznata grupa \"grupa\", unesite \"SO3\", \"SE3\", "
                "\"so3\" ili \"se3\""
            )

def Rot(
    osa_rotacije: Literal['x', 'y', 'z'],
    ugao: int | float | np.int32 | np.float64,
    grupa: Literal["SO3", "SE3"] = "SE3"
) -> NDArray[np.float64]:
    """Matrica iz grupe SO(3) ili SE(3) u slucaju da se osa rotacije/osa
    zavrtnja (gde je korak zavrtnja jednak nuli) poklapa sa osama x, y ili z.
    Funkcija je preporucena kao zamena za `exp()` jer je brzi proracun

    Parametri
    ---------
    osa_rotacije : Literal['x', 'y', 'z']
        Osa rotacije u slucaju da je izlaz matrica iz grupe SO(3) ili pravac ose
        zavrtnja ciji je korak jednak nuli
    ugao : int | float | np.int32 | np.float64
        Ugao rotacije oko ose rotacije ili oko ose zavrtnja
    grupa : Literal[3, 4], opcionalno
        Odredjuje da li je izlaz funkcije iz grupe SO(3) kada je
        `grupa == "SO3` ili iz grupe SE(3) `grupa == "SE3"` (automatska vrednost
        je "SE3")

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Matrica iz grupe SO(3) ili SE(3) u zavisnosti od parametra `grupa`

    Greske
    ------
    ValueError
        `osa_rotacije` nije "x", "y" ili "z". `grupa` nije "SO3" ili "SE3".
        `ugao` nije pravilnog tipa nego je niz tipa Sequence ili NDArray
    
    Primeri
    -------
    >>> Rot('x', np.pi/4, "SO3")
    np.array([[1.0,   0.0,    0.0],
              [0.0, 0.707, -0.707],
              [0.0, 0.707,  0.707]])

    >>> Rot('y', np.pi/2)
    np.array([[ 0.0, 0.0, 1.0, 0.0],
              [ 0.0, 1.0, 0.0, 0.0],
              [-1.0, 0.0, 0.0, 0.0],
              [ 0.0, 0.0, 0.0, 1.0]])
    """
    if hasattr(ugao, "__len__"):
        raise ValueError("\"ugao\" ne sme biti niz tipa Sequence ili NDArray")
    
    match osa_rotacije:
        case 'x':
            Rot = np.array([[1.0,          0.0,           0.0],
                            [0.0, np.cos(ugao), -np.sin(ugao)],
                            [0.0, np.sin(ugao),  np.cos(ugao)]])
        case 'y':
            Rot = np.array([[ np.cos(ugao), 0.0, np.sin(ugao)],
                            [          0.0, 1.0,          0.0],
                            [-np.sin(ugao), 0.0, np.cos(ugao)]])
        case 'z':
            Rot = np.array([[np.cos(ugao), -np.sin(ugao), 0.0],
                            [np.sin(ugao),  np.cos(ugao), 0.0],
                            [         0.0,           0.0, 1.0]])
        case _:
            raise ValueError(
                "\"osa_rotacije\" mora biti \"x\", \"y\" ili \"z\""
            )

    if grupa == "SE3":
        return np.vstack([
            np.hstack([Rot, np.zeros((3, 1))]),
            [0.0, 0.0, 0.0, 1.0]
        ])
    elif grupa == "SO3":
        return Rot
    else:
        raise ValueError("\"grupa\" mora biti \"SO3\" ili \"SE3\"")

def Trans(vek: Sequence | NDArray) -> NDArray[np.float64]:
    """Matrica translacije iz grupe SE(3)

    Parametri
    ---------
    vek : Sequence | NDArray
        Vektor pozicije/translacije dimenzije 1x3 ili 3x1

    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Matrica ciste translacije iz grupe SE(3)
        
    Greska
    ------
    ValueError
        `vek` nije dimenzije 1x3 ili 3x1
    
    Primeri
    -------
    >>> Trans([1, 2, 3])
    np.array([[1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 2.0],
              [0.0, 0.0, 1.0, 3.0],
              [0.0, 0.0, 0.0, 1.0]])
    """
    vek = np.array(vek, dtype=float)

    _alati._vek_provera(vek, 3)
        
    return np.vstack([
        np.hstack([np.eye(3), vek.reshape((3, 1))]),
        [0.0, 0.0, 0.0, 1.0]
    ])

def exp_vek(vek: Sequence | NDArray) -> NDArray[np.float64]:
    """Skraceni zapis od `exp(lijeva_algebra_od_vek(vek))`"""
    return exp(lijeva_algebra_od_vek(vek))

def exp_vek_ugao(
    vek: Sequence | NDArray,
    ugao: int | float | np.int32 | np.float64
) -> NDArray[np.float64]:
    """Skraceni zapis od
    `exp(lijeva_algebra_od_vek(v_prostor_normiranje(vek))*ugao)`
    """
    return exp(lijeva_algebra_od_vek(v_prostor_normiranje(vek))*ugao)

def homogeni_vek(vek: Sequence | NDArray) -> NDArray[np.float64]:
    """Homogena reprezentacija vektora
    
    Parametri
    ---------
    vek : Sequence | NDArray
        Ulazni vektor dimenzije 1x3 ili 3x1
    
    Povratna vrednost
    -----------------
        Homogena reprezentacija vektora `vek`. Vektor je dimenzije 1x4 ili 4x1
        kada je ulaznim vektor dimenzije 1x3 ili 3x1 respektivno.

    Greske
    ------
    ValueError
        `vek` nije dimenzije 1x3 ili 3x1    
    
    Primeri
    -------
    >>> homogeni_vek([1, 2, 3])
    np.array([1.0, 2.0, 3.0, 1.0])
    >>> homogeni_vek([[1],
                      [2],
                      [3]])
    np.array([[1.0],
              [2.0],
              [3.0],
              [1.0]])
    """
    vek = np.array(vek, dtype=float)
    _alati._vek_provera(vek, 3)
    
    if vek.ndim == 1:
        return np.hstack([vek, 1.0])
    else:
        return np.vstack([vek, 1.0])
        
def SE3_proizvod_3D(
    mat: Sequence | NDArray,
    vek: Sequence | NDArray
) -> NDArray[np.float64]:
    """Za homogenu transformacionu matricu `mat = (R, p)` funkcija vraca
    `R@vek + p`

    Parametri
    ---------
    mat : Sequence | NDArray
        Matrica iz grupe SE(3) dimenzije 4x4
    vek : Sequence | NDArray
        Vektor dimenzije 3x1 ili 1x3
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Za matricu `mat = (R, p)` vraca proizvod `R@vek + p` gde dimenzije
        izlaznog vektora je isti kao i dimenzije `vek` 
    
    Greske
    ------
    ValueError
        "vek" ili "mat" nisu pravilnih dimenzija
    Primeri
    -------
    >>> SE3_proizvod_3D(
        [[0, -1, 0, 1],
         [1,  0, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]],
        [1, 2, 3]
    )
    np.array([-1.0, 1.0, 3.0])
    >>> SE3_proizvod_3D(
        [[0, -1, 0, 1],
         [1,  0, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]],
        [[1], [2], [3]]
    )
    np.array([[-1.0],
              [ 1.0], [
              [ 3.0]])
    """
    mat = np.array(mat, dtype=float)
    vek = np.array(vek, dtype=float)
    
    _alati._mat_provera(mat, (4, 4))
    _alati._vek_provera(vek, 3)
    
    return (mat@homogeni_vek(vek))[:3]