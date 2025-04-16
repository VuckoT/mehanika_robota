"""
Mehanika Niryo One manipulatora
===============================

Specijalizovan modul za efikasnije proracune mehanike Niryo One manipulatora.
Takodje sadrzi konstante vezane za mehanicke karakteristike Niryo One
manipulatora

Preporucen nacin uvoza modula je
>>> import mehanika_robota.roboti.niryo_one as n_one
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import namedtuple
from types import MappingProxyType
from typing import ClassVar, Sequence, Dict, Literal, Tuple, Union
from mehanika_robota import _alati
from mehanika_robota.mehanika import kinematika as kin
from mehanika_robota.mehanika.kinematika import InvKinError
from mehanika_robota.mehanika import mat_prostor as mp
from itertools import product

"""
*** API KLASE ***
"""
@dataclass(frozen=True)
class NiryoOne:
    """Objekat koji sadrzi bitne karakteristike Niryo One robota. Individualni
    objekti u sklopu `NiryoOne` klase su namenjeni da ostanu nepromenjeni 
    """
    
    L: ClassVar[NDArray] = np.array(
        [103e-3, 80e-3, 210e-3, 30e-3, 41.5e-3, 180e-3, 23.7e-3, 5.5e-3],
        dtype=float
    )
    """ Vektor red karakteristicnih duzina strukture robota
    """
    
    M: ClassVar[NDArray] = np.array([
        [1.0, 0.0, 0.0, 245.2e-3],
        [0.0, 1.0, 0.0,      0.0],
        [0.0, 0.0, 1.0, 417.5e-3],
        [0.0, 0.0, 0.0,      1.0]
    ], dtype=float)
    """Pocetna konfiguracija robota. Matrica iz grupe SE(3) koja sadrzi
    orijentaciju i poziciju hvataca robota kada su svi zglobovi
    u nultoj/pocetnoj poziciji
    """
    
    S_PROSTOR: ClassVar[NDArray] = np.array([
        [0.0,  0.0, 1.0,    0.0,      0.0,       0.0],
        [0.0, -1.0, 0.0, 183e-3,      0.0,       0.0],
        [0.0, -1.0, 0.0, 393e-3,      0.0,       0.0],
        [1.0,  0.0, 0.0,    0.0,   423e-3,       0.0],
        [0.0, -1.0, 0.0, 423e-3,      0.0, -221.5e-3],
        [1.0,  0.0, 0.0,    0.0, 417.5e-3,       0.0]
    ], dtype=float)
    """Matrica ciji su redovi ose zavrtnja za sve zglobove robota u prostornim
    koordinatama
    """
    
    S_TELO: ClassVar[NDArray] = np.array([
        [0.0,  0.0, 1.0,       0.0, 245.2e-3,      0.0],
        [0.0, -1.0, 0.0, -234.5e-3,      0.0, 245.2e-3],
        [0.0, -1.0, 0.0,  -24.5e-3,      0.0, 245.2e-3],
        [1.0,  0.0, 0.0,       0.0,   5.5e-3,      0.0],
        [0.0, -1.0, 0.0,    5.5e-3,      0.0,  23.7e-3],
        [1.0,  0.0, 0.0,       0.0,      0.0,      0.0]
    ], dtype=float)
    """Matrica ciji su redovi ose zavrtnja za sve zglobove robota u
    koordinatama hvataca
    """

    # MappingProxyType se koristi za pravljenje konstantnih recnika
    TETA_OPSEG: ClassVar[namedtuple] = MappingProxyType({
            "1_min": np.float64(-3.05432619099),
            "1_max": np.float64( 3.05432619099),
            "2_min": np.float64(-1.91008833338),
            "2_max": np.float64(0.640012236706),
            "3_min": np.float64(-1.39434353942),
            "3_max": np.float64( 1.57009819509),
            "4_min": np.float64(-3.05013740079),
            "4_max": np.float64( 3.05432619099),
            "5_min": np.float64(-1.74532925199),
            "5_max": np.float64( 1.92003671012),
            "6_min": np.float64(-2.53002928369),
            "6_max": np.float64( 2.53002928369)
    })
    """Recnik koji sadrzi opsege i-tog zgloba u obliku TETA_OPSEG["i_min"]
    ili TETA_OPSEG["i_max"] gde je "i" oznacava redni broj zgloba. Zbog 
    ogranicene numericke preciznosti, opseg svakog zgloba je smanjen za
    0.2 stepeni, tj. i_max -= 0.1, i_min += 0.1 ili
    opseg = absolute(i_min) + absolute(i_max) - 0.2
    """

    def __post_init__(self):
        # Onesposobi promenu vrednosti polja
        self.M.flags.writeable         = False
        self.S_PROSTOR.flags.writeable = False
        self.S_TELO.flags.writeable    = False

"""
*** PRIVATNE FUNKCIJE ***
"""
def _unutar_opsega_aktuiranja(teta_lista: Sequence | NDArray) -> bool:
    # Proverava da li je su uglovi iz vektora teta_lista unutar opsega
    # aktuiranja
    return np.all(tuple(
            NiryoOne.TETA_OPSEG[f"{i + 1}_min"]
            <= teta
            <= NiryoOne.TETA_OPSEG[f"{i + 1}_max"]
            for i, teta in enumerate(teta_lista)
    ))

def _exp_proizvod(
    teta_lista: Sequence | NDArray, i: int | np.int32
) -> NDArray[np.float64]:
    # Proracunava homogenu transformacionu matricu za Niryo One kao proizvod od
    # 1. do i-tog ugla
    # exp([S_1]teta_1) exp([S_2]teta_2) ... exp([S_n]teta_n)
    exp_proizvod = np.eye(4)

    if not 1 <= i <= 6:
        raise ValueError(
            "Parametar \"i\" mora zadovoljavati 1 <= i <= 6"
        )

    for i in range(i):
        exp_proizvod = exp_proizvod@mp.exp_vek_ugao(
            NiryoOne.S_PROSTOR[i], teta_lista[i]
        )

    return exp_proizvod

def _teta12_proracun(
    pd_resenja: Dict[str, Union[NDArray[np.float64], None]],
    grana: Literal[1, 2],
    T1: NDArray[np.float64],
    presek_ose456: NDArray[np.float64]
) -> None:
    # Pomocna funkcija za odredjivanje uglova teta1 i teta2
    
    if grana != 1 and grana != 2:
        raise ValueError("Nepostojeca grana grafa")
    
    if pd_resenja[f"{grana}-1-1"] is not None:
        try:
            teta12 = kin.paden_kahan2(
                NiryoOne.S_PROSTOR[0],
                NiryoOne.S_PROSTOR[1],
                mp.SE3_proizvod_3D(
                    mp.exp_vek_ugao(
                        NiryoOne.S_PROSTOR[2], pd_resenja[f"{grana}-1-1"][2]
                    ),
                    presek_ose456
                ),
                mp.SE3_proizvod_3D(T1, presek_ose456)
            )
        except kin.PadenKahanError:            
            pd_resenja[f"{grana}-1-1"] = None
            pd_resenja[f"{grana}-1-2"] = None
            pd_resenja[f"{grana}-2-1"] = None
            pd_resenja[f"{grana}-2-2"] = None
            
            return 

        if isinstance(teta12[0], tuple):
            pd_resenja[f"{grana}-1-1"][:2] = teta12[0]
            pd_resenja[f"{grana}-1-2"][:2] = teta12[0]

            pd_resenja[f"{grana}-2-1"][:2] = teta12[1]
            pd_resenja[f"{grana}-2-2"][:2] = teta12[1]
        else:
            pd_resenja[f"{grana}-1-1"][:2] = teta12
            pd_resenja[f"{grana}-1-2"][:2] = teta12
            
            pd_resenja[f"{grana}-2-1"] = None
            pd_resenja[f"{grana}-2-2"] = None

def _teta45_proracun(
    pd_resenja: Dict[str, Union[NDArray[np.float64], None]],
    grana: Tuple[int, int],
    T1: NDArray[np.float64]
) -> None:
    # Pomocna funkcija za odredjivanje uglova teta4 i teta5
    
    podgrana = f"{grana[0]}-{grana[1]}-"
    
    if pd_resenja[podgrana + '1'] is not None:
        try:
            teta45 = kin.paden_kahan2(
                NiryoOne.S_PROSTOR[3],
                NiryoOne.S_PROSTOR[4],
                
                # Najbliza tacka ose zavrtnja 6 iz koordinatnog pocetka        
                (0.0, 0.0, NiryoOne.L[:4].sum()),
                
                mp.SE3_proizvod_3D(
                    mp.inv(
                        _exp_proizvod(pd_resenja[podgrana + '1'][:3], 3)
                    )@T1,
                    (0.0, 0.0, NiryoOne.L[:4].sum())
                )
            )
        except kin.PadenKahanError:
            pd_resenja[podgrana + '1'] = None
            pd_resenja[podgrana + '2'] = None
            
            return

        if isinstance(teta45[0], tuple):
            pd_resenja[podgrana + '1'][3:5] = teta45[0]
            pd_resenja[podgrana + '2'][3:5] = teta45[1]
        else:
            pd_resenja[podgrana + '1'][3:5] = teta45
            pd_resenja[podgrana + '2'] = None

def _teta6_proracun(
    pd_resenja: Dict[str, Union[NDArray[np.float64], None]],
    grana: Tuple[int, int, int],
    T1: NDArray[np.float64]
) -> None:
    # Pomocna funkcija za odredjivanje uglova teta4 i teta5

    grana = f"{grana[0]}-{grana[1]}-{grana[2]}"
    
    if pd_resenja[grana] is not None:
        try:
            teta6 = kin.paden_kahan1(
                (1.0, 0.0, 0.0, 0.0, NiryoOne.L[:4].sum(), 0.0),
                (1, 0, 0),
                mp.SE3_proizvod_3D(
                    mp.inv(_exp_proizvod(pd_resenja[grana][:5], 5))@T1,
                    (1, 0, 0)
                )
            )
        except kin.PadenKahanError:
            pd_resenja[grana] = None
            return
        else:
            pd_resenja[grana][5] = teta6

"""
*** API FUNKCIJE ***
"""
def dir_kin(
    teta_lista: Sequence | NDArray,
    koord_sistem_prostor: bool = True,
) -> NDArray[np.float64]:
    """Odredjuje direktnu kinematiku gde je rezultat proracuna u prostornom
    koordinatnom sistemu ili u koordinatnom sistemu hvataca
    
    Parametri
    ---------
    teta_list : Sequence | NDArray
        Spisak/lista uglova rotacije zglobova dimenzije 1x6 ili 6x1     
    koord_sistem_prostor : bool
        Odredjuje da li je matrica povratne vrednosti u prostornom koordinatnom
        sistemu-True ili u koordinatnom sistemu hvataca-False (automatska
        vrednost je True) 
    
    Povratna vrednost
    -----------------
    NDArray[np.float64]
        Homogena transformaciona matrica iz grupe SE(3) koja predstavlja
        konfiguraciju robota u prostornom koordinatnom sistemu
    
    Greske
    ------
    ValueError
        Nepravilne dimenzije parametra `teta_lista` ili `teta_lista` je van
        opsega aktuiranja

    Primeri
    -------
    >>> dir_kin(np.deg2rad([-10, 10, -10, 23, 3, -50]))
    np.array([[ 0.980,  0.194, 0.046,  0.206]
              [-0.194,  0.870, 0.453, -0.035]
              [ 0.048, -0.453, 0.890,  0.416]
              [   0.0,    0.0,    0.0,    1.0]])
    >>> dir_kin(np.deg2rad([-34, 20, -13, 2, 0, -50]), False)
    np.array([[0.823, -0.555,  0.122, -0.220]
              [0.449,  0.504, -0.738,  0.305]
              [0.348,  0.662,  0.664, -0.275]
              [  0.0,    0.0,    0.0,    1.0]])
    """
    if not _unutar_opsega_aktuiranja(teta_lista):
        raise ValueError(
            "Uglovi iz \"teta_lista\" nisu unutar opsega aktuiranja"
        )    
    
    if koord_sistem_prostor:
        return kin.dir_kin(
            NiryoOne.M,
            NiryoOne.S_PROSTOR,
            teta_lista,
        )
    else:
        return mp.inv(kin.dir_kin(
            NiryoOne.M,
            NiryoOne.S_PROSTOR,
            teta_lista
        ))

def inv_kin(
    Tk: Sequence | NDArray,
    tol_omega: float | np.float64,
    tol_v: float | np.float64,
    koord_sistem_prostor: bool = True
) -> Tuple[NDArray[np.float64], ...]:
    """Odredjuje inverznu kinematiku gde su parametri proracuna u prostornom
    koordinatnom sistemu ili u koordinatnom sistemu hvataca. Za detalje
    proracuna videti odeljak Beleske

    Parametri
    ---------
    Tk : Sequence | NDArray
        SE(3) matrica konacne/zeljene konfiguracije robota
    tol_omega : float | np.float64
        Dozvoljena tolerancija za odstupanje po uglu od zeljene konfiguracije 
    tol_v : float | np.float64
        Dozvoljena tolerancija za odstupanje po poziciji od zeljene
        konfiguracije 
    koord_sistem_prostor : bool, opcionalno
        Odredjuje da li je `Tk` matrica pozicije i orijentacije koordinatnog
        sistema hvataca u odnosu na prostorni koordinatni sistem
        (True-automatska vrednost) ili da je `Tk` matrica pozicije i
        orijentacije prostornom koordinatnog sistema u odnosu na koordinatni
        sistem hvataca (False)

    Povratna vrednost
    -----------------
    Tuple[NDArray[np.float], ...]
        Lista generalisanih koordinata zglobova cija direktna kinematika
        priblizno odgovara (unutar tolerancija `tol_omega` i `tol_v`) zeljenoj
        konfiguraciji `Tk`
        
    Beleske
    -------
        Algoritam primenjen je detaljno objasnjen u radu dostupnom na
        permalinku
        "https://github.com/VuckoT/mehanika_robota/blob/
        d5be8f70fab51edff40f040d621e5313220bb34d/
        Upravljanje%20Niryo%20One%20manipulatorom.pdf"

    Greske
    ------
    ValueError
        Nepravilne dimenzije ulaznih parametara. Tolerancije `tol_v` i
        `tol_omega` nisu >0
    InvKinError
        Inverzna kinematika nema resenja za zadate parametre
    
    Primeri
    -------
    >>> n_one.inv_kin(
        [[ 0, 0, 1,  150e-3],
         [ 0, 1, 0, -150e-3],
         [-1, 0, 0,   88e-3],
         [ 0, 0, 0,       1]],
        0.001,
        0.001
    )
    (np.array([-0.785, -0.822, -0.594, 0.0, -0.155, -0.785]),)
    >>> n_one.inv_kin(
        n_one.dir_kin(np.deg2rad([-10, 10, -10, 23, 3, -50])),
        0.001,
        0.001,
        False
    )
    (
        np.array([-0.176,  0.175, -0.174, 0.386,  0.051, -0.857]),
        np.array([-0.176,  0.175, -0.174, 0.386,  0.051, -0.857]))
    """
    # Proracun je namenjen za matricu `Tk` u prostornom koordinatnom sistemu
    if koord_sistem_prostor:
        Tk = np.array(Tk, dtype=float)
    else:
        Tk = mp.inv(Tk)

    _alati._mat_provera(Tk, (4, 4), "Tk")
    _alati._tol_provera(tol_omega, "tol_omega")
    _alati._tol_provera(tol_v, "tol_v")    

    T1 = Tk@mp.inv(
        [[1.0, 0.0, 0.0, NiryoOne.L[4:7].sum()],
         [0.0, 1.0, 0.0,                   0.0],
         [0.0, 0.0, 1.0,  NiryoOne.L[:4].sum()],
         [0.0, 0.0, 0.0,                   1.0]]
    )
    
    # Recnik aproksimativnih resenja primenom Paden-Kahanovih podproblema gde
    # smo rekli da je `NiryoOne.L[7] == 0`. Svaki objekat u recniku predstavlja
    # vektor red resenja. Ukoliko resenje ne postoji, vrednost u recniku je
    # `None`
    pd_resenja = {
        "1-1-1": None,
        "1-1-2": None,
        "1-2-1": None,
        "1-2-2": None,
        "2-1-1": None,
        "2-1-2": None,
        "2-2-1": None,
        "2-2-2": None
    }
        
    # Odredjivanje ugla teta3 #################################################

    # Vektor preseka ose zavrtnja 4, 5 i 6
    presek_ose456 = np.array(
        [NiryoOne.L[4:6].sum(), 0.0, NiryoOne.L[:4].sum()],
        dtype=float
    )
    
    # Vektor preseka ose zavrtnja 1 i 2
    presek_ose12 = np.array(
        [0.0, 0.0, NiryoOne.L[:2].sum()],
        dtype=float
    )
    
    try:
        teta3 = kin.paden_kahan3(
            NiryoOne.S_PROSTOR[2],
            presek_ose456,
            presek_ose12,
            np.linalg.norm(
                mp.SE3_proizvod_3D(T1, presek_ose456) - presek_ose12
            )
        )
    except kin.PadenKahanError:
        raise InvKinError(
            "Inverzna kinematika nema resenja za zadate parametre"
        ) from None
    
    if isinstance(teta3, tuple):
        pd_resenja["1-1-1"] = np.array(
            (0.0, 0.0, teta3[0], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-1-2"] = np.array(
            (0.0, 0.0, teta3[0], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-2-1"] = np.array(
            (0.0, 0.0, teta3[0], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-2-2"] = np.array(
            (0.0, 0.0, teta3[0], 0.0, 0.0, 0.0),
            dtype=float
        )
                    
        pd_resenja["2-1-1"] = np.array(
            (0.0, 0.0, teta3[1], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["2-1-2"] = np.array(
            (0.0, 0.0, teta3[1], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["2-2-1"] = np.array(
            (0.0, 0.0, teta3[1], 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["2-2-2"] = np.array(
            (0.0, 0.0, teta3[1], 0.0, 0.0, 0.0),
            dtype=float
        )
    else:
        pd_resenja["1-1-1"] = np.array(
            (0.0, 0.0, teta3, 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-1-2"] = np.array(
            (0.0, 0.0, teta3, 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-2-1"] = np.array(
            (0.0, 0.0, teta3, 0.0, 0.0, 0.0),
            dtype=float
        )
        
        pd_resenja["1-2-2"] = np.array(
            (0.0, 0.0, teta3, 0.0, 0.0, 0.0),
            dtype=float
        )

    # Odredjivanje uglova teta1 i teta2 #######################################
    _teta12_proracun(pd_resenja, 1, T1, presek_ose456)
    _teta12_proracun(pd_resenja, 2, T1, presek_ose456)
    
    if np.all(tuple(value is None for value in pd_resenja.values())):
        raise InvKinError(
            "Inverzna kinematika nema resenja za zadate parametre"
        )

    # Odredjivanje uglova teta4 i teta5 #######################################
    for grana in product((1, 2), repeat=2):
        _teta45_proracun(pd_resenja, grana, T1)

    if np.all(tuple(value is None for value in pd_resenja.values())):
        raise InvKinError(
            "Inverzna kinematika nema resenja za zadate parametre"
        )

    # Odredjivanje uglova teta6 ###############################################
    for grana in product((1, 2), repeat=3):
        _teta6_proracun(pd_resenja, grana, T1)
            
    if np.all(tuple(value is None for value in pd_resenja.values())):
        raise InvKinError(
            "Inverzna kinematika nema resenja za zadate parametre"
        )
    
    # Odredjivanje tacnog resenja na osnovu aproksimativnog iz `pd_resenja` ###
    resenja = []
    
    for pd_resenje in pd_resenja.values():
       if pd_resenje is not None:
            try:
                resenja.append(kin.inv_kin(
                    NiryoOne.M,
                    NiryoOne.S_PROSTOR,
                    pd_resenje,
                    Tk,
                    tol_omega,
                    tol_v
                ))
            except InvKinError:
                pass
    
    # Odbacivanje resenja koja su van granice aktuiranja zglobova #############
    resenja = (tuple(
        resenje for resenje in resenja if _unutar_opsega_aktuiranja(resenje)
    ))
    
    # Provera da li postoje resenja
    if len(resenja) != 0:
        return resenja
    else:
        raise InvKinError(
            "Inverzna kinematika nema resenja za zadate parametre"
        )