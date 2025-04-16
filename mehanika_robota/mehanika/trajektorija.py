"""
Trajektorija
============
Modul za generisanje trajektorija 

Preporucen nacin uvoza modula je
>>> import mehanika_robota.mehanika.trajektorija as traj
"""

"""
*** BIBLIOTEKE ***
"""
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple, Sequence, Mapping
from mehanika_robota import _alati
import mehanika_robota.mehanika.mat_prostor as mp

"""
*** PRIVATNE FUNKCIJE ***
"""

"""
*** API FUNKCIJE ***
"""
def vreme_skal(
    t: np.float64 | Sequence | NDArray[np.float64],
    t_ukupno: np.float64,
    stepen: Literal[3, 5]
) -> np.float64 | NDArray[np.float64]:
    """Odredjuje parametar vremenskog skaliranja kao polinom petog ili sedmog
    stepena

    Parametri
    ---------
    t : np.float64 | NDArray[np.float64]
        Trenutno vreme koje zadovoljava 0 <= t <= t_ukupno. Ukoliko je dato kao
        matrica tipa Sequence ili NDArray[np.float64], obracunava parametar
        vremenskog skaliranje za svaki elemenat matrice posebno
    t_ukupno : np.float64
        Ukupno vreme kretanja
    stepen : Literal[3, 5]
        Stepen polinoma vremenskog skaliranja 

    Povratna vrednost
    -----------------
    np.float64 | NDArray[np.float64]
        Parametar vremenskog skaliranja u trenutku t. Ukoliko je t skalar, onda
        je povratna vrednost skalarna, ukoliko je t matrica, onda je povratna
        matrica istih dimenzija cije vrednosti odgovaraju korespodentnim
        vrednostima matrici t

    Greske
    ------
    ValueError
        Parametar t ili t_ukupno nisu unutar granica 0 <= t <= t_ukupno i
        t_ukupno > 0. Stepen polinoma nije 3 ili 5

    Primeri
    -------
    >>> vreme_skal(0.6, 2, 3)
    np.float64(0.216)
    >>> vreme_skal([[0.2], [1]], 2, 3)
    np.array([[0.028], [0.5]])
    >>> vreme_skal([[1, 2], [3, 4]], 5, 5)
    np.array([[0.058, 0.317], [0.683, 0.942]])
    """
    t = np.array(t, dtype=float)
    t_ukupno = np.float64(t_ukupno)
    
    if t_ukupno <= 0.0:
        raise ValueError(
            "Parametar \"t_ukupno\" ne zadovoljava uslov t_ukupno > 0"
        )

    if np.any((t < 0.0) | (t > t_ukupno)):
        raise ValueError(
            "Parametar \"t\" ne zadovoljava uslov 0 <= t <= t_ukupno"
        )
    
    if stepen == 3:
        return 3.0*(t/t_ukupno)**2 - 2.0*(t/t_ukupno)**3
    elif stepen == 5:
        return (
            10.0*(t/t_ukupno)**3
            - 15.0*(t/t_ukupno)**4
            + 6.0*(t/t_ukupno)**5
        )
    else:
        raise ValueError("Stepen polinoma nije 3 ili 5")

def pravolin_traj(
    T_start: Sequence | NDArray,
    T_fin: Sequence | NDArray,
    skal: Mapping[Literal["n"], np.int64]
        | Mapping[Literal["skal"], Sequence | NDArray]
        | Mapping[
            Literal['t_ukupno'] | Literal['stepen'] | Literal["n"],
            np.float64 | Literal[3, 5] | np.int64
        ]
) -> Tuple[NDArray[np.float64]]:
    """Odredjuje parametar vremenskog skaliranja kao polinom petog ili sedmog
    stepena

    Parametri
    ---------
    T_start : Sequence | NDArray
        Pocetna konfiguracija trajektorije. Matrica je iz grupe SE(3)
    T_fin : Sequence | NDArray
        Finalna konfiguracija trajektorije. Matrica je iz grupe SE(3)
    skal : Mapping[Literal["n"], np.int64] | Mapping[Literal["skal"], Sequence
    | NDArray] | Mapping[Literal['t_ukupno'] | Literal['stepen'], np.float64
    | Literal[3, 5]]
        Parametar skaliranja. Mora biti definisano na jedan od tri nacina:
        1. "n" (linearno skaliranje) - broj tacaka za linearno skaliranje. Mora
        biti >= 2;
        2. "skal" (proizvoljno skaliranje) - proizvoljan vektor skaliranja koji
        mora zadovoljiti sledece uslove:
            - `0 <= skal <= 1`,
            - `skal[0] == 0.0`,
            - `skal[-1] == 1.0`,
            - dimenzije 1xm ili mx1,
            - `len(skal) >= 2` i
            - monotono rastuci elementi.
        3. "t_ukupno", "stepen" i "n" (vremensko skaliranje) recnik sa ukupnim
        vremenom, stepenom polinoma i broj tacaka skaliranja koji predstavljaju
        parametre vremenskog skaliranja. Treba biti oblika {"t_ukupno":
        np.float64, "stepen": Literal[3, 5], "n": np.int64}. 

    Povratna vrednost
    -----------------
    Tuple[NDArray[np.float64]]
        Trajektorija sacinjena od vise matrica iz grupe SE(3) skalirana na
        nacin odredjen parametrom `skal`. Prva matrica trajektorije je
        `T_start`, dok je poslednja `T_fin`.

    Greske
    ------
    ValueError
        Nepravilne dimenzije matrica `T_start` i `T_fin`. Parametar `skal` ne
        ispunjava navedene uslove u odelju Parametri

    Primeri
    -------
    >>> vreme_skal(0.6, 2, 3)
    np.float64(0.216)
    >>> vreme_skal([[0.2], [1]], 2, 3)
    np.array([[0.028], [0.5]])
    >>> vreme_skal([[1, 2], [3, 4]], 5, 5)
    np.array([[0.058, 0.317], [0.683, 0.942]])
    """
    if all(param in skal for param in ("t_ukupno", "stepen", "n")):
        t_ukupno = np.float64(skal["t_ukupno"])
        stepen = np.int64(skal["stepen"])
        n = np.int64(skal["n"])

        if n < 2:
            raise ValueError("Parametar \"skal[\"n\"]\" mora biti >= 2")

        # `n == 2` je poseban slucaj gde zelimo prazan vektor skal i kao takav
        # regulisacemo ga tako da uvek bude pravilno odredjen
        if n == 2:
            skal = np.array([], dtype=float)
        else:
            podeok = t_ukupno/(n - 1.0)

            # Posto je prvi i poslednji element liste `traj` odredjen sa
            # T_start i T_fin, potrebni su nam elementi vektora skaliranja
            # izmedju te dve matrice
            skal = vreme_skal(
                np.linspace(podeok, t_ukupno - podeok, n - 2),
                t_ukupno,
                stepen
            )

    elif "n" in skal:
        n = np.int64(skal["n"])

        if n < 2:
            raise ValueError("Parametar \"skal[\"n\"]\" mora biti >= 2")

        # `n == 2` je poseban slucaj gde zelimo prazan vektor skal i kao takav
        # regulisacemo ga tako da uvek bude pravilno odredjen
        if n == 2:
            skal = np.array([], dtype=float)
        else:
            podeok = 1.0/(n - 1.0)

            # Posto je prvi i poslednji element liste `traj` odredjen sa
            # `T_start`` i `T_fin`, potrebni su nam elementi vektora skaliranja
            # izmedju te dve matrice
            skal = np.linspace(podeok, 1.0 - podeok, n - 2)

    elif "skal" in skal:
        skal = np.array(skal["skal"], dtype=float)

        # Algoritam je napravljen da funkcionise sa vektorom reda `skal`
        if skal.ndim <= 2:
            if skal.ndim == 2:
                if skal.shape[1] > 1:
                    raise ValueError(
                        "Vektor \"skal[\"skal\"]\" mora biti dimenzije 1xm "
                        "ili mx1"
                    )
                else:
                    skal.reshape(skal.shape[0])
        else:
            raise ValueError(
                "Vektor \"skal[\"skal\"]\" mora biti dimenzije 1xm ili mx1"
            )

        if len(skal) < 2:
            raise ValueError(
                "Vektora \"skal[\"skal\"]\" mora imati >= 2 elemenata"
            )

        if not np.all(np.diff(skal) >= 0):
            raise ValueError(
                "Vektor \"skal[\"skal\"]\" mora biti monotono rastuci"
            )

        if not np.isclose(skal[0], 0.0):
            raise ValueError(
                "\"skal[\"skal\"]\" mora imati \"skal[\"skal\"][0]\" == 0.0"
            )

        if not np.isclose(skal[-1], 1.0):
            raise ValueError(
                "\"skal[\"skal\"]\" mora imati \"skal[\"skal\"][-1]\" == 1.0"
            )
        # Posto je prvi i poslednji element liste `traj` odredjen sa T_start i
        # T_fin, potrebni su nam elementi vektora skaliranja izmedju te dve
        # matrice
        skal = skal[1:-1]

    
    else:
        raise ValueError(
            "Parametar \"skal\" mora imati jedan od sledecih oblika: "
            "\"skal[\"n\"]: np.int64\", "
            "\"skal[\"skal\"]: Sequence | NDArray\", ili "
            "\"skal\" == {\"t_ukupno\": np.float64, \"stepen\": "
            "Literal[3, 5], \"n\": np.float64}"
        )

    T_start = np.array(T_start, dtype=float)
    T_fin = np.array(T_fin, dtype=float)

    _alati._mat_provera(T_start, (4, 4), "T_start")
    _alati._mat_provera(T_fin, (4, 4), "T_fin")

    R_start, p_start = mp.SE3_rastavi(T_start)
    R_fin, p_fin = mp.SE3_rastavi(T_fin)

    R_start_fin = R_start.T @ R_fin
    p_start_fin = p_fin - p_start

    # Naprave sve konfiguracije trajektorije bez prve i poslednje konfiguracije
    # (njihovo mesto bice `T_start` i `T_fin`)
    traj = [
        mp.SE3_sastavi(
            R_start @ mp.exp(mp.log(R_start_fin)*skal[i]),
            p_start + (p_start_fin)*skal[i]        
        ) for i in range(len(skal))
    ]

    traj.insert(0, T_start)
    traj.append(T_fin)

    return tuple(traj)