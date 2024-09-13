"""Upravljanje Niryo One manipulatorom"""

"""
*** BIBLIOTEKE ***
"""
from mehanika_robota import niryo_one as n_one
import pyniryo as pn
import os
import logging
from msvcrt import getch

"""
*** KONSTANTE ***
"""
NIRYO_ONE_ETHERNET_IP = "169.254.200.200"

POZICIJA1 = n_one.inv_kin(
    [[ 0.0, 0.0, 1.0,  148e-3],
     [ 0.0, 1.0, 0.0, -148e-3],
     [-1.0, 0.0, 0.0,  150e-3],
     [ 0.0, 0.0, 0.0,     1.0]],
    0.1,
    0.1
)[0]

POZICIJA2 = n_one.inv_kin(
    [[ 0.0, 0.0, 1.0,  148e-3],
     [ 0.0, 1.0, 0.0, -148e-3],
     [-1.0, 0.0, 0.0,   88e-3],
     [ 0.0, 0.0, 0.0,     1.0]],
    0.001,
    0.001
)[0]

POZICIJA3 = n_one.inv_kin(
    [[ 0.0, 0.0, 1.0,  148e-3],
     [ 0.0, 1.0, 0.0,  148e-3],
     [-1.0, 0.0, 0.0,  150e-3],
     [ 0.0, 0.0, 0.0,     1.0]],
    0.1,
    0.1
)[0]

POZICIJA4 = n_one.inv_kin(
    [[ 0.0, 0.0, 1.0, 148e-3],
     [ 0.0, 1.0, 0.0, 148e-3],
     [-1.0, 0.0, 0.0,  88e-3],
     [ 0.0, 0.0, 0.0,    1.0]],
    0.001,
    0.001
)[0]

"""
*** FUNKCIJE ***
"""
def ocisti_terminal():
    """Cisti terminal na COMMAND.COM i cmd.exe CLI
    """
    os.system("cls")
    
def gasenje_robota(robot: pn.NiryoRobot, izuzetak: Exception = None) -> None:
    """Procedura sigurnog gasenja robota

    Parametri
    ---------
    robot : pn.NiryoRobot
        Robot koji treba ugasiti
    izuzetak : Exception, opcionalno
        Ukoliko je nastala greska, pribeleziti izuzetak koji je nastao
        (automatska vrednost je None)
        
    Primeri
    -------
        >>> robot = pn.NiryoRobot("169.254.200.200")
        >>> gasenje_robota(robot)
        print("Robot je uspesno ugasen")
        >>> robot = pn.NiryoRobot("169.254.200.200")
        >>> gasenje_robota(robot, ValueError)
        print(
            "Doslo je do neocekivane greske, proverite greske.log za detalje"
        )
    """
    ocisti_terminal()
    
    if izuzetak is not None:
        logging.error(
            f"{robot.get_hardware_status()}\n"
            f"-----------------------------------------------------------------\n"
            f"Pozicija zglobova = {robot.get_joints()}\n"
            f"-----------------------------------------------------------------\n"
            f"{izuzetak}\n\n"
        )
            
    robot.set_learning_mode(True)
    robot.close_connection()
    
    if izuzetak is not None:
        print(
            "Doslo je do neocekivane greske, proverite greske.log za vise "
            "detalja"
        )
    else:
        print("Robot je uspesno ugasen")
        

def main():   
    # Log podesavanje
    logging.basicConfig(
        filename    = "../log/greske.log",
        level       = logging.ERROR,
        format      = "%(asctime)s %(message)s",
        datefmt     = "Date: %d-%m-%Y   Time: %H:%M:%S"
    )
    # Povezivanje standardnom adresom robota za Ethernet TCP/IP komunikaciju
    try:
        robot = pn.NiryoRobot(NIRYO_ONE_ETHERNET_IP)
    except Exception as izuzetak:
        logging.error(f"{izuzetak}\n\n")
        print(
            "Nije moguce povezati se sa Niryo One robotom, videti greske.log "
            "za vise detalja"
        )
        return 
    
    try:
        # Kalibracija
        robot.calibrate(pn.CalibrateMode.AUTO)
        print("Pritisnite taster [d] da potvrdite rucnu kalibraciju: ")
        
        while getch().decode() != 'd':
            pass
        
        robot.calibrate(pn.CalibrateMode.MANUAL)
        print("Rucna kalibracija je uspesna! Molim Vas, udaljite se od robota")
        robot.wait(3)

        # Priblizi se objektu
        robot.set_arm_max_velocity(70)
        robot.move_joints(POZICIJA1)

        # Pokupi objekat
        robot.set_arm_max_velocity(30)
        robot.move_joints(POZICIJA2)
        robot.wait(1)
        robot.close_gripper(hold_torque_percentage=60)
        robot.wait(1)
        robot.move_joints(POZICIJA1)
        
        # Pomeri se iznad konacne pozicije
        robot.set_arm_max_velocity(70)
        robot.move_joints(POZICIJA3)
        
        # Ispusti objekat
        robot.set_arm_max_velocity(30)    
        robot.move_joints(POZICIJA4)
        robot.wait(1)
        robot.open_gripper()
        robot.wait(1)
        robot.move_joints(POZICIJA3)
        
        # Vrati se u pocetnu konfiguraciju
        robot.set_arm_max_velocity(70)
        robot.move_to_home_pose()

    except Exception as izuzetak:
        gasenje_robota(robot, izuzetak)
    else:
        print("Program je uspesno zavrsen")
        gasenje_robota(robot)

if __name__ == "__main__":
    ocisti_terminal()
    main()
