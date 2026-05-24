from detect_players_pose import run_all as run_detection
from separacion_por_equipo import run_all as run_team_separation
from equipo_atacante import run_all as run_offside_detection
from seguimiento_atacantes_decision import run_all as run_tracking


def main():
    print("\n==============================")
    print("[1/4] DETECCION DE JUGADORES")
    print("==============================")
    run_detection()

    print("\n==============================")
    print("[2/4] SEPARACION POR EQUIPOS")
    print("==============================")
    run_team_separation()

    print("\n==============================")
    print("[3/4] DETECCION DE OFFSIDE")
    print("==============================")
    run_offside_detection()

    print("\n==============================")
    print("[4/4] TRACKING + DECISION")
    print("==============================")
    run_tracking()

    print("\n==============================")
    print("PIPELINE COMPLETO FINALIZADO")
    print("==============================")


if __name__ == "__main__":
    main()