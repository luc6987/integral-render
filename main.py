from pathlib import Path
from src.linear_system import main_export_linear_system
from src.solve import solve_from_csv
from src.render import main_from_csv


def main():
    csv_dir = Path(__file__).parent / "linear_system_csv"
    print("[Main] Exporting A and b (from setup.scenarios)...")
    main_export_linear_system()
    print("[Main] Solving Ax=b from CSV ...")
    solve_from_csv(csv_dir)
    print("[Main] Rendering from CSV solution ...")
    main_from_csv()


if __name__ == "__main__":
    main()
