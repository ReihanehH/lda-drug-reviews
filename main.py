from src.simulation.simulation import Simulation


def main() -> None:
    """
    The main function that initializes and runs the simulation.

    Args:
        None

    Returns:
        None
    """
    simulation = Simulation(config_path="./config/config.yaml")
    simulation.run()


if __name__ == "__main__":
    main()
