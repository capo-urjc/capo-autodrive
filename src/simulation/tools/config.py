import yaml
from box import Box


def read_config(config_file: str):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            return CarlaGeneratorConfig(config)  # Use Box to enable dot notation access
    except (yaml.YAMLError, FileNotFoundError, IOError) as exc:
        raise RuntimeError(f"Error reading the YAML configuration file: {exc}")


class CarlaGeneratorConfig(Box):
    pass



