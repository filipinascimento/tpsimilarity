import toml

def get_version():
    with open("project.toml", "r") as f:
        data = toml.load(f)
    return data["project"]["version"]

__version__ = get_version()
