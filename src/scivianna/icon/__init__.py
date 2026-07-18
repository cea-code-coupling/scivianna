from pathlib import Path

def get_icon(icon_name: str):
    with open(str(Path(__file__).parent / icon_name) + ".svg", "r") as f:
        return f.read()
