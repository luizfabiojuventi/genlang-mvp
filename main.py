import sys, os
from genlang.runtime import parse_script, run_program

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <arquivo.gs>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    cmds = parse_script(text)
    run_program(cmds, echo=True)

if __name__ == "__main__":
    main()
