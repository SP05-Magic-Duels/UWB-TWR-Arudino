import socket

HOST = "172.20.10.2"
PORT = 65432

while True:
    spell = input("Enter spell (F = fire, L = lightning, H = healing): ").strip().upper()

    if spell not in ["F", "L", "H"]:
        print("Invalid input")
        continue

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(spell.encode())
            print(f"Sent: {spell}")
    except ConnectionRefusedError:
        print("Unity is not running or server not started.")


while True:
    spell = input("Enter spell (F = fire, L = lightning, H = healing): ").strip().upper()

    if spell not in ["F", "L", "H"]:
        print("Invalid input")
        continue

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(spell.encode())
            print(f"Sent: {spell}")
    except ConnectionRefusedError:
        print("Unity is not running or server not started.")