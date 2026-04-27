import warnings
import os
import socket
import time

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'


# --- UNITY NETWORK CONFIGURATION ---
UNITY_HOST = "172.20.10.2" # ⚠️ CHANGE THIS to the second laptop's IP address
UNITY_PORT = 65432


# --- NETWORK COMMUNICATION ---
def send_spell_to_unity(spell_name):
    # Map the recognized spell name from your templates to Unity's expected format
    spell_map = {
        "fireball": "F",
        "lightning": "L",
        "heal": "H"
        # Add more mappings here if you add more spells later
    }

    spell_char = spell_map.get(spell_name.lower())

    if not spell_char:
        print(f"⚠️  Spell '{spell_name}' is not mapped to a Unity command yet.")
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Add a 2-second timeout so the program doesn't freeze if Unity is off
            s.settimeout(2.0)
            s.connect((UNITY_HOST, UNITY_PORT))
            s.sendall(spell_char.encode())
            print(f"⚡ Successfully sent to Unity: {spell_char} ({spell_name})")
    except ConnectionRefusedError:
        print("⚠️ Unity is not running or the server on the second laptop is not started.")
    except socket.timeout:
        print("⚠️ Connection to Unity timed out. Double-check the IP address and ensure both laptops are on the same Wi-Fi.")
    except Exception as e:
        print(f"⚠️ Network error: {e}")


def run_spell_hotkeys():
    """Simple console controls.

    Press:
      1 = fireball
      2 = lightning
      3 = heal
      q = quit

    Note: This uses `input()` so you must press ENTER after the key.
    """
    print("\nSpell hotkeys ready:")
    print("  [1] Fireball")
    print("  [2] Lightning")
    print("  [3] Heal")
    print("  [q] Quit\n")

    while True:
        cmd = input("Enter spell key (1/2/3) or q: ").strip().lower()

        if cmd == "1":
            print("Casting: fireball")
            send_spell_to_unity("fireball")

        elif cmd == "2":
            print("Casting: lightning")
            send_spell_to_unity("lightning")

        elif cmd == "3":
            print("Casting: heal")
            send_spell_to_unity("heal")

        elif cmd == "q":
            print("Quitting.")
            break

        else:
            print("Unknown input. Use 1, 2, 3, or q.")

        # small pause to keep output readable if the user pastes inputs
        time.sleep(0.05)


# --- MAIN ---
if __name__ == '__main__':
    run_spell_hotkeys()
