import json
import socket
import threading
import queue
import time
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

def load_config(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON config: {e}")

cfg = load_config(CONFIG_PATH)

CLIENT_HOST = cfg.get("client_host", "0.0.0.0")
CLIENT_PORT = cfg.get("client_port", 5000)

REMOTE_HOST = cfg.get("remote_host", "127.0.0.1")
REMOTE_PORT = cfg.get("remote_port", 6000)

RECONNECT_DELAY = float(cfg.get("reconnect_delay", 3.0))

CLIENT_TIMEOUT = float(cfg.get("client_timeout", 30))
LISTEN_BACKLOG = int(cfg.get("listen_backlog", 10))

msg_queue = queue.Queue()

remote_sock = None

def connect_remote():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((REMOTE_HOST, REMOTE_PORT))
            print(f"[REMOTE] Connected {REMOTE_HOST}:{REMOTE_PORT}")
            return s

        except socket.timeout:
            print("[REMOTE] timeout")
        except ConnectionRefusedError:
            print("[REMOTE] refused")
        except socket.gaierror:
            print("[REMOTE] DNS error")
        except OSError as e:
            print(f"[REMOTE] OS error: {e}")

        time.sleep(RECONNECT_DELAY)


def remote_sender():
    global remote_sock

    remote_sock = connect_remote()

    while True:
        msg = msg_queue.get()

        while True:
            try:
                remote_sock.sendall((msg + "\n").encode())

                ack = remote_sock.recv(1024)

                if not ack:
                    raise ConnectionResetError("remote closed")

                if ack.decode().strip() == "ACK":
                    print("[REMOTE] ACK")
                    break

            except socket.timeout:
                print("[REMOTE] ACK timeout")
            except ConnectionResetError:
                print("[REMOTE] reset")
                remote_sock = connect_remote()
            except BrokenPipeError:
                print("[REMOTE] broken pipe")
                remote_sock = connect_remote()
            except OSError as e:
                print(f"[REMOTE] socket error: {e}")
                remote_sock = connect_remote()

        msg_queue.task_done()

def handle_client(conn, addr):
    print(f"[CLIENT] connected {addr}")

    try:
        while True:
            try:
                data = conn.recv(4096)

                if not data:
                    print(f"[CLIENT] disconnected {addr}")
                    break

                msg = data.decode(errors="ignore").strip()

                if not msg:
                    continue

                msg_queue.put(msg)

                conn.sendall(b"OK\n")

            except ConnectionResetError:
                print(f"[CLIENT] reset {addr}")
                break
            except socket.timeout:
                print(f"[CLIENT] timeout {addr}")
            except OSError as e:
                print(f"[CLIENT] error {addr}: {e}")
                break

    finally:
        conn.close()

def client_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((CLIENT_HOST, CLIENT_PORT))
    server.listen(LISTEN_BACKLOG)

    print(f"[SERVER] listening {CLIENT_HOST}:{CLIENT_PORT}")

    while True:
        try:
            conn, addr = server.accept()
            conn.settimeout(CLIENT_TIMEOUT)

            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            ).start()

        except OSError as e:
            print(f"[SERVER] accept error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=remote_sender, daemon=True).start()
    client_server()