import struct
import socket
import time

# TCP Connection Parameters
MESSAGE_SIZE = 24
DELIMITER = b"\n"
TCP_PORT = 50000
BUFFER_SIZE = MESSAGE_SIZE if MESSAGE_SIZE else 32


def send_data(conn, val):
    """Sends a double-precision number."""
    msg = struct.pack(">d", val)
    conn.send(msg)


def receive_data(conn):
    """Receives three double-precision numbers."""
    data = b""
    while len(data) < 24:
        data += conn.recv(24 - len(data))

    val1, val2, Time = struct.unpack(">ddd", data)
    return val1, val2, Time

def bind_socket(ip, tcp_port, max_retries=5, delay=1):
    retries = 0
    while retries < max_retries:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, tcp_port))
            print(f"Successfully bound to {ip}")
            return s
        except socket.error as e:
            print(f"Failed to bind {ip}: {e}. Retrying {retries + 1}/{max_retries}...")
            retries += 1
            time.sleep(delay)
    raise RuntimeError(f"Failed to bind {ip} after {max_retries} retries.")

def websocket(ip, tcp_port):
    s = bind_socket(ip, tcp_port)
    s.listen(1)
    print(f"Waiting for Simulink to start on {ip}")
    conn, addr = s.accept()
    print(f"TCP connection established on {ip}")
    return conn



