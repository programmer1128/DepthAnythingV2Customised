import socket
import struct
import requests

# Replace with your actual Modal deployment URL
MODAL_URL = "https://programmer1128--depth-pipeline-app-web-endpoint.modal.run"
# 6 uint32_t (4 bytes each) + 1 uint64_t (8 bytes) = 32 bytes
HEADER_FORMAT = "<6IQ" 
HEADER_SIZE = 32

def start_proxy():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 9092))
    server.listen(5)
    print("Local Proxy listening on 127.0.0.1:9092...")

    while True:
        conn, addr = server.accept()
        header_data = conn.recv(HEADER_SIZE)
        
        if len(header_data) < HEADER_SIZE:
            conn.close()
            continue
        
        # Unpack to find the payload_len (index 6)
        header = struct.unpack(HEADER_FORMAT, header_data)
        payload_len = header[6]
        
        payload = bytearray()
        while len(payload) < payload_len:
            packet = conn.recv(payload_len - len(payload))
            if not packet: 
                break
            payload.extend(packet)

        # Forward the exact raw bytes (header + image payload) to Modal
        response = requests.post(MODAL_URL, data=(header_data + payload))
        
        # The C++ client expects a raw float matrix back 
        conn.sendall(response.content)
        conn.close()

if __name__ == "__main__":
    start_proxy()