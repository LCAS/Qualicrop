import socket,threading,queue,struct,pickle
HOST = "127.0.0.1"  # Server IP
PORT = 54000        # Server Port
data_list=list()
CHUNK_SIZE=917504
#CHUNK_SIZE=512



def tcp_listener(server_host=HOST, server_port=PORT):
    count = 1
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_host, server_port))
    print(f"[Client] Connected to {server_host}:{server_port}")

    try:
        while True:
            if count<=80:
                data = client_socket.recv(CHUNK_SIZE)
                if not data:
                    print("[Client] Connection closed by server.")
                    break
                uint16_array = np.frombuffer(data, dtype=np.uint16)
                spectra = np.asarray(uint16_array).reshape(448, 1024)
                data_list.append(data)
                count =count + 1
            else:
                with open('specim_data.pkl', 'wb') as f:
                    pickle.dump(data_list, f)
                break
            #print(f"[Client] Received: {data.decode('utf-8')}")
    except Exception as e:
        print(f"[Client] Error: {e}")
        client_socket.close()
        print("[Client] Disconnected.")



def client(command,q):
    data=None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode())
            data = s.recv(1024)

    except Exception as e:
            data=e
    q.put(data)

def receive_all(sock, length):
    """Helper function to receive `length` bytes exactly."""
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed before receiving all data")
        data += more
    return data


def listener():
    while True:
        try:
            data = receive_all(client, 500)
            #message = data.decode('utf-8')

            msg_length = struct.unpack("i", data)[0]
            message_bytes = receive_all(client, msg_length)
            print("Received:", message_bytes.decode('utf-8'))
        except Exception as e:
            print("Connection closed or error:", e)
            break


import socket




def send_command(command):
    # Create thread
    output='OK'
    q = queue.Queue()
    try:
        thread = threading.Thread(target=client, args=(command,q))
        # Start thread
        thread.start()

        # Wait for the thread to finish
        thread.join()
    except Exception as e:
        output=e
    return output,q.get()

def receive_data():
    # Create thread
    output='OK'
    q = queue.Queue()
    try:
        thread = threading.Thread(target=tcp_listener)
        # Start thread
        thread.start()

        # Wait for the thread to finish
        thread.join()
    except Exception as e:
        output=e
    return output,q.get()