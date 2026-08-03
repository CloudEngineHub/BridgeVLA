import random
import socket
import re
import time
import os
import threading 
import pickle
import numpy as np
from datetime import datetime
from pynput import keyboard
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat
import transforms3d

# Dataset directory that trajectory replay reads from. There is no sensible default — point it at your own
#     export REAL_REPLAY_DATASET_DIR=/abs/path/to/your/dataset/<task_slug>
# absolute path. Unset, the replay flow exits with a hint rather than guessing a directory that does not exist.
REPLAY_DATASET_DIR = os.environ.get("REAL_REPLAY_DATASET_DIR", "")

# Robot / workstation addresses. The defaults are PLACEHOLDERS — replace them
# with your own, or override per run:
#   ARM_IP=... LOCAL_IP=... python arm_teleop.py
ARM_IP     = os.environ.get("ARM_IP", "192.168.1.6")
ARM_PORT   = int(os.environ.get("ARM_PORT", 29999))
LOCAL_IP   = os.environ.get("LOCAL_IP", "192.168.1.100")
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", 12345))

return_to_initial_pose = False
replay_motion = False
claw_open = False
claw_close = False  
set_drag = False
reset_drag = False
lift_before_return = False
key_listener = None

class Server():
    def __init__(self, ip, port, host, app_port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp = None
        self.host = host
        self.app_port = app_port
        self.baudrate = 115200
        self.modbus = None
        self.modbusRTU = None
        self.timestamp = None

    def init_com(self):
        self.modbus = self.get_id(self.modbus)
        self.modbusRTU = self.get_id(self.modbusRTU)

    def get_id(self, response):
        match = re.search(r'\{(.+?)\}', response)
        if match:
            numbers_str = match.group(1)
            return int(numbers_str)

    def start_server(self):
        """Start the server"""
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app.bind((self.host, self.app_port))
        self.app.listen(5)
        print(f"Server listening on {self.host}:{self.app_port}")

        try:
            while True:
                self.tcp, addr = self.app.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                client_handler = threading.Thread(
                    target=self.handle_client,
                    args=(self.tcp, )
                )
                client_handler.daemon = True
                client_handler.start()
        except KeyboardInterrupt:
            print("Stopping server...")
            self.app.close()
            self.tcp.close()

    def handle_client(self,sock):
        """Handle communication with one client"""
        global return_to_initial_pose
        global replay_motion
        global claw_open
        global claw_close
        try:
            while True:
                data = sock.recv(1024).decode().strip()
                if not data:
                    print("Client disconnected")
                    break
                print(f"Received data: {data}")
                if data == 'replay':
                    replay_motion = True
                elif data == 'back':
                    return_to_initial_pose = True
                elif data == 'open':
                    claw_open = True
                elif data == 'close':
                    claw_close = True
                sock.send("Message received".encode())
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            sock.close()

class Point():
    def __init__(self, name:str, position: dict):
        self.name = name
        self.position = position
        self.timestamp = None
        self.position_str = None
        self.claw = None
        self.position_to_string()

    def get_timestamp(self):
        dt = datetime.now()
        micro = dt.microsecond // 1000
        self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
        return self.timestamp

    def position_to_string(self):
        if self.position == None:
            return None
        self.position_str = f"{{{self.position['x']:.4f},{self.position['y']:.4f},{self.position['z']:.4f},\
            {self.position['rx']},{self.position['ry']},{self.position['rz']}}}"
        return self.position_str


def find_latest_number_folder(folder_path):
    """Find the folder with the largest numeric name"""
    if not os.path.exists(folder_path):
        return None
        
    dir_list = os.listdir(folder_path)
    number_folders = []
    
    for name in dir_list:
        full_path = os.path.join(folder_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            number_folders.append(int(name))
    
    if not number_folders:
        return None
    
    # Return the folder name with the largest number
    max_number = max(number_folders)
    max_number=0#gai
    return str(max_number)


def wait_and_prompt(sock, point=None, state = True, replay = False):
    while get_status(sock) not in [5, 6]:
        time.sleep(0.1)
    if not state:
        user_input = input("Arm idle; press Enter to continue to the next action: ")
        while user_input.lower() != '':
            print("Invalid input; press Enter to confirm.")
            user_input = input("Arm idle; press Enter to continue to the next action: ")

ptest = Point('ptest', None)

def replay_motion_trajectory(sock, modbus, timestamp, replay=True):
    trajectory_points = []
    gripper_states = []  # gripper states
    global ptest
    print("Trajectory file found; press Enter to replay it, any other key to cancel:")
    if replay:
        user_choice = input().lower()
        if user_choice != '':
            print("Replay cancelled.")
            return 0
    
    try:
        # Find the folder with the largest numeric name
        base_folder = REPLAY_DATASET_DIR  # collection dataset path, see REPLAY_DATASET_DIR at the top of the file
        if not base_folder:
            print("[config] No trajectory-replay dataset directory configured. Set it to your own absolute path:\n"
                  "         export REAL_REPLAY_DATASET_DIR=/abs/path/to/your/dataset/<task_slug>")
            return 0
        if not os.path.isdir(base_folder):
            print(f"[config] Dataset directory does not exist: {base_folder}\n"
                  f"         Point REAL_REPLAY_DATASET_DIR at the right directory")
            return 0
        latest_folder = find_latest_number_folder(base_folder)
        
        if latest_folder is None:
            print("No numerically named folder found")
            return 0
        
        actions_folder = os.path.join(base_folder, latest_folder, 'actions')
        
        if not os.path.exists(actions_folder):
            print(f"actions folder does not exist: {actions_folder}")
            return 0
        
        # Collect every pkl file and sort numerically
        pkl_files = []
        for filename in os.listdir(actions_folder):
            if filename.endswith('.pkl') and filename[:-4].isdigit():
                pkl_files.append((int(filename[:-4]), filename))
        
        pkl_files.sort(key=lambda x: x[0])  # sort numerically
        
        # Read each pkl file
        for number, filename in pkl_files:
            pkl_path = os.path.join(actions_folder, filename)
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Data format: [x, y, z, qx, qy, qz, qw, gripper_state]
                if len(data) >= 8:
                    x, y, z = data[0]*1000, data[1]*1000, data[2]*1000
                    pose_quat = data[3:7]
                    gripper_state = data[7]  # gripper state
                    pose_eurl = transforms3d.euler.quat2euler(pose_quat, axes='sxyz')
                    pose_eurl = np.rad2deg(np.asarray(pose_eurl))
                    # Convert the quaternion to Euler angles
                    rx = float(pose_eurl[0])  
                    ry = float(pose_eurl[1])
                    rz = float(pose_eurl[2])
                    
                    
               
                    trajectory_points.append({
                        'x': x, 'y': y, 'z': z,
                        'rx': rx, 'ry': ry, 'rz': rz
                    })
                    gripper_states.append(gripper_state)
                else:
                    print(f"Warning: {filename} has the wrong data format (8 values expected); skipping")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        
        
        if len(trajectory_points) > 0:
            print(f"Starting trajectory replay: {actions_folder}")
            last_gripper_state = None  # previous gripper state, to avoid repeating the same command
            
            for i, point in enumerate(trajectory_points):
                point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']:.4f},{point['ry']:.4f},{point['rz']:.4f}}}"
                send_movj_command(sock, point_str)
                wait_and_prompt(sock, state=True, replay=False)
                
                # Check whether the gripper state changed
                if i < len(gripper_states):
                    current_gripper_state = gripper_states[i]
                    if last_gripper_state is None or current_gripper_state != last_gripper_state:
                        print(f"Gripper state changed: {last_gripper_state} -> {current_gripper_state}")
                        if current_gripper_state == 1:
                            print("Closing the gripper")
                            claws_control(sock, 0, modbus, ptest)  # 0 = close
                        elif current_gripper_state == 0:
                            print("Opening the gripper")
                            claws_control(sock, 1, modbus, ptest)  # 1 = open
                        last_gripper_state = current_gripper_state
                
            print("Trajectory replay complete.")
        else:
            print("No valid trajectory points found")
    except Exception as e:
        print(f"Trajectory replay failed: {e}")

def main():
    global return_to_initial_pose
    global replay_motion
    global claw_open
    global claw_close
    global set_drag
    global reset_drag
    global lift_before_return  # flag
    server = Server(ARM_IP, ARM_PORT, LOCAL_IP, LOCAL_PORT)
    server.sock.connect((server.ip, server.port))
    initialize_robot(server.sock)
    server_modbus = f'ModbusCreate("{ARM_IP}", 502,2)'
    server_modbusrtu = 'ModbusRTUCreate(1, 115200, "N", 8, 1)'
    server.modbus = send_modbus_command(server.sock, server_modbus)
    server.modbusRTU = send_modbus_command(server.sock,server_modbusrtu)
    server.init_com()
    
    p1 = Point('p1', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    p2 = Point('p2', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': -3.2263, 'rz': -59.2746})
    p3 = Point('p3', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p4 = Point('p4', {'x': -187.2389, 'y':-610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p5 = Point('p5', {'x': -77.1242, 'y': -610.7508, 'z': 129.0285, 'rx': 90.2763, 'ry': -2.4535, 'rz': -88.7509})
    p6 = Point('p6', {'x': 168.5996, 'y': -223.9824, 'z': 60, 'rx': 90.1654, 'ry': 3.2263, 'rz': -59.2746})
    p7 = Point('p7', {'x': 173.2972, 'y': -130.0664, 'z': 167.6071, 'rx': 91.3785, 'ry': -0.9884, 'rz': -67.1547})
    point_list = [p1, p2, p3, p4, p5, p6, p7]

    def on_key_press(key):
        """Triggers on key-down; no Enter needed."""
        global return_to_initial_pose
        global claw_open
        global claw_close
        global set_drag
        global reset_drag
        global lift_before_return
        try:
            ch = key.char
        except AttributeError:
            return
        if ch == 'i':  # return straight to the initial position
            return_to_initial_pose = True
            lift_before_return = False
        elif ch == 'u':  # lift first, then return to the initial position
            return_to_initial_pose = True
            lift_before_return = True
        elif ch == 'e':
            claw_close = True
        elif ch == 'r':
            claw_open = True
        elif ch == 'o':
            set_drag = True
        elif ch == 'p':
            reset_drag = True

    global key_listener
    key_listener = keyboard.Listener(on_press=on_key_press)
    key_listener.start()

    print("Keys (act on key-down, no Enter needed):")
    print("'i' - return straight to the initial position")
    print("'u' - lift first, then return to the initial position")
    print("'e' - close the gripper")
    print("'r' - open the gripper")
    print("'o' - enable drag mode")
    print("'p' - reset drag mode")

    try:
        while True:
            while not (replay_motion or return_to_initial_pose or claw_open or claw_close or set_drag or reset_drag):
                time.sleep(0.05)
            print("Starting arm motion ...")
            
            dt = datetime.now()
            micro = dt.microsecond // 1000
            timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_right")
            
            if replay_motion:
                replay_motion_trajectory(server.sock, server.modbusRTU, timestamp_start)
            elif return_to_initial_pose:
                if lift_before_return:  # lift first when requested
                    # Read the current position
                    current_pose = send_command(server.sock, "GetPose()")
                    try:
                        # Extract the pose values from the response
                        pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                        x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                        
                        lift_pose = f"{{{x+20:.4f},{y:.4f},{z+150:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                        send_movj_command(server.sock, lift_pose)
                        wait_and_prompt(server.sock)
                        current_pose = send_command(server.sock, "GetPose()")
                        pose_data = re.search(r'\{(.+?)\}', current_pose).group(1)
                        x, y, z, rx, ry, rz = map(float, pose_data.split(','))
                        lift_pose = f"{{{x+100:.4f},{y-110:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f}}}"
                        send_movj_command(server.sock, lift_pose)
                        wait_and_prompt(server.sock)
                        print("Lifted; returning to the initial position")
                    except Exception as e:
                        print(f"Failed to read or parse the current position: {e}")

                # Home joint pose to return to; adjust for your own workspace.
                initial_pose = [224, 21, -87, -18 ,88, 37]

                valid_length = np.arange(0, 10, 0.1)
                valid_deg = np.arange(0, 1 ,0.1)
                initial_pose = [f"{num:.4f}" for num in initial_pose]
                initial_pose_str = f"{{{','.join(initial_pose)}}}"
                send_movjoint_commad(server.sock, initial_pose_str)
                time.sleep(2)
                
            elif claw_close:
                claws_control(server.sock, 0, server.modbusRTU, p2)
                claws_control_degree(server.sock, server.modbusRTU, 0, None)
            elif claw_open:
                claws_control(server.sock, 1, server.modbusRTU, p4)
            elif set_drag:
                send_command(server.sock, 'StartDrag()')
            elif reset_drag:
                send_command(server.sock,'ClearError()')
            
            print("Arm motion complete.")
            
            replay_motion = False
            return_to_initial_pose = False
            claw_open = False
            claw_close = False
            set_drag = False
            reset_drag = False
            
    finally:
        if key_listener is not None:
            key_listener.stop()
        close_modbus(server.sock)
        server.sock.close()
        server.app.close()

def initialize_robot(sock):
    send_command(sock, "PowerOn()")
    time.sleep(1)
    send_command(sock, "EnableRobot()")
    send_command(sock, "ClearError()")

def send_command(sock, command):
    sock.sendall(f"{command}\n".encode('utf-8'))
    response = sock.recv(1024).decode('utf-8')
    print(f"Command: {command}")
    print(f"Response: {response}")
    return response

def send_movj_command(sock, point):
    command = f"MovJ(pose={point},a=30,v=30)"
    send_command(sock, command)

def send_movjoint_commad(sock,point):
    command = f"MovJ(joint={point},a=30,v=30)"
    send_command(sock, command)

def send_modbus_command(sock, command):
    command = f"{command}"
    return send_command(sock, command)

def claws_send_command(sock, id, num1, num2, num3):
    command = f'SetHoldRegs({id}, {num1}, {num2}, {{{num3}}}, "U16")'
    send_command(sock, command)

def claws_control(sock, status, id, point):
    if status:
        claws_send_command(sock, id, 258, 1, 0)
        claws_send_command(sock, id, 259, 1, 1)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 0)
        time.sleep(1)
        point.claw = 0
    else:
        claws_send_command(sock, id, 258, 1, 1)
        claws_send_command(sock, id, 259, 1, 0)
        claws_send_command(sock, id, 264, 1, 1)
        claws_send_command(sock, 0, 258, 1, 1) 
        time.sleep(1)
        point.claw = 1
def claws_control_degree(sock, id, set_degree, point):
        if point != None:
            point.claw = set_degree
        if set_degree < 0:
            set_degree = 0
        if set_degree > 100:
            set_degree = 100
        control_value = int(9000 - set_degree * 9000 / 100) # linear map from angle to control value

        claws_send_command(sock, id, 258, 1, 0)
        claws_send_command(sock, id, 259, 1, control_value)
        claws_send_command(sock, id, 264, 1, 1)
        time.sleep(1)
def close_modbus(sock):
    for index in range(4):
        send_modbus_command(sock, f'Modbusclose({index})')
        send_modbus_command(sock, f'Modbusclose({index})')
def get_status(sock):
    command = "RobotMode()"
    response = send_command(sock, command)
    status_code = int(response.split(',')[1][1])
    return status_code

if __name__ == "__main__":
    main()