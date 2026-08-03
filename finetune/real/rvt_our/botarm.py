import random
import socket
import re
import time
import os
import threading 
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import numpy as np
from pymodbus.client import ModbusTcpClient
import transforms3d
import random
import struct

import glob
import logging
from logging.handlers import RotatingFileHandler

# Global tool coordinate system index (Dobot Dashboard: Tool(index))
TOOL_INDEX = 3
DASHBOARD_PORT = 29999

# Robot / workstation addresses for the manual entry points at the bottom of
# this file. The defaults are PLACEHOLDERS — replace them with your own, or
# override per run:  ARM_IP=... LOCAL_IP=... python botarm.py
ARM_IP     = os.environ.get("ARM_IP", "192.168.1.6")
LOCAL_IP   = os.environ.get("LOCAL_IP", "192.168.1.100")
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", 12345))

log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs")

if not os.path.exists(log_directory):
    try:
        os.makedirs(log_directory)
    except Exception as e:
        raise

logger = logging.getLogger("dobot_log")
logger.setLevel(logging.INFO)

log_filename = f'dobot_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
file_handler = RotatingFileHandler(os.path.join(log_directory, log_filename),maxBytes= 1024*1024*200,backupCount=10)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
        self.signal = {'replay':False,'claw_open':False,'claw_close':False, 'set_drag':False, 'reset_drag':False}
    def init_com(self, bot):
        msg_modbus = f'Create("{self.ip}", 502, 2)'
        msg_modbusrtu = f'Create(1, {self.baudrate}, "N", 8, 1)'
        self.modbus = self._parse_id(bot.send_modbus_command('Modbus', msg_modbus))
        self.modbusRTU = self._parse_id(bot.send_modbus_command('ModbusRTU', msg_modbusrtu))

    def _parse_id(self, response):
        match = re.search(r'\{(.+?)\}', response)

        if match:
            numbers_str = match.group(1)
            return int(numbers_str)
    def start_server(self):
        """Start the server"""
        self.app = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.app.bind((self.host, self.app_port))
        self.app.listen(5)  # backlog of 5 pending connections
        print(f"Server listening on {self.host}:{self.app_port}")

        try:
            while True:
                self.tcp, addr = self.app.accept()
                print(f"Accepted connection from {addr[0]}:{addr[1]}")
                # Handle the client on a new thread
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

    def handle_client(self):
        """Handle communication with one client"""
        try:
            while True:
                data = self.sock.recv(1024).decode().strip()
                if not data:
                    print("Client disconnected")
                    break
                print(f"Received data: {data}")
                if data == 'replay':
                    self.signal['replay'] = True
                elif data == 'open':
                    self.signal['claw_open'] = True
                elif data == 'close':
                    self.signal['claw_close'] = True
                elif data == 'set':
                    self.signal['set_drag'] = True
                elif data == 'reset':
                    self.signal['reset_drag'] = True
                self.sock.send("Message received".encode())
        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            # Close the client socket
            self.sock.close()
class Point():
    def __init__(self,  position: list, quaternion: list, claw, gripper_thres=None, name=None):
        self.name = name  
        self.position = position
        self.quaternion = quaternion
        self.euler = None
        self.timestamp = None
        self.position_quaternion_claw = None
        self.claw = claw
        self.gripper_thres = gripper_thres
        self.__position_to_string()
        self.__quaternion_to_euler()
        self.__get_position_and_quaternion()
    def get_timestamp(self):
        dt = datetime.now()
        micro = dt.microsecond // 1000
        self.timestamp = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}")
        return self.timestamp
    def __position_to_string(self):
        if self.position is None:
            return None
        self.position = f"{{{self.position[0]:.4f},{self.position[1]:.4f},{self.position[2]:.4f},"
        
    # Quaternion -> Euler angles
    def __quaternion_to_euler(self):

        # Convert the list to a numpy array if needed
        self.quaternion = np.array(self.quaternion)

        # Make sure the quaternion is a unit quaternion
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)

        # Build a Rotation object
        rotation = R.from_quat(self.quaternion)

        # Convert to Euler angles (radians)
        euler_angles_rad = rotation.as_euler('xyz')


        # Convert to Euler angles (degrees)
        euler_angles_deg = np.degrees(euler_angles_rad)
    # --- Angle-wrap detection ---
    # --- Angle-wrap correction ---
        if hasattr(self, 'last_euler_deg'):  # if a previous angle exists
            for i in range(3):  # iterate over rx, ry, rz
                delta = euler_angles_deg[i] - self.last_euler_deg[i]
                
                # A change beyond 180° means a periodic wrap (e.g. +178° -> -178°)
                if abs(delta) > 270:
                    # Take the shortest path (±360°)
                    if delta > 270:
                        euler_angles_deg[i] -= 360
                    elif delta < -270:
                        euler_angles_deg[i] += 360
        self.last_euler_deg = euler_angles_deg  # keep the current angle for the next comparison
        self.euler = f"{euler_angles_deg[0]:.4f},{euler_angles_deg[1]:.4f},{euler_angles_deg[2]:.4f}}}"

        return euler_angles_deg



  
    def __get_position_and_quaternion(self):
        self.position_quaternion_claw = self.position + self.euler

class DobotController():
    def __init__(self, sock):
        self.sock = sock
        self.modbus = None
        self.modbusRTU = None
        self.current_pose = None
        
    
    def _initialize(self, server):
        self.send_command("PowerOn()")
        time.sleep(1)
        self.send_command("EnableRobot()")
        self.send_command("ClearError()")
        self.send_command(f"Tool({TOOL_INDEX})")
        self.modbus = server.modbus
        self.modbusRTU = server.modbusRTU
    def point_control(self, point = None):
        print(f"Target pose: {point.position_quaternion_claw}")
        if point != None:
            pose = point.position_quaternion_claw
            pose_value = [float(v) for v in pose.strip('{}').split(',')]
            if self.current_pose is not None:
                cur_pose_value = [float(v) for v in self.current_pose.strip('{}').split(',')]
                z_cur = cur_pose_value[2]
                z_next = pose_value[2]

                h = 100.
                if z_cur < 75. and z_next >75. :
                    pose_value_1 = cur_pose_value
                    pose_value_1[2] = h
                    pose_1 = "{" + ",".join(map(str, pose_value_1)) + "}"
                    print(f"Inserting lift mid-point: {pose_1}")
                    self.current_pose = pose_1
                    self.move_joint(pose_1)
            self.current_pose = pose
            return self.move_joint(point.position_quaternion_claw)

    def Pause(self):
        """Pause the robot"""
        res = self.send_command("Pause()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers
    def Continue(self):
        """Resume motion"""
        res = self.send_command("Continue()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers
    def send_command(self, command: str) -> str:
        """Send a command and return the response"""
        self.sock.sendall(f"{command}\n".encode())
        response = self.sock.recv(1024).decode().strip()
        logger.info(f"Command: {command}")
        logger.info(f"Response: {response}")

        print(f"Command: {command}")
        print(f"Response: {response}")
        return response
    def get_pose(self):
        """Get the arm's current pose"""
        res = self.send_command("GetPose()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers
    ####11111111111111111111
    def get_angle(self):
        res = self.send_command("GetAngle()")
        part = res.split('{')[1].split('}')[0] 
        numbers = part.split(',')
        print(part)
        return numbers

    def send_modbus_command(self, modbus: str, msg):
    # Send a ModBus command
        command = f"{modbus}{msg}" 
        return self.send_command(command)
    def claws_send_command(self, id, num1, num2, num3):
        command = f'SetHoldRegs({id}, {num1}, {num2}, {{{num3}}}, "U16")'
        self.send_command(command)
    def claws_read_command(self, id, num1, num2):
        command = f'GetHoldRegs({id}, {num1}, {num2}, "U16")'
        return self.send_command(command)
    def _read_claw_status(self) -> tuple:  # read the gripper state
        """Read the claw status via Modbus."""  
        client = ModbusTcpClient(self.ip, port=502)  # Modbus connection to robot IP  
        if not client.connect():  
            return False, "Failed to connect to Modbus server."  

        try:  
            unit_id = 2  # Device unit ID  
            read_address = 258  # Register address for claw status  
            read_response = client.read_holding_registers(read_address, count=1, unit=unit_id)  

            if not read_response.isError():  
                return True, read_response.registers[0]  # Return flag indicating success and claw status  
            else:  
                return False, str(read_response)  # Return error message if reading fails  
        except Exception as err:  
            print(f"Error reading claw status: {err}")  
            return False, err  
        finally:  
            client.close()  # Ensure the Modbus connection is closed    

    def joint_inverse_kin(self, pose: list, useJointNear=False, JointNear=[0, 0, 0, 0, 0, 0]):
        """Joint-motion command wrapper"""
        response = self.send_command(
            f"InverseKin({','.join(str(v) for v in pose)}, "
            f"useJointNear={str(int(True))}, "
            f"jointNear={{{','.join(str(v) for v in JointNear)}}})"
        )
        res_flag = int(response.split(',')[0])
        if res_flag !=0:
             return None

        start = response.find('{')
        end = response.find('}')

        # Extract the contents of the first pair of braces
        res = response[start + 1:end]
        print(res)
        # Convert the extracted string into a list
        res_joint = [float(num) for num in res.split(',')]
        return res_joint

    def control_movement(self, mode, value: list, a=30, v=30, wait_flag=True):
        """Joint-motion command wrapper"""
        ALLOWED_MODES = {'joint', 'pose'}
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid mode: {mode}. Allowed modes are: {ALLOWED_MODES}")
        value_str = ",".join(str(v) for v in value)
        response = self.send_command(f"MovJ({mode}={{{value_str}}},a={a},v={v})")
        return response

    def claws_control(self, status, id, point = None):  # (1 = open, 0 = close)
        if point != None:
            point.claw = status
        if status: # open the gripper
            self.claws_send_command(id, 258, 1, 0)
            self.claws_send_command(id, 259, 1, 1)
            self.claws_send_command(id, 264, 1, 1)
            self.claws_send_command(0, 258, 1, 0)
            time.sleep(1)
        else: # close the gripper
            self.claws_send_command(id, 258, 1, 1)
            self.claws_send_command(id, 259, 1, 0)
            self.claws_send_command(id, 264, 1, 1)
            self.claws_send_command(0, 258, 1, 1) 
            time.sleep(1)


    def move_l_pose(self,pose:str,a=30,v=30):
        logger.info(f"move_l_pose :{pose}")
        response = self.send_command(f"MovL(pose={pose},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag


    def move_point_pose(self,pose:str,a=30,v=30):
        logger.info(f"move_point_pose :{pose}")
        response = self.send_command(f"MovJ(pose={pose},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
    def move_joint_pose(self,pose:str,a=30,v=30):
        logger.info(f"move_joint_pose :{pose}")
        response = self.send_command(f"MovJ(joint={pose},a={a},v={v})")
        res_flag = int(response.split(',')[0])
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag

    def judge_goal_reached(self,mode = None,target_position =None):
        start_time = time.time()
        if(target_position == None or mode ==None):
            try:
                status = 1
                while(status):
                    end_time = time.time()
                    if end_time - start_time >20:
                        break
                    status = self.status
                    time.sleep(0.1)
                    if status not in [5, 6]:
                        if status == 9:
                            raise RuntimeError("the arm is in an error state")
                        pass
                    elif status == 5:
                        break
            except BaseException as e:
                logger.error("error code:", status)
                raise e
        else:
            if mode == 'pose':
                try:
                    while(True):
                        end_time = time.time()
                        if end_time - start_time >20:
                            break
                        if self.status == 9:
                            raise RuntimeError("the arm is in an error state")
                        current_pose = self.get_pose
                        if None == current_pose:
                            continue
                        pos_error = math.dist(current_pose[:3], target_position[:3])
                        if(pos_error<1):
                            break
                        time.sleep(0.1)
                        
                except BaseException as e:
                    raise e
            elif mode =='joint':   
                try:
                    while(True):
                        print("___________________")
                        end_time = time.time()
                        if end_time - start_time >20:
                            break
                        if self.status == 9:
                            raise RuntimeError("the arm is in an error state")
                        current_joint = self.get_angle()
                        if None == current_joint:
                            continue
                        current_joint = [float(x) for x in current_joint]


                        angle_error = max(abs(c - t) for c, t in zip(current_joint[:6], target_position[:6]))
                        if(angle_error <1):
                            break
                        time.sleep(0.1)
                        
                except BaseException as e:
                    raise e    
            
    def move_joint(self, pose: str, a=30, v=30):
        """Joint-motion command wrapper"""
        res_flag =1
        if res_flag != 0:
            self.clear_error()
            logger.error('An error occurred with pose-specified motion. Attempting to use joint motion.')
            current_joint = self.get_angle()
            value = [float(v) for v in pose.strip('{}').split(',')]



            max_attempts = 5
            for attempt in range(max_attempts):
                if attempt == 0:
                    inverse_joint = self.joint_inverse_kin(value, useJointNear=True, JointNear=current_joint)
                else:
                    target_near = [float(j) + random.uniform(-5, 5) for j in value]  # tune the perturbation range as needed
                    inverse_joint = self.joint_inverse_kin(target_near, useJointNear=True, JointNear=current_joint)
                    logger.warning(f"Attempt {attempt+1}: Using perturbed target pose: {target_near}")

                if inverse_joint is None:
                    continue




                inverse_joint_str = ",".join(str(v) for v in inverse_joint)
                response_joint = self.send_command(f"MovJ(joint={{{inverse_joint_str}}},a={a},v={v})")
                f_inverse_joint = [float(x) for x in inverse_joint]

                self.judge_goal_reached("joint",f_inverse_joint)
                res_flag = int(response_joint.split(',')[0])
                if res_flag ==0:
                    return 'Success'
                else:
                    logger.error(f"attempt {attempt+1} failed with error code {res_flag}")
                    self.clear_error()

            
            
        if res_flag == 0:
            return 'Success'
        else:
            return res_flag
    def clear_error(self):
        """Clear the current alarm"""
        command = f"ClearError()"
        self.send_command(command)
    def interrupt_close(self):
        """Abort on error"""
        for index in range(4):
            self.send_modbus_command('Modbus', f'close({index})')
            self.send_modbus_command('Modbus', f'close({index})')
    @property
    def status(self) -> int:
        """Get the arm's status"""
        response = self.send_command("RobotMode()")
        return int(response.split(',')[1][1])

    def switch_drag(self, status: bool):

        """Toggle drag mode"""
        command = f"StartDrag()" if status else "StopDrag()"
        self.send_command(command)

    def wait_and_prompt(self, replay = True, point = None):
        while self.status not in [5, 6]:
            time.sleep(0.1)
        if replay:
            user_input = input("Arm idle; press Enter to continue to the next action: ")

            while user_input.lower() != '':
                print("Invalid input; press Enter to confirm.")
                user_input = input("Arm idle; press Enter to continue to the next action: ")

    def wait_and_control(self, replay = True, point = None):
        while self.status not in [5, 6]:
            time.sleep(0.1)
        if replay:
            user_input = input("Arm idle; press Enter to continue to the next action: ")
            while user_input.lower() != 'q' or user_input.lower() != 'e':
                print("Invalid input; press 'q' or 'e' to confirm.")
                if user_input.lower()=='q':
                    return True
                elif user_input.lower()=='e':
                    return False
                user_input = input("Arm idle; press Enter to continue to the next action: ")

    def replay_motion_trajectory(self, modbus, timestamp, replay = True):
        trajectory_points = []
        cnt = 0
        print("Trajectory file found; press Enter to replay it, any other key to cancel:")
        if replay:
            user_choice = input().lower()
        
            if user_choice != '':
                print("Replay cancelled; still waiting for the 'a' key ...")
                return 0
        def find_latest_timestamp_folder(folder_path):
            dir_list = os.listdir(folder_path)
            timestamp_folders = []
            pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_left$'# timestamp format
            for name in dir_list:
                if os.path.isdir(os.path.join(folder_path, name)) and re.match(pattern, name):
                    timestamp_folders.append(name)
            if not timestamp_folders:
                return None
            # Sort by timestamp and take the newest folder
            timestamp_folders = sorted(
                timestamp_folders,
                key=lambda x: datetime.strptime(x[:19], "%Y-%m-%d_%H-%M-%S"),
                reverse=True
            )
            latest_folder = timestamp_folders[0]
            return latest_folder
        try:
            folder_path = find_latest_timestamp_folder('data/left/')
            with open(os.path.join('data/left/', folder_path, 'pose.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) >=7:  # at least a timestamp plus 6 pose values
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        rx = float(parts[4])
                        ry = float(parts[5])
                        rz = float(parts[6])
                        trajectory_points.append({
                            'x':x, 'y':y, 'z':z,
                            'rx':rx, 'ry':ry, 'rz':rz
                        })
                    except ValueError:
                        print(f"Warning: invalid data line: {line}")
            if len(trajectory_points) >=1:
                print(f"Starting trajectory replay: {os.path.join('data/left/', folder_path, 'pose.txt')}")
                for point in trajectory_points:
                    point_str = f"{{{point['x']:.4f},{point['y']:.4f},{point['z']:.4f},{point['rx']},{point['ry']},{point['rz']}}}"
                    self.move_joint(point_str)
                    self.wait_and_prompt(replay = False)
                    if cnt == 1:
                        self.wait_and_prompt(replay = True)
                        self.claws_control(0, modbus)
                        
                    elif cnt == 4:
                        self.claws_control(1, modbus)
                        self.wait_and_prompt(replay = False)
                    cnt += 1
                self.switch_drag(True)
                print("Trajectory replay complete.")
            else:
                print("Too few trajectory points; replay skipped")
        except FileNotFoundError:
            print("Trajectory file not found; cannot replay")

def open_gripper():
    server = Server(ARM_IP, DASHBOARD_PORT, LOCAL_IP, LOCAL_PORT)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    bot._initialize(server)

    # Open the gripper immediately on startup
    bot.claws_control(1, server.modbusRTU)
    print("Gripper opened.")
def main():
    def generate_position():
        """Generate random position data (adjust the ranges to your setup)"""
        return {
            'x': round(random.uniform(-1000, 1000), 4),
            'y': round(random.uniform(-1000, 1000), 4),
            'z': round(random.uniform(0, 500), 4),
            'rx': random.randint(0, 360),
            'ry': random.randint(-180, 180),
            'rz': random.randint(-180, 180)
        }
    # Create a TCP/IP socket
    server = Server(ARM_IP, DASHBOARD_PORT, LOCAL_IP, LOCAL_PORT)
    # Connect to the DoBot arm's Dashboard port (29999)
    server.sock.connect((server.ip, server.port))
    bot = DobotController(server.sock)
    server.init_com(bot)
    # Initialise the robot
    bot._initialize(server)
    listen_app = threading.Thread(target = server.start_server, args = ())
    listen_app.start()
    
    print(bot.claws_read_command(server.modbusRTU, 258, 2))
    time.sleep(1)
    bot.claws_control(0, server.modbusRTU)
    time.sleep(1)
    print(bot.claws_read_command(server.modbusRTU, 258, 1))
    time.sleep(1)
    bot.claws_control(1, server.modbusRTU)
    time.sleep(1)
    print(bot.claws_read_command(server.modbusRTU, 258, 1))
    try:
        while True:

            print("Press 'q' to replay the trajectory, 'e' to close the gripper, 'r' to open it ...")

            # Wait for the user to press 'a'
            while not any(server.signal.values()):
                pass
            print("Starting arm motion ...")
            
            # # Send the joint-motion commands in order
            dt = datetime.now()
            micro = dt.microsecond // 1000
            timestamp_start = dt.strftime(f"%Y-%m-%d_%H-%M-%S-{micro:03d}_right")

            if server.signal['replay']:
                bot.replay_motion_trajectory(server.modbusRTU, timestamp_start)
            elif server.signal['claw_close']:
                bot.claws_control(0, server.modbusRTU)
                bot.wait_and_prompt()
            elif server.signal['claw_open']:
                bot.claws_control(1, server.modbusRTU)
                bot.wait_and_prompt()
            elif server.signal['set_drag']:
                bot.get_pose()
            elif server.signal['reset_drag']:
                bot.switch_drag(False)
            elif server.signal['play']:
                for point in point_list:
                    joint_positions = point.position
                    joint_angles = f"{{{joint_positions['x']:.4f},{joint_positions['y']:.4f},{joint_positions['z']:.4f},{joint_positions['rx']},{joint_positions['ry']},{joint_positions['rz']}}}"
                    bot.move_joint(joint_angles)
                    bot.wait_and_prompt()

            print("Arm motion complete.")
            for key in server.signal:
                server.signal[key] = False
            
    finally:
        # Close the socket connection
        bot.interrupt_close()
        server.sock.close()
        server.app.close()

if __name__ == "__main__":
    main()
