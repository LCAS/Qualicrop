#!/usr/bin/env python3
"""
File: grblHAL Controller Class (RIGController)
Author: Robert Stevenson
Handles communication with grblHAL-controlled BTT SKR Pico over USB serial for the HSI Rig
"""

import serial
import serial.tools.list_ports
import time


class RIGController:
    def __init__(self, port=None, baudrate=115200):
        self.serial_conn = None
        self.port = port
        self.baudrate = baudrate
        self.feed_rate = 1000  # Default feed rate in mm/min

    def list_ports(self):
        """List available serial ports"""
        ports = serial.tools.list_ports.comports()
        print("Available serial ports:")
        for i, port in enumerate(ports):
            print(f"{i + 1}: {port.device} - {port.description}")
        return ports

    def connect(self, port=None):
        """Connect to the grblHAL controller"""
        if port:
            self.port = port

        # if no port was passed then we look and select a port from the list of found ports
        if not self.port:
            ports = self.list_ports()
            if not ports:
                print("No serial ports found!")
                return False

            try:
                # select a valid port
                choice = int(input("Select port number: ")) - 1
                self.port = ports[choice].device
            except (ValueError, IndexError):
                print("Invalid selection!")
                return False

        try:  # try to connect to the controller
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2,
                # NOTE: These setting are rarely changed
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            time.sleep(2)  # We need to wait after connecting for it to initialise fully

            # simple command to test connection
            # NOTE:(could be removed later)
            response = self.send_command("?")
            if response:
                print(f"Connected to grblHAL on {self.port}")
                print(f"Response: {response}")  # (NOTE: maybe this is all that is needed to be removed in the future)
                return True
            else:
                print("No response from controller")
                return False

        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the controller"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.port = None
            print("Disconnected from controller")

    def send_command(self, command):
        """Send a command to grblHAL and return the response"""
        # check if we are connected to controller before senting connection
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Not connected to controller!")
            return None

        try:
            # Add newline if not present
            if not command.endswith('\n'):
                command += '\n'

            # Send command
            self.serial_conn.write(command.encode('utf-8'))
            self.serial_conn.flush()

            # Read response
            response = ""
            start_time = time.time()
            while time.time() - start_time < 3:  # 3 second timeout
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8')
                    response += data
                    # NOTE: We check the response to see if the command has been accepted my the controller yet
                    if 'ok' in response.lower() or 'error' in response.lower():
                        break
                time.sleep(0.1)

            return response.strip()

        except Exception as e:
            print(f"Error sending command: {e}")
            return None

    def home_axes(self, axes="YZ"):
        """Home specified axes (Y, Z, or both)"""

        # We have the ability to home both the axes individualy or at the same time.
        # NOTE: homing at the same time will cause a higher power draw which we might want to reduce
        axes = axes.upper()
        if axes == "YZ" or axes == "ZY":
            command = "$H"
            print("Homing Y and Z axes...")
        elif axes == "Y":
            command = "$HY"
            print("Homing Y axis...")
        elif axes == "Z":
            command = "$HZ"
            print("Homing Z axis...")
        else:
            print("Invalid axes! Use Y, Z, or YZ")
            return False

        # NOTE: This user interaction could be removed if intergrate into a better/autonomous system
        # print("WARNING: Ensure the machine is clear of obstacles!")
        # confirm = input("Continue with homing? (y/n): ")
        # if confirm.lower() != 'y':
        #     print("Homing cancelled")
        #     return False

        response = self.send_command(command)
        print(f"Homing response: {response}")
        return True

    # TODO: Implement a version that converts meters per second to mm/min
    def set_feed_rate(self, rate):
        """Set the feed rate (Speed) for movements (non-firmware)"""
        try:
            self.feed_rate = float(rate)
            print(f"Feed rate (Speed) set to {self.feed_rate} mm/min")
            return True
        except ValueError:
            print("Invalid feed rate! Please enter a number.")
            return False

    def move_axis(self, axis, distance, rapid=False):
        """Move specified axis by given distance"""
        axis = axis.upper()
        if axis not in ['Y', 'Z']:
            print("Invalid axis! Use Y or Z")
            return False

        try:
            distance = float(distance)
        except ValueError:
            print("Invalid distance! Please enter a number.")
            return False

        # Use G0 for rapid movement, G1 for controlled movement
        # NOTE: G0 (rapid) using the set Feed rate of the device firmware
        # NOTE: G1 (feed) move at the accompaning feedrate passed with the command
        move_type = "G0" if rapid else "G1"

        # Create movement command
        if rapid:
            command = f"{move_type} {axis}{distance}"
        else:
            command = f"{move_type} {axis}{distance} F{self.feed_rate}"  # this feedrate is from setting it before in this program

        print(f"Moving {axis} axis to absolute position {distance}mm...")
        response = self.send_command(command)
        print(f"Movement response: {response}")
        return True

    def get_status(self):
        """Get current machine status"""
        response = self.send_command("?")
        if response:
            print(f"Status: {response}")
        return response

    def emergency_stop(self):
        """Send emergency stop command"""
        print("EMERGENCY STOP!")
        self.send_command("!")
        time.sleep(0.5)
        self.send_command("\x18")  # Ctrl-X reset

    def reset_controller(self):
        """Send soft reset command to controller"""
        print("Resetting controller...")
        response = self.send_command("\x18")  # Ctrl-X soft reset
        time.sleep(1)  # Give controller time to reset
        print(f"Reset response: {response}")
        return response

    def get_current_position(self):
        """Get current machine position from status"""
        response = self.send_command("?")
        if response and 'MPos:' in response:
            # Parse position from status response like: <Idle|MPos:0.000,10.000,5.000|...>
            try:
                pos_start = response.index('MPos:') + 5
                pos_end = response.index('|', pos_start)
                positions = response[pos_start:pos_end].split(',')
                # Assuming Y is second, Z is third (no X-axis)
                return {'Y': float(positions[1]), 'Z': float(positions[2])}
            except (ValueError, IndexError):
                print("Could not parse position from response")
                return None
        return None
    
    def is_connected(self):
        if self.serial_conn:
            return self.serial_conn.is_open
        return False

    # THIS IS PURELY FOR ILLUSTRATIVE/TESTING PURPOSE
    def run_test_routine(self, cam_height=0.0, scan_speed=443.33, init_pos=80.0, scan_pos=650.0):
        """
        Run a test routine for scanning:
        1. Reset controller and home axes
        2. Move Y axis to position 80 at rate 4000 mm/min
        3. Wait until position is reached
        4. Set rate to 443 mm/min and move to position 650
        5. Move back to position 80 at rate 4000 mm/min
        """

        TIMEOUT = 95
        TRAVEL_SPEED = 6000

        print("\n=== Starting Test Routine ===\n")

        if not self.serial_conn or not self.serial_conn.is_open:
            print("ERROR: Not connected to controller!")
            return False

        # Step 1: Reset and home
        print("Step 1: Resetting controller and homing axes...")
        self.reset_controller()
        time.sleep(2)

        # Home without user confirmation
        print("Homing Y and Z axes...")
        response = self.home_axes()
        print(f"Homing response: {response}")

        # Wait for homing to complete by checking status
        print("Waiting for homing to complete...")
        start_time = time.time()
        while time.time() - start_time < TIMEOUT:  # HOMING take a max time of 90 secs (95s is safe)
            pos = self.get_current_position()
            if pos is not None and pos["Y"] == 0.0 and pos["Z"] == 0.0:  # check to see if our position are y = 0mm and z = 0mm, if true exit the waiting
                print(f"Homing finished early @ \n{pos}")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("Homing finished")

        # Step 3: move and wait till in camara scan init position
        print("moving into initial pos")
        self.set_feed_rate(TRAVEL_SPEED)
        # move to scan init pos
        self.move_axis("Y", init_pos)

        # wait till we at init position
        while time.time() - start_time < TIMEOUT:
            pos = self.get_current_position()
            if pos is not None and pos["Y"] == init_pos and pos["Z"] == cam_height:  # check to see if our position are y = 0mm and z = -758mm, if true exit the waiting
                print("At initial position early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("At initial position")

        # Step 4: Set rate to 443 and move to 650
        print("\nStep 4: Moving to Y position 650mm at 443mm/min...")
        # set speed to match frame rate of camera
        self.set_feed_rate(scan_speed)

        ## START CAMERA CAPTURE
        print(f"!!! START CAMERA CAPTURE !!!")
        # TODO: Run Camera capture function

        # move so that we do a full bed scan
        self.move_axis("Y", scan_pos)
        # wait till we at do a full bed scan
        while time.time() - start_time < TIMEOUT:
            pos = self.get_current_position()
            if pos is not None and pos["Y"] == scan_pos and pos["Z"] == cam_height:  # check to see if our position are y = 0mm and z = -758mm, if true exit the waiting
                print("Finished Full bed scan early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms
        print("Full bed scann finished")

        print("\nStep 5: Moving back to Y position 80mm at 4000mm/min...")
        print("move back into initial pos")
        self.set_feed_rate(TRAVEL_SPEED)
        # move to scan init pos
        self.move_axis("Y", init_pos)

        # wait till we at init position
        while time.time() - start_time < TIMEOUT:
            pos = self.get_current_position()
            if pos is not None and pos["Y"] == init_pos and pos["Z"] == cam_height:  # check to see if our position are y = 0mm and z = -758mm, if true exit the waiting
                print("Back to initial position early early")
                break  # Condition met
            time.sleep(0.1)  # Check every 100ms

        print("\n=== Test Routine Completed Successfully ===\n")
        return True