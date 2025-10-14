#!/usr/bin/env python3
"""
grblHAL Controller Class
Handles communication with grblHAL-controlled BTT SKR Pico over USB serial
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
            distance = -1 * float(distance)
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

        print(f"Moving {axis} axis by {distance}mm...")
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
