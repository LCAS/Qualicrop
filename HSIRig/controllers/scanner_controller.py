#!/usr/bin/env python3
"""
Interactive command line interface for controlling the CNC Controller
NOTE: THIS IS DESIGNED FOR A MACHINE THAT DOES NOT HAVE AN X-AXIS
"""
from .grblhal_controller import GRBLController

class ScannerController:
    def __init__(self, port=None, baudrate=115200):
        self.serial_conn = None
        self.port = port
        self.baudrate = baudrate
        self.feed_rate = 1000  # Default feed rate in mm/min
        self.controller = GRBLController()

    def send_command(self,command):
        try:
            if not command:
                return
            command = command.strip().split()
            cmd = command[0].lower()

            if cmd == "quit" or cmd == "exit":
                return
            elif cmd == "connect":
                self.controller.connect(port="COM3")
            elif cmd == "home":
                axes = command[1] if len(command) > 1 else "YZ"
                self.controller.home_axes(axes)
            elif cmd == "speed":
                if len(command) < 2:
                    print("Usage: speed <rate>")
                    return
                self.controller.set_feed_rate(command[1])
            elif cmd == "move": # This one used speed set via the `speed` command above
                if len(command) < 3:
                    print("Usage: move <Y|Z> <distance>")
                    return
                self.controller.move_axis(command[1], command[2], rapid=False)
            elif cmd == "rapid":
                if len(command) < 3:
                    print("Usage: rapid <Y|Z> <distance>")
                    return
                self.controller.move_axis(command[1], command[2], rapid=True)
            elif cmd == "status":
                self.controller.get_status()
            elif cmd == "stop":
                self.controller.emergency_stop()
            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")


        except Exception as e:
            print(e)
            self.controller.disconnect()
            print("Goodbye!")
