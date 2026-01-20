#!/usr/bin/env python3
"""
Safe hover test for PPO controller.
Dead-man's switch: Only hovers while you hold a key.
"""

import time
import signal
import sys
import threading
import argparse
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Platform-specific key detection
try:
    import msvcrt  # Windows
    def key_pressed():
        return msvcrt.kbhit()
    def get_key():
        return msvcrt.getch()
    PLATFORM = "windows"
except ImportError:
    import select
    import tty
    import termios
    PLATFORM = "linux"
    def key_pressed():
        return select.select([sys.stdin], [], [], 0)[0]
    def get_key():
        return sys.stdin.read(1)

# Default URI - change if needed
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E7'

# Global state
_scf = None
_running = True
_hovering = False

def signal_handler(sig, frame):
    """Handle Ctrl+C immediately"""
    global _running
    print("\n\n---EMERGENCY STOP ---")
    _running = False
    if _scf and _scf.cf.link:
        _scf.cf.commander.send_stop_setpoint()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def test_hover_deadman(uri: str, height: float = 0.3):
    """
    Dead-man's switch hover test.
    Hover only while SPACE is held (or any key on Windows).
    """
    global _scf, _running, _hovering
    
    print(f"Initializing drivers...")
    cflib.crtp.init_drivers()
    
    print(f"Connecting to {uri}...")
    
    # Set up terminal for raw input on Linux
    if PLATFORM == "linux":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    try:
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            _scf = scf
            cf = scf.cf
            
            print("Connected!")
            print("")
            print("=" * 50)
            print("  DEAD-MAN'S SWITCH MODE")
            print("=" * 50)
            print("  HOLD SPACE = Hover")
            print("  RELEASE    = Stop motors")
            print("  Q or ESC   = Quit")
            print("=" * 50)
            print("")
            
            # Unlock commander
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            
            last_key_time = 0
            KEY_TIMEOUT = 0.15  # Stop if no key for 150ms
            
            while _running:
                if key_pressed():
                    key = get_key()
                    
                    # Decode if bytes (Windows)
                    if isinstance(key, bytes):
                        key = key.decode('utf-8', errors='ignore')
                    
                    # Quit on Q or ESC
                    if key.lower() == 'q' or key == '\x1b':
                        print("\nQuitting...")
                        break
                    
                    # Any other key = hover
                    if key == ' ' or key == '\r' or key == '\n':
                        last_key_time = time.time()
                        if not _hovering:
                            print("--- HOVERING... ---")
                            _hovering = True
                
                # Check if key is still being "held" (received recently)
                if _hovering and (time.time() - last_key_time) > KEY_TIMEOUT:
                    print("--- STOPPED ---")
                    _hovering = False
                    cf.commander.send_stop_setpoint()
                
                # Send commands based on state
                if _hovering:
                    cf.commander.send_hover_setpoint(0, 0, 0, height)
                
                time.sleep(0.02)  # 50Hz loop
            
            # Final stop
            cf.commander.send_stop_setpoint()
            print("--- Test ended safely ---")
            
    finally:
        # Restore terminal on Linux
        if PLATFORM == "linux":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if _scf and _scf.cf.link:
            _scf.cf.commander.send_stop_setpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dead-man switch hover test for PPO controller')
    parser.add_argument('--uri', default=DEFAULT_URI, help='Crazyflie URI')
    parser.add_argument('--height', type=float, default=0.3, help='Hover height in meters')
    parser.add_argument('--scan', action='store_true', help='Scan for Crazyflies')
    
    args = parser.parse_args()
    
    if args.scan:
        print("Scanning for Crazyflies...")
        cflib.crtp.init_drivers()
        available = cflib.crtp.scan_interfaces()
        if available:
            print("Found Crazyflies:")
            for i, uri in enumerate(available):
                print(f"  [{i}] {uri[0]}")
            print(f"\nTo test, run:")
            print(f"  python {__file__} --uri {available[0][0]}")
        else:
            print("No Crazyflies found.")
    else:
        print("=" * 50)
        print("PPO CONTROLLER - DEAD-MAN'S SWITCH TEST")
        print("=" * 50)
        print(f"Height: {args.height}m")
        print()
        print("--- SAFETY WARNING ---")
        print("- Hold SPACE or ENTER to hover")
        print("- RELEASE to immediately stop motors")
        print("- Press Q or ESC to quit")
        print()
        
        input("Press ENTER to connect and begin...")
        test_hover_deadman(args.uri, args.height)
