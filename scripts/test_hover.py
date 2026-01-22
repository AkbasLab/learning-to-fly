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

# Connection settings
CONNECTION_TIMEOUT = 5.0  # seconds to wait for connection
PACKET_LOSS_THRESHOLD = 10  # consecutive lost packets before disconnect

# Global state
_scf = None
_cf = None
_running = True
_hovering = False
_connected = False
_consecutive_failures = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C immediately"""
    global _running
    print("\n\n--- EMERGENCY STOP ---")
    _running = False
    safe_disconnect()
    sys.exit(0)

def safe_disconnect():
    """Safely disconnect from the Crazyflie"""
    global _cf, _scf, _connected
    try:
        if _cf and _connected:
            try:
                _cf.commander.send_stop_setpoint()
            except:
                pass
        if _cf and _cf.link:
            try:
                _cf.close_link()
            except:
                pass
    except:
        pass
    _connected = False

signal.signal(signal.SIGINT, signal_handler)

def test_hover_deadman(uri: str, height: float = 0.3):
    """
    Dead-man's switch hover test.
    Hover only while SPACE is held (or any key on Windows).
    """
    global _scf, _cf, _running, _hovering, _connected, _consecutive_failures
    
    print(f"Initializing drivers...")
    cflib.crtp.init_drivers()
    
    print(f"Connecting to {uri} (timeout: {CONNECTION_TIMEOUT}s)...")
    
    # Set up terminal for raw input on Linux
    if PLATFORM == "linux":
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    try:
        # Create Crazyflie instance with callbacks
        cf = Crazyflie(rw_cache='./cache')
        _cf = cf
        
        # Connection state tracking
        connection_complete = threading.Event()
        connection_failed = threading.Event()
        
        def on_connected(link_uri):
            global _connected
            _connected = True
            connection_complete.set()
        
        def on_connection_failed(link_uri, msg):
            print(f"\nConnection failed: {msg}")
            connection_failed.set()
        
        def on_disconnected(link_uri):
            global _connected, _running
            if _connected:
                print("\n--- DRONE DISCONNECTED ---")
            _connected = False
            _running = False
        
        def on_link_quality(quality):
            global _consecutive_failures
            if quality < 10:  # Very low link quality
                _consecutive_failures += 1
            else:
                _consecutive_failures = 0
        
        # Register callbacks
        cf.connected.add_callback(on_connected)
        cf.connection_failed.add_callback(on_connection_failed)
        cf.disconnected.add_callback(on_disconnected)
        cf.link_quality_updated.add_callback(on_link_quality)
        
        # Open link
        cf.open_link(uri)
        
        # Wait for connection with timeout
        start_time = time.time()
        while not connection_complete.is_set() and not connection_failed.is_set():
            if time.time() - start_time > CONNECTION_TIMEOUT:
                print(f"\n--- CONNECTION TIMEOUT ({CONNECTION_TIMEOUT}s) ---")
                cf.close_link()
                return
            time.sleep(0.1)
        
        if connection_failed.is_set():
            cf.close_link()
            return
        
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
        
        while _running and _connected:
            # Check for packet loss / drone power off
            if _consecutive_failures >= PACKET_LOSS_THRESHOLD:
                print("\n--- LINK LOST (packet loss) ---")
                break
            
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
                try:
                    cf.commander.send_stop_setpoint()
                except:
                    break
            
            # Send commands based on state
            if _hovering:
                try:
                    cf.commander.send_hover_setpoint(0, 0, 0, height)
                except:
                    print("\n--- SEND FAILED ---")
                    break
            
            time.sleep(0.02)  # 50Hz loop
        
        # Final stop
        try:
            cf.commander.send_stop_setpoint()
        except:
            pass
        print("--- Test ended safely ---")
        
        # Close link gracefully
        cf.close_link()
            
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Restore terminal on Linux
        if PLATFORM == "linux":
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        safe_disconnect()


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
