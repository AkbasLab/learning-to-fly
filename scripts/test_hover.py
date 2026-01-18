#!/usr/bin/env python3
"""
Safe hover test for PPO controller.
Hovers at low altitude for a few seconds, then lands.
"""

import time
import argparse
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# Default URI - change if needed
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E7E7'

def test_hover(uri: str, height: float = 0.3, duration: float = 3.0):
    """
    Perform a simple hover test.
    
    Args:
        uri: Crazyflie radio URI
        height: Hover height in meters (default 0.3m - very low)
        duration: How long to hover in seconds
    """
    print(f"Initializing drivers...")
    cflib.crtp.init_drivers()
    
    print(f"Connecting to {uri}...")
    
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Connected!")
        print(f"Starting hover test: {height}m for {duration}s")
        print("Press Ctrl+C to abort!")
        
        try:
            with MotionCommander(scf, default_height=height) as mc:
                print(f"Hovering at {height}m...")
                time.sleep(duration)
                print("Landing...")
            print("Test complete!")
        except KeyboardInterrupt:
            print("\nAborted! Landing...")
        except Exception as e:
            print(f"Error: {e}")
            raise

def scan_for_crazyflies():
    """Scan for available Crazyflies."""
    print("Scanning for Crazyflies...")
    cflib.crtp.init_drivers()
    available = cflib.crtp.scan_interfaces()
    
    if available:
        print("Found Crazyflies:")
        for i, uri in enumerate(available):
            print(f"  [{i}] {uri[0]}")
        return available
    else:
        print("No Crazyflies found. Make sure:")
        print("  - Crazyradio PA is connected")
        print("  - Crazyflie is powered on")
        print("  - Correct radio channel (default: 80)")
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Safe hover test for PPO controller')
    parser.add_argument('--uri', default=DEFAULT_URI, help='Crazyflie URI')
    parser.add_argument('--height', type=float, default=0.3, help='Hover height in meters')
    parser.add_argument('--duration', type=float, default=3.0, help='Hover duration in seconds')
    parser.add_argument('--scan', action='store_true', help='Scan for Crazyflies')
    
    args = parser.parse_args()
    
    if args.scan:
        available = scan_for_crazyflies()
        if available:
            print(f"\nTo test, run:")
            print(f"  python {__file__} --uri {available[0][0]}")
    else:
        print("=" * 50)
        print("PPO CONTROLLER HOVER TEST")
        print("=" * 50)
        print(f"Height: {args.height}m")
        print(f"Duration: {args.duration}s")
        print()
        print("⚠️  SAFETY WARNING ⚠️")
        print("- Clear the area around the drone")
        print("- Be ready to power off if unstable")
        print("- Start with drone on flat surface")
        print()
        
        input("Press ENTER to start (Ctrl+C to abort)...")
        test_hover(args.uri, args.height, args.duration)
