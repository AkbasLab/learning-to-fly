import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.crtpstack import CRTPPort
from cflib.crazyflie.commander import SET_SETPOINT_CHANNEL, META_COMMAND_CHANNEL, TYPE_HOVER 
import time
import struct
import argparse
import threading
import json
import os
import math
import sys

# Global state for packet logging
_packet_log = []
_packet_seq = 0
_log_rate_limit_hz = 10.0
_last_log_time = 0.0
_dump_file = None
_dry_run = False

def validate_float(value, name, min_val=-1e6, max_val=1e6):
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} is NaN or Inf: {value}")
    if value < min_val or value > max_val:
        raise ValueError(f"{name} out of range [{min_val}, {max_val}]: {value}")
    return value

def validate_packet_size(data, expected_size, packet_type):
    if len(data) != expected_size:
        raise ValueError(f"{packet_type} packet size mismatch: expected {expected_size}, got {len(data)}")

def log_packet(pk, packet_type, fields=None):
    global _packet_seq, _last_log_time, _dump_file
    now = time.time()
    _packet_seq += 1
    
    if _log_rate_limit_hz > 0 and (now - _last_log_time) < (1.0 / _log_rate_limit_hz):
        return
    _last_log_time = now
    
    record = {
        "seq": _packet_seq,
        "ts": now,
        "type": packet_type,
        "port": pk.port,
        "channel": pk.channel,
        "size": len(pk.data),
        "hex": pk.data.hex(),
        "fields": fields or {}
    }
    _packet_log.append(record)
    
    if _dump_file:
        _dump_file.write(json.dumps(record) + "\n")
        _dump_file.flush()
    
    print(f"[PKT {_packet_seq:06d}] {packet_type} port={pk.port} ch={pk.channel} hex={pk.data.hex()} fields={fields}")

def send_hover_packet(cf, height, vx=0, vy=0, yawrate=0):
    global _dry_run
    
    validate_float(height, "height", -10.0, 10.0)
    validate_float(vx, "vx", -10.0, 10.0)
    validate_float(vy, "vy", -10.0, 10.0)
    validate_float(yawrate, "yawrate", -10.0, 10.0)
    
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = SET_SETPOINT_CHANNEL
    pk.data = struct.pack('<Bffff', TYPE_HOVER, vx, vy, yawrate, height)
    
    validate_packet_size(pk.data, 17, "HOVER")
    
    log_packet(pk, "HOVER", {"type": TYPE_HOVER, "vx": vx, "vy": vy, "yawrate": yawrate, "height": height})
    
    if not _dry_run:
        cf.send_packet(pk)

def send_learned_policy_packet(cf):
    global _dry_run
    
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = META_COMMAND_CHANNEL
    pk.data = struct.pack('<B', 1)
    
    validate_packet_size(pk.data, 1, "LEARNED_POLICY")
    
    log_packet(pk, "LEARNED_POLICY", {"trigger": 1})
    
    if not _dry_run:
        cf.send_packet(pk)

def get_firmware_identity(cf):
    try:
        algo = cf.param.get_value("rltv.algo")
        git_hash = cf.param.get_value("rltv.hash")
        return {"algo": algo, "git_hash": git_hash}
    except KeyError:
        return None

def verify_firmware(cf, expected_algo=None):
    identity = get_firmware_identity(cf)
    if identity is None:
        print("WARNING: Firmware does not expose rltv.algo/rltv.hash params (old firmware?)")
        return False
    print(f"Firmware identity: algo={identity['algo']} git_hash={identity['git_hash']}")
    if expected_algo and identity['algo'] != expected_algo:
        print(f"ERROR: Expected algo={expected_algo}, got {identity['algo']}")
        return False
    return True

def mode_hover_original(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        prev = current
        send_hover_packet(cf, args.height)

def mode_hover_learned(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.wn", 1)
    set_param(cf, "rlt.motor_warmup", 1, optional=True)
    set_param(cf, "rlt.target_z", args.height)
    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        prev = current
        send_learned_policy_packet(cf)

def set_param(cf, name, target, optional=False):
    try:
        print(f"Parameter {name} was {cf.param.get_value(name)}, setting to {target}")
        while abs(float(cf.param.get_value(name)) - float(target)) > 1e-5:
            cf.param.set_value(name, target)
            time.sleep(0.1)
        print(f"Parameter {name} is {cf.param.get_value(name)} now")
    except KeyError:
        if optional:
            print(f"Parameter {name} not found (optional, skipping)")
        else:
            raise


def mode_trajectory_tracking(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command  (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.motor_warmup", 0, optional=True)
    set_param(cf, "rlt.wn", 4)
    set_param(cf, "rlt.fei", args.trajectory_interval)
    set_param(cf, "rlt.fes", args.trajectory_scale)
    set_param(cf, "rlt.target_z", args.height)

    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    start_time = time.time()
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        now = time.time()
        if current - prev > 0.1:
            start_time = now
        prev = current

        if now - start_time < args.transition_timeout:
            send_hover_packet(cf, args.height)
        else:
            send_learned_policy_packet(cf)

def mode_takeoff_and_switch(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command  (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.wn", 1)
    set_param(cf, "rlt.target_z", 0)
    set_param(cf, "rlt.motor_warmup", 0, optional=True)

    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    start_time = time.time()
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        now = time.time()
        if current - prev > 0.1:
            start_time = now
        prev = current

        if now - start_time < args.transition_timeout:
            send_hover_packet(cf, args.height)
        else:
            send_learned_policy_packet(cf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_uri = 'radio://0/80/2M/E7E7E7E7E7'
    parser.add_argument('--uri', default=default_uri)
    parser.add_argument('--height', default=0.2, type=float)
    parser.add_argument('--mode', default='hover_learned', choices=['hover_learned', 'hover_original', 'takeoff_and_switch', 'trajectory_tracking'])
    parser.add_argument('--trajectory-scale', default=1, type=float, help="Scale of the trajectory")
    parser.add_argument('--trajectory-interval', default=5.5, type=float, help="Interval of the trajectory")
    parser.add_argument('--transition-timeout', default=3, type=float, help="Time after takeoff with the original controller after which the learned controller is used for trajectory tracking")
    parser.add_argument('--dry-run', action='store_true', help="Do not transmit packets over radio")
    parser.add_argument('--dump', type=str, default=None, help="Path to JSONL dump file for packet logging")
    parser.add_argument('--log-rate', type=float, default=10.0, help="Rate limit for packet logging (Hz)")
    parser.add_argument('--verify-firmware', action='store_true', help="Verify firmware identity before sending")
    parser.add_argument('--expected-algo', type=str, default=None, choices=['TD3', 'PPO'], help="Expected algorithm for firmware verification")

    args = parser.parse_args()
    
    _dry_run = args.dry_run
    _log_rate_limit_hz = args.log_rate
    
    if args.dump:
        os.makedirs(os.path.dirname(args.dump) if os.path.dirname(args.dump) else ".", exist_ok=True)
        _dump_file = open(args.dump, 'w')
    
    if _dry_run:
        print("=== DRY RUN MODE - No packets will be transmitted ===")
    
    uri = uri_helper.uri_from_env(default=default_uri)
    cflib.crtp.init_drivers()

    if _dry_run:
        print("Dry run: skipping Crazyflie connection")
        print("Simulating packet sends...")
        class FakeCF:
            def send_packet(self, pk): pass
            class param:
                @staticmethod
                def get_value(name): return "N/A"
                @staticmethod
                def set_value(name, val): pass
        fake_cf = FakeCF()
        for i in range(10):
            send_learned_policy_packet(fake_cf)
            time.sleep(0.1)
        print(f"Dry run complete. {len(_packet_log)} packets logged.")
        if _dump_file:
            _dump_file.close()
        sys.exit(0)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='/tmp/cf_cache')) as scf:
        if args.verify_firmware:
            if not verify_firmware(scf.cf, args.expected_algo):
                print("Firmware verification failed. Aborting.")
                sys.exit(1)
        
        if args.mode == "hover_learned":
            mode_hover_learned(scf.cf, args)
        elif args.mode == "hover_original":
            mode_hover_original(scf.cf, args)
        elif args.mode == "takeoff_and_switch":
            mode_takeoff_and_switch(scf.cf, args)
        elif args.mode == "trajectory_tracking":
            mode_trajectory_tracking(scf.cf, args)
        else:
            print("Unknown mode")
        
        if _dump_file:
            _dump_file.close()
            print(f"Packet log written to {args.dump}")

