import time
import sys
import subprocess
import socket

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

def find_go2_interface():
    """Go2に接続可能なネットワークインターフェースを自動検出"""
    go2_ips = ['192.168.123.161', '192.168.123.1', '192.168.123.15']
    
    try:
        # 利用可能なインターフェースを取得
        result = subprocess.run(['ip', 'route', 'show'], capture_output=True, text=True)
        interfaces = []
        
        for line in result.stdout.split('\n'):
            if 'dev' in line:
                parts = line.split()
                if 'dev' in parts:
                    dev_idx = parts.index('dev')
                    if dev_idx + 1 < len(parts):
                        interface = parts[dev_idx + 1]
                        if interface not in interfaces:
                            interfaces.append(interface)
        
        print(f"Available interfaces: {interfaces}")
        
        # 各IPアドレスに対してping確認
        for ip in go2_ips:
            print(f"Testing connection to {ip}...")
            ping_result = subprocess.run(['ping', '-c', '2', '-W', '2', ip], 
                                       capture_output=True, text=True)
            if ping_result.returncode == 0:
                # 該当IPアドレスのルートを確認
                route_result = subprocess.run(['ip', 'route', 'get', ip], 
                                            capture_output=True, text=True)
                if route_result.returncode == 0:
                    for line in route_result.stdout.split('\n'):
                        if 'dev' in line:
                            parts = line.split()
                            if 'dev' in parts:
                                dev_idx = parts.index('dev')
                                if dev_idx + 1 < len(parts):
                                    interface = parts[dev_idx + 1]
                                    print(f"✓ Found Go2 at {ip} via interface: {interface}")
                                    return interface, ip
        
        return None, None
        
    except Exception as e:
        print(f"Error detecting interface: {e}")
        return None, None

def check_network_interfaces():
    """ネットワークインターフェースを確認"""
    try:
        print("=== Network Interface Information ===")
        
        # インターフェース一覧
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        print("Available network interfaces:")
        print(result.stdout)
        
        print("\n=== Routing Table ===")
        route_result = subprocess.run(['ip', 'route'], capture_output=True, text=True)
        print(route_result.stdout)
        
        # Go2の自動検出
        print("\n=== Go2 Detection ===")
        interface, ip = find_go2_interface()
        
        if interface and ip:
            print(f"✓ Go2 detected at {ip} via {interface}")
            return interface
        else:
            print("✗ Go2 not detected. Manual configuration required.")
            return None
            
    except Exception as e:
        print(f"Error checking network: {e}")
        return None

def main():
    print("Go2 Connection Test for Docker")
    print("=" * 40)
    
    # ネットワークインターフェースの確認
    detected_interface = check_network_interfaces()
    
    if len(sys.argv) > 1:
        interface = sys.argv[1]
        print(f"\nUsing specified interface: {interface}")
    elif detected_interface:
        interface = detected_interface
        print(f"\nUsing detected interface: {interface}")
    else:
        # Dockerでよく使われるデフォルトインターフェース
        interface = "eth0"  # Dockerのデフォルト
        print(f"\nUsing default Docker interface: {interface}")
    
    try:
        print(f"Initializing channel with interface: {interface}")
        ChannelFactoryInitialize(0, interface)
        print("✓ Channel initialization successful!")
        
        # 簡単な接続テスト
        time.sleep(1)

        import os
        # 環境変数の追加
        os.environ['GO2interface'] = 'interface'

        print(f"\nAdd Env value: {os.environ['GO2interface']}")

        print("Connection test completed.")
        
    except Exception as e:
        print(f"✗ Channel initialization failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Docker is running with --network host")
        print("2. Check if Go2 is powered on and connected")
        print("3. Verify Ethernet cable connection")
        print("4. Try different interface names:")
        print("   - eth0 (Docker default)")
        print("   - enp0s3, enp0s8 (VirtualBox)")
        print("   - ens33, ens34 (VMware)")

    print(f"\nDocker Usage examples:")
    print("1. Host network: docker run --network host -it image_name python networkTest.py")
    print("2. Specific interface: docker run --network host -it image_name python networkTest.py eth0")

if __name__ == '__main__':
    main()