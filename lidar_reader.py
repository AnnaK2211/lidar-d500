import csv
import struct
from datetime import datetime
from pathlib import Path
import serial

DATA_DIR = Path.home() / 'lidar_d500' / 'data'
PORT = '/dev/ttyUSB0'
BAUDRATE = 230400


def read_lidar(port=PORT, n_scans=20):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    file = DATA_DIR / f'scan_{datetime.now():%Y%m%d_%H%M%S}.csv'

    scan = -1
    previous_angle = None

    print('Reading LiDAR...')

    with serial.Serial(port, BAUDRATE, timeout=1) as ser:
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scan', 'angle_deg', 'dist_mm', 'intensity'
            ])

            while scan < n_scans:
                if ser.read(1) != b'\x54':
                    continue

                packet = ser.read(46)

                if len(packet) != 46 or packet[0] != 0x2C:
                    continue

                start = struct.unpack_from('<H', packet, 3)[0] / 100
                end = struct.unpack_from('<H', packet, 41)[0] / 100

                if previous_angle is not None and start < previous_angle:
                    scan += 1

                previous_angle = start

                if scan < 0 or scan >= n_scans:
                    continue

                step = ((end - start) % 360) / 11

                for i in range(12):
                    distance, intensity = struct.unpack_from(
                        '<HB', packet, 5 + 3 * i
                    )

                    if distance > 0:
                        angle = (start + step * i) % 360
                        writer.writerow([
                            scan, angle, distance, intensity
                        ])

    print('Saved:', file)
    return file


if __name__ == '__main__':
    read_lidar()