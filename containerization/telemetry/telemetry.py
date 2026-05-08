import serial
import time

SERIAL_PORT = "/dev/ttyUSB1"
BAUDRATE = 115200

def convert(raw, direction):
    if not raw:
        return 0.0

    if direction in ("N", "S"):
        d = int(raw[:2])
        m = float(raw[2:])
    else:
        d = int(raw[:3])
        m = float(raw[3:])

    v = d + m / 60

    if direction in ("S", "W"):
        v *= -1

    return v

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

lat = 0
lon = 0
speed = 0
alt = 0
sat = 0

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if line.startswith("$GPRMC"):
        p = line.split(",")
        lat = convert(p[3], p[4])
        lon = convert(p[5], p[6])
        speed = float(p[7]) * 1.852 if p[7] else 0

        print(f"[GPS] {lat}, {lon} | {speed:.2f} km/h")

    elif line.startswith("$GPGGA"):
        p = line.split(",")
        alt = float(p[9]) if p[9] else 0
        sat = int(p[7]) if p[7] else 0

        print(f"[ALT] {alt} m | {sat} sats")