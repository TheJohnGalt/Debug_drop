import serial

PORT = "/dev/ttyUSB1"
BAUD = 9600

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)

    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line.startswith("$G"):
            print(line, flush=True)

def mock_telemetry():
    print("telemetry is runing")

if __name__ == "__main__":
    mock_telemetry()