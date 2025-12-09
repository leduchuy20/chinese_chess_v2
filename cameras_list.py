import cv2


def list_working_cameras():
    """
    Tests the ports and returns a list of working camera indices.
    """
    working_ports = []
    # Test up to a reasonable number of ports, e.g., 10
    for dev_port in range(10):
        camera = cv2.VideoCapture(dev_port)
        if camera.isOpened():
            is_reading, img = camera.read()
            if is_reading:
                print(f"Port {dev_port} is working.")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} is present but does not read images.")
            camera.release()  # Release the camera after testing
        else:
            # Once a port fails to open, you can assume no more ports exist in sequence
            # break
            pass

    return working_ports


if __name__ == '__main__':
    working_cameras = list_working_cameras()
    print(f"Found working cameras at indices: {working_cameras}")
