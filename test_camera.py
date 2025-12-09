import cv2
import AdjustCameraLocation as ad

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit('Camera is not open.')

print("Nhấn 'q' để thoát.")

while True:
    ret, current_frame = cap.read()
    if not ret:
        print("Không đọc được frame từ webcam.")
        break

    # Hiển thị frame từ webcam
    cv2.imshow("Webcam CIFAR-10 Demo", current_frame)

    current_frame = current_frame[0:480, 0:480]

    # Vẽ khung như code gốc của bạn
    cv2.rectangle(
        current_frame,
        ad.begin,
        (ad.begin[0] + 400, ad.begin[1] + 400),
        (255, 255, 255),
        2
    )

    # Tính toạ độ khung
    x0, y0 = ad.begin
    x1, y1 = x0 + 400, y0 + 400

    # Tâm khung
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    # Vẽ đường dọc đi qua tâm
    cv2.line(current_frame, (cx, y0), (cx, y1), (255, 255, 255), 1)

    # Vẽ đường ngang đi qua tâm
    cv2.line(current_frame, (x0, cy), (x1, cy), (255, 255, 255), 1)

    cv2.imshow("Canh khung rồi bấm 's' để chụp Step 0", current_frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('s'):
        # Lưu frame hiện tại làm Step 0
        step0 = current_frame.copy()
        cv2.imwrite('./Test_Image/Step 0.png', step0)
        print("Đã chụp Step 0, shape =", step0.shape)
        break

    elif key == ord('q'):
        break

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()