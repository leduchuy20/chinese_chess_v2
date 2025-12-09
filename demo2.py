import AdjustCameraLocation as ad
import cv2, os, time
import numpy as np
import onnxruntime as ort

# --- CẤU HÌNH ---
IS_RED_TURN = True  # Lượt đi hiện tại
GRID_ROWS = 10
GRID_COLS = 9
ROI_SIZE = 20  # Bán kính vùng cắt (crop 40x40 pixel quanh giao điểm)
INTENSITY_THRESHOLD = 30  # Ngưỡng thay đổi màu sắc trung bình để kích hoạt AI

# Mapping nhãn
label_type = ['b_jiang', 'b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
              'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
dic_name = {'b_jiang':'Black General', 'b_ju':'Black Rook', 'b_ma':'Black Horse', 'b_pao':'Black Cannon',
            'b_shi':'Black Guard', 'b_xiang':'Black Elephant', 'b_zu':'Black Soldier',
            'r_bing':'Red Soldier', 'r_ju':'Red Rook', 'r_ma':'Red Horse', 'r_pao':'Red Cannon',
            'r_shi':'Red Guard', 'r_shuai':'Red General', 'r_xiang':'Red Elephant'}

# --- BIẾN TOÀN CỤC ---
ort_session = None
input_name = None
grid_points = []  # Danh sách 90 toạ độ (x, y)
board_state = []  # Trạng thái hiện tại của 90 ô (Lưu tên quân cờ)
bg_means = []  # Lưu độ sáng trung bình của 90 ô để so sánh nhanh


def init_onnx():
    global ort_session, input_name
    model_path = 'chinese_chess_model.onnx'
    if not os.path.exists(model_path):
        exit(f"Lỗi: Không tìm thấy {model_path}")
    ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    print("AI Model loaded.")


def predict_piece(img_roi):
    # Pre-process giống lúc train
    img = cv2.resize(img_roi, (56, 56))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    preds = ort_session.run(None, {input_name: img})[0]
    idx = int(np.argmax(preds, axis=1)[0])
    return label_type[idx]


def init_grid(frame):
    global grid_points, board_state, bg_means

    # Dùng HoughCircles để tìm 4 góc hoặc tự tính toán như code cũ
    # Ở đây tôi giữ logic tính toán dựa trên 4 điểm biên của bạn nhưng làm gọn lại
    # Giả sử AdjustCameraLocation trả về begin (x,y) chuẩn

    # Tìm quân cờ để xác định biên chính xác hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=20, minRadius=15, maxRadius=25)

    if circles is None:
        exit("Không tìm thấy quân cờ để khởi tạo lưới!")

    circles = np.uint16(np.around(circles))[0]

    # Tìm min/max x, y từ các quân cờ tìm được để chia lưới
    min_x = np.min(circles[:, 0])
    max_x = np.max(circles[:, 0])
    min_y = np.min(circles[:, 1])
    max_y = np.max(circles[:, 1])

    w_step = (max_x - min_x) / 8
    h_step = (max_y - min_y) / 9

    grid_points = []
    board_state = []
    bg_means = []

    print("Đang khởi tạo trạng thái bàn cờ ban đầu...")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx = int(min_x + c * w_step)
            cy = int(min_y + r * h_step)
            grid_points.append((cx, cy))

            # Cắt ROI và nhận diện ngay từ đầu
            y1, y2 = max(0, cy - ROI_SIZE), min(480, cy + ROI_SIZE)
            x1, x2 = max(0, cx - ROI_SIZE), min(480, cx + ROI_SIZE)
            roi = frame[y1:y2, x1:x2]

            # 1. Lưu độ sáng trung bình
            mean_val = np.mean(roi)
            bg_means.append(mean_val)

            # 2. Nhận diện quân cờ
            label = predict_piece(roi)
            board_state.append(label)

    print(f"Đã khởi tạo {len(grid_points)} điểm lưới.")


def scan_board_changes(current_frame):
    global board_state, bg_means

    changes = []  # Lưu các thay đổi: (index, old_label, new_label)

    for i, (cx, cy) in enumerate(grid_points):
        # Cắt ROI
        y1, y2 = max(0, cy - ROI_SIZE), min(480, cy + ROI_SIZE)
        x1, x2 = max(0, cx - ROI_SIZE), min(480, cx + ROI_SIZE)
        roi = current_frame[y1:y2, x1:x2]

        # 1. FAST PASS: Kiểm tra độ sáng trung bình trước
        current_mean = np.mean(roi)
        diff = abs(current_mean - bg_means[i])

        # Nếu độ sáng thay đổi lớn hơn ngưỡng -> Có biến động (quân đi hoặc quân đến)
        if diff > INTENSITY_THRESHOLD:
            # 2. DEEP PASS: Chạy AI nhận diện lại
            new_label = predict_piece(roi)

            # Nếu nhãn thực sự thay đổi (VD: grid -> r_pao, hoặc r_pao -> grid)
            if new_label != board_state[i]:
                changes.append((i, board_state[i], new_label))

                # Cập nhật trạng thái mới
                board_state[i] = new_label
                bg_means[i] = current_mean  # Cập nhật độ sáng nền mới

    return changes


def process_move_logic(changes):
    global IS_RED_TURN

    if not changes:
        return

    # Phân tích các thay đổi để suy ra nước đi
    # Thông thường 1 nước đi sẽ có 2 thay đổi: 
    # 1. Điểm đi (Source): Có quân -> Grid (Trong)
    # 2. Điểm đến (Target): Grid/Quân bị ăn -> Quân mới

    src = None
    dst = None

    for idx, old, new in changes:
        r, c = idx // 9, idx % 9
        print(f"Change tại ({r},{c}): {old} -> {new}")

        if new == 'grid':
            src = (idx, old)  # Đây là nơi quân cờ rời đi
        else:
            dst = (idx, new)  # Đây là nơi quân cờ đi tới

    if src and dst:
        s_r, s_c = src[0] // 9, src[0] % 9
        d_r, d_c = dst[0] // 9, dst[0] % 9
        piece_name = dic_name.get(dst[1], dst[1])

        print(f"\n>>> PHÁT HIỆN NƯỚC ĐI: {piece_name}")
        print(f"    Từ: ({s_r}, {s_c}) -> Đến: ({d_r}, {d_c})")

        # Đổi lượt
        IS_RED_TURN = not IS_RED_TURN
        turn_str = "DO" if IS_RED_TURN else "DEN"
        print(f"    Lượt tiếp theo: {turn_str}\n")
        return True
    return False


# --- MAIN ---
if __name__ == '__main__':
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('./Sources/test.avi')

    # Bước 1: Canh chỉnh camera
    while True:
        ret, frame = cap.read()
        frame = frame[0:480, 0:480]
        cv2.rectangle(frame, ad.begin, (ad.begin[0] + 400, ad.begin[1] + 400), (0, 255, 0), 2)
        cv2.putText(frame, "Canh khung & Bam 's' de bat dau", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Setup', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            init_onnx()
            init_grid(frame)  # Khởi tạo lưới dựa trên hình ảnh tĩnh đầu tiên
            break

    cv2.destroyWindow('Setup')

    # Bước 2: Vòng lặp chính
    static_count = 0
    prev_score = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = frame[0:480, 0:480]
        display = frame.copy()

        # --- LOGIC PHÁT HIỆN TĨNH (MOTION DETECTION) ---
        # Chỉ khi nào tay rút ra (ảnh tĩnh) thì mới quét lưới
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if 'prev_gray' in locals():
            diff = cv2.absdiff(gray, prev_gray)
            score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])

            # Nếu ít biến động
            if score < 2000:
                static_count += 1
            else:
                static_count = 0

            # Nếu đã tĩnh đủ lâu (khoảng 1 giây = 20-30 frames)
            if static_count == 20:
                print("Frame tĩnh. Quét bàn cờ...")
                changes = scan_board_changes(frame)
                if process_move_logic(changes):
                    # Nếu có nước đi thành công, reset count để tránh detect lặp
                    static_count = 0

        prev_gray = gray

        # --- VẼ DEBUG ---
        # Vẽ các điểm lưới và trạng thái
        for i, (cx, cy) in enumerate(grid_points):
            color = (0, 255, 0)  # Màu xanh = Grid trống
            if board_state[i] != 'grid':
                color = (0, 0, 255)  # Màu đỏ = Có quân
            cv2.circle(display, (cx, cy), 3, color, -1)

        cv2.putText(display, f"Static: {static_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Smart Chess Tracker', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()