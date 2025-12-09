# Chinese Chess Recognition

import AdjustCameraLocation as ad
import cv2, os, operator
import numpy as np
import onnxruntime as ort   # <- dùng ONNXRuntime
import torch

ip = ad.ip
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
        'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
pieceTypeList_with_grid = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
label_type = pieceTypeList_with_grid
dic = {'b_jiang':'Black General', 'b_ju':'Black Rook', 'b_ma':'Black Horse', 'b_pao':'Black Cannon', 'b_shi':'Black Guard', 'b_xiang':'Black Elephant', 'b_zu':'Black Soldier',
        'r_bing':'Red Soldier', 'r_ju':'Red Rook', 'r_ma':'Red Horse', 'r_pao':'Red Cannon', 'r_shi':'Red Guard', 'r_shuai':'Red General', 'r_xiang':'Red Elephant'}

isRed = True

def initialization():
    global GRID_WIDTH_HORI, GRID_WIDTH_VERTI, begin_point, cap, db, cursor
    global step, legal_move, model, target_size, isRed, input_name  # thêm input_name

    step = 0           # For recording steps
    legal_move = True  # To decide if the move is legal or illegal
    isRed = True       # To decide which team moves now
    target_size = (56, 56)  # CNN input image size

    # ---------------- ONNX model ----------------
    # Đổi đường dẫn dưới đây thành file .onnx của bạn
    model = ort.InferenceSession(
        "./chinese_chess_model_int8.onnx",   # ví dụ: đường dẫn model ONNX
        providers=["CPUExecutionProvider"]
    )
    input_name = model.get_inputs()[0].name
    # -------------------------------------------

    # Initialize grid width (giữ nguyên phần này)
    frame0 = cv2.imread('./Test_Image/Step 0.png', 0)
    img_circle = cv2.HoughCircles(
        frame0,
        cv2.HOUGH_GRADIENT,
        1,
        40,
        param1=100,
        param2=20,
        minRadius=18,
        maxRadius=22
    )[0]
    begin_point = img_circle[np.sum(img_circle, axis=1).tolist().index(min(np.sum(img_circle, axis=1).tolist()))]
    end_point = img_circle[np.sum(img_circle, axis=1).tolist().index(max(np.sum(img_circle, axis=1).tolist()))]
    GRID_WIDTH_HORI = (end_point[0] - begin_point[0]) / 8
    GRID_WIDTH_VERTI = (end_point[1] - begin_point[1]) / 9
    print('Recognition Initialized.\n')


def piece_prediction(model, img, target_size, top_n=3):
    # Chuẩn hoá ảnh giống như lúc training
    x = cv2.resize(img, target_size)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # shape (1, 56, 56, 3) nếu training kiểu channels_last

    # Nếu model ONNX của bạn dùng channels_first (NCHW),
    # thì cần đổi lại:
    # x = np.transpose(x, (0, 3, 1, 2))  # (1, 3, 56, 56)

    # Chạy ONNX
    global input_name
    preds = model.run(None, {input_name: x})[0]  # preds shape: (1, num_classes)

    idx = int(np.argmax(preds, axis=1)[0])
    return label_type[idx]


def save_path(beginPoint, endPoint, piece):
    global legal_move	# For indicating error movement
    begin = (np.around(abs(beginPoint[0]-begin_point[0])/GRID_WIDTH_HORI), np.around(abs(beginPoint[1]-begin_point[1])/GRID_WIDTH_VERTI))
    #print(beginPoint, endPoint)
    end = begin
    updown = np.around(abs(beginPoint[1]-endPoint[1])/GRID_WIDTH_VERTI)
    leftright = np.around(abs(beginPoint[0]-endPoint[0])/GRID_WIDTH_HORI)
    #print(updown, leftright)
    predict_category = piece_prediction(model, piece, target_size)
    variety = predict_category.split('_')[-1]
    color = predict_category.split('_')[0]

    # Print the path
    if beginPoint[1] - endPoint[1] > 0:
        end = (end[0], end[1] - updown)
    else:
        end = (end[0], end[1] + updown)

    if beginPoint[0] - endPoint[0] > 0:
        end = (end[0] - leftright, end[1])
    else:
        end = (end[0] + leftright, end[1])

    if predict_category == 'grid':
        return None

    print('{} moved from point {} to point {}'.format(dic[predict_category],
                                                      (float(begin[0]), float(begin[1])),
                                                      (float(end[0]), float(end[1]))))

    # Using chinese chess rules to reduce error movement
    if variety in ['ma']:
        if not (updown == 1 and leftright == 2) and not (updown == 2 and leftright == 1):
            legal_move = False
    elif variety in ['xiang']:
        if not (updown == 2 and leftright == 2) and not (updown == 2 and leftright == 2):
            legal_move = False
    elif variety in ['shi']:
        if not (updown == 1 and leftright == 1) and not (updown == 1 and leftright == 1):
            legal_move = False
    elif variety in ['jiang', 'shuai']:
        if not (updown == 1 and leftright == 0) and not (updown == 0 and leftright == 1):
            legal_move = False
    elif variety in ['ju', 'pao']:
        if updown != 0 and leftright != 0:
            legal_move = False
    elif variety in ['bing']:
        if begin[1] < end[1] or (begin[1] >= 5.0 and begin[0] != end[0]) or (begin[1]-end[1] > 1):
            legal_move = False
    elif variety in ['zu']:
        if begin[1] > end[1] or (begin[1] <= 4.0 and begin[0] != end[0]) or (end[1]-begin[1] > 1):
            legal_move = False

    if isRed:
        if color == 'b':
            legal_move = False
            print('It''s red team''s turn to move')
    else:
        if color == 'r':
            legal_move = False
            print('It''s black team''s turn to move')

    if not legal_move:
        cv2.imwrite('./pieces/%d.png' % np.random.randint(10000), piece)
    text = str(int(begin[0])) + str(int(begin[1])) + str(int(end[0])) + str(int(end[1]))
    return text, predict_category


def find_point(point, pointset):
    flag = False
    point_finetune = []
    for i in pointset:
        #point is (y, x)
        v1 = np.array([i[1], i[0]])
        v2 = np.array(point)
        d = np.linalg.norm(v1 - v2)
        if d < 25:
            flag = True
            point_finetune = i
            break
    return flag, point_finetune


def calculate_trace(pre_img, cur_img, x, y, w, h):
    # Input loca = [x, y, w, h], return all circle center inside the rectangular to pointSet
    pointSet = []
    beginPoint = []
    endPoint = []

    # --- 1. Cắt ROI để giảm diện tích HoughCircles phải xử lý ---
    pre_roi = pre_img[y:y + h, x:x + w]
    cur_roi = cur_img[y:y + h, x:x + w]

    pre_img_gray = cv2.cvtColor(pre_roi, cv2.COLOR_BGR2GRAY)
    cur_img_gray = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)

    pre_img_circle = cv2.HoughCircles(
        pre_img_gray,
        cv2.HOUGH_GRADIENT,
        1,
        40,
        param1=100,
        param2=20,
        minRadius=18,
        maxRadius=18
    )
    cur_img_circle = cv2.HoughCircles(
        cur_img_gray,
        cv2.HOUGH_GRADIENT,
        1,
        40,
        param1=100,
        param2=20,
        minRadius=18,
        maxRadius=18
    )

    # Nếu không tìm được vòng tròn nào thì trả về rỗng
    if pre_img_circle is None or cur_img_circle is None:
        return [], [], []

    pre_img_circle = pre_img_circle[0]
    cur_img_circle = cur_img_circle[0]

    # --- 2. Dịch toạ độ circle từ hệ ROI về hệ toàn ảnh ---
    # HoughCircles trên ROI cho center (x_roi, y_roi),
    # ta cộng offset (x, y) để ra (x_global, y_global)
    for c in pre_img_circle:
        c[0] += x
        c[1] += y
    for c in cur_img_circle:
        c[0] += x
        c[1] += y

    # --- 3. Xây pointSet như cũ (toạ độ global) ---
    for j in range(int(np.around(h / GRID_WIDTH_VERTI))):
        for i in range(int(np.around(w / GRID_WIDTH_HORI))):
            pointSet.append([
                y + (j + 0.5) * GRID_WIDTH_VERTI,
                x + (i + 0.5) * GRID_WIDTH_HORI
            ])

    # --- 4. Tìm beginPoint & endPoint như logic gốc ---
    for p in pointSet:
        if len(beginPoint) > 0 and len(endPoint) > 0:
            break

        flag1, p1 = find_point(p, pre_img_circle)
        flag2, p2 = find_point(p, cur_img_circle)

        # Trường hợp số quân trước > sau 1: có ăn quân
        if len(pre_img_circle) - len(cur_img_circle) == 1:
            if flag1 is True and flag2 is False:
                # có quân ở ảnh trước, không có ở ảnh sau -> điểm xuất phát
                beginPoint = p1
            elif flag1 is True and flag2 is True:
                # cả hai đều có quân -> kiểm tra loại quân có đổi không
                pre_piece = pre_img[
                    int(p1[1] - p1[2]):int(p1[1] + p1[2]),
                    int(p1[0] - p1[2]):int(p1[0] + p1[2])
                ]
                cur_piece = cur_img[
                    int(p2[1] - p2[2]):int(p2[1] + p2[2]),
                    int(p2[0] - p2[2]):int(p2[0] + p2[2])
                ]
                if piece_prediction(model, pre_piece, target_size) != \
                   piece_prediction(model, cur_piece, target_size):
                    endPoint = p2

        # Trường hợp số quân không đổi: chỉ di chuyển
        elif len(pre_img_circle) == len(cur_img_circle):
            if flag1 is True and flag2 is False:
                beginPoint = p1
            elif flag1 is False and flag2 is True:
                endPoint = p2

    # --- 5. Cắt ảnh quân cờ tại beginPoint (vẫn dùng ảnh gốc) ---
    if len(beginPoint) > 0 and len(endPoint) > 0:
        piece = pre_img[
            int(beginPoint[1] - beginPoint[2]):int(beginPoint[1] + beginPoint[2]),
            int(beginPoint[0] - beginPoint[2]):int(beginPoint[0] + beginPoint[2])
        ]
    else:
        return [], [], []

    return beginPoint, endPoint, piece


def change_detection(previous_step, current_step, visual=False):
    # Resize xuống 240x240
    small_prev = cv2.resize(previous_step, (240, 240))
    small_cur  = cv2.resize(current_step,  (240, 240))

    prev_gray = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    cur_gray  = cv2.cvtColor(small_cur,  cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(cur_gray, prev_gray)
    frame_diff = cv2.medianBlur(frame_diff, 3)
    _, frame_diff = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_OTSU)
    frame_diff = cv2.medianBlur(frame_diff, 3)

    x, y, w, h = cv2.boundingRect(frame_diff)

    # Map lại về toạ độ ảnh 480x480
    scale_x = previous_step.shape[1] / 240.0
    scale_y = previous_step.shape[0] / 240.0
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)

    if visual:
        vis = current_step.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.imshow('diff', vis)
        cv2.waitKey(20)

    return x, y, w, h


def compare(img1, img2, x, y, w, h):
    subset = []
    r = 19
    dict1 = {}
    dict2 = {}
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_circle = cv2.HoughCircles(img1_gray, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=20, minRadius=18, maxRadius=18)[0]
    for i in img1_circle:
        if x<i[0]<x+w and y<i[1]<y+h:
            subset.append(i)
    for point in subset:
        coordinate = ((point[0] - begin_point[0]) / GRID_WIDTH_HORI, (point[1] - begin_point[1]) / GRID_WIDTH_VERTI)
        piece1 = img1[int(point[1] - r):int(point[1] + r), int(point[0] - r):int(point[0] + r)]
        piece2 = img2[int(point[1] - r):int(point[1] + r), int(point[0] - r):int(point[0] + r)]
        cat1 = piece_prediction(model, piece1, target_size)
        cat2 = piece_prediction(model, piece2, target_size)
        dict1[(coordinate[0], coordinate[1])] = cat1
        dict2[(coordinate[0], coordinate[1])] = cat2
    return operator.eq(dict1, dict2)


def pieces_change_detection(current_step):
    global legal_move
    previous_step = cv2.imread('./Test_Image/Step %d.png' % step)
    x, y, w, h = change_detection(previous_step, current_step, False)
    if w * h < 50*50 or x == 0 or y == 0 or x+w == 480 or y+h == 480:	#棋子没有移动
        return 0
    else:
        beginPoint, endPoint, piece = calculate_trace(previous_step, current_step, x, y, w, h)
        if len(beginPoint) > 0 and len(endPoint) > 0 and len(piece)>0:
            text, predict_category = save_path(beginPoint, endPoint, piece)

        else:
            return 0

        if legal_move:
            # cv2.imwrite('./Test_Image/Step %d.png' % (step + 1), current_step)
            return 1
        else:
            print('Illegal move detected – skip rollback (debug mode).')
            legal_move = True
            return 0


if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./Sources/test.avi')

    if not cap.isOpened():
        exit('Camera is not open.')

    print("Camera mở rồi.")
    print("Đặt bàn cờ vào trong KHUNG VUÔNG.")
    print("Bấm phím 'd' để CHỤP Step 0, 'q' để thoát.")

    step0 = None
    os.makedirs('./Test_Image', exist_ok=True)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Không đọc được frame từ camera.")
            break

        # Cắt 480x480 góc trên trái
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

        cv2.imshow("Canh Khung Ban Co", current_frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('d'):
            # Lưu frame hiện tại làm Step 0
            step0 = current_frame.copy()
            cv2.imwrite('./Test_Image/Step 0.png', step0)
            print("Đã chụp Step 0, shape =", step0.shape)
            break

        elif key == ord('q'):
            break

    if step0 is None:
        cap.release()
        cv2.destroyAllWindows()
        exit("Không có Step 0 để khởi tạo.")

    print('Camera Initialized.')
    previous_frame = step0.copy()

    initialization()

    # --- Vòng lặp chính phía sau giữ như cũ ---
    frame_count = 0

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        current_frame = current_frame[0:480, 0:480]
        frame_count += 1
        if frame_count % 5 == 0:
            x, y, w, h = change_detection(current_frame, previous_frame)
            if x == 0 and y == 0 and w == 480 and h == 480:
                num = pieces_change_detection(current_frame)
                if num == 1:
                    step += 1
                    isRed = bool(1 - isRed)

            previous_frame = current_frame.copy()

        else:
            continue

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
        cv2.imshow('', current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
