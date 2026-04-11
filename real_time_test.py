# Chinese Chess Recognition
import AdjustCameraLocation as ad
import cv2, os, pymysql, operator, copy
import numpy as np
import time
import threading
import queue as _q
from collections import Counter
from piece_detector import PieceDetector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ENFORCE_MOVE_RULES = True
MOVE_CAPTURE_DIR = './Move_Captures'
STABLE_DIFF_THRESHOLD = 1300
STABLE_SECONDS_REQUIRED = 0.3
STEP0_STABLE_DIFF_THRESHOLD = 1800
STEP0_STABLE_SECONDS_REQUIRED = 2.0
PROCESS_SIZE = 480
ROI_MOVE_STEP = 12
ROI_SIZE_STEP = 20
MOTION_BINARY_THRESHOLD = 15
DETECT_DURING_MOTION_EVERY = 4
ENABLE_DETECT_DURING_MOTION = False
BOARD_DETECT_CONF = 0.12
LABEL_SWITCH_CONF_MIN = 0.45
SOLDIER_ADVISER_SWITCH_CONF_MIN = 0.72


ip = ad.ip
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
        'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
pieceTypeList_with_grid = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'grid',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']
label_type = pieceTypeList_with_grid
dic = {'b_jiang':'Black King', 'b_ju':'Black Rook', 'b_ma':'Black Knight', 'b_pao':'Black Cannon', 'b_shi':'Black Guard', 'b_xiang':'Black Elephant', 'b_zu':'Black Pawn',
        'r_bing':'Red Soldier', 'r_ju':'Red Chariot', 'r_ma':'Red Horse', 'r_pao':'Red Cannon', 'r_shi':'Red Adviser', 'r_shuai':'Red General', 'r_xiang':'Red Minister'}

board_state_map = {}


class TemporalBoardVoter:
    """
    Lưu lịch sử N lần quét bàn cờ gần nhất.
    get_stable_state() trả về trạng thái được vote đa số trên từng ô lưới,
    giúp lọc nhiễu single-frame từ YOLO (nhầm lẫn r_bing/r_shi, b_zu/b_shi...).
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._history: list[dict] = []

    def push(self, label_map: dict) -> None:
        self._history.append(dict(label_map))
        if len(self._history) > self.window:
            self._history.pop(0)

    def clear(self) -> None:
        self._history.clear()

    def ready(self) -> bool:
        """True khi đã có ít nhất 2 mẫu để vote."""
        return len(self._history) >= 2

    def get_stable_state(self) -> dict:
        """Majority-vote nhãn theo từng ô (bx, by). Ô có đa số 'grid' bị bỏ qua."""
        if not self._history:
            return {}
        all_keys: set = set()
        for m in self._history:
            all_keys.update(m.keys())
        threshold = max(1, len(self._history) // 2 + 1)
        result: dict = {}
        for key in all_keys:
            labels = [m.get(key, 'grid') for m in self._history]
            best_label, best_count = Counter(labels).most_common(1)[0]
            if best_label != 'grid' and best_count >= threshold:
                result[key] = best_label
        return result


board_voter = TemporalBoardVoter(window=5)

# --- Background detection thread ---
_detect_lock = threading.Lock()
_detect_active = False          # True khi inference đang chạy trong thread
_detect_result_q = _q.Queue(maxsize=1)  # Nhận kết quả 0/1 từ thread


def _detection_worker(frame: np.ndarray) -> None:
    """Chạy PiecesChangeDetection trong background thread để không block display."""
    global _detect_active
    result = PiecesChangeDetection(frame)
    try:
        _detect_result_q.put_nowait(result)
    except _q.Full:
        pass
    with _detect_lock:
        _detect_active = False


def normalize_roi(x, y, side, frame_w, frame_h):
    side = int(max(200, min(side, min(frame_w, frame_h))))
    x = int(max(0, min(x, frame_w - side)))
    y = int(max(0, min(y, frame_h - side)))
    return x, y, side


def extract_process_frame(frame, roi):
    x, y, side = roi
    patch = frame[y:y + side, x:x + side]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (PROCESS_SIZE, PROCESS_SIZE), interpolation=cv2.INTER_AREA)


def Initialization():
    global GRID_WIDTH_HORI, GRID_WIDTH_VERTI, begin_point, cap, db, cursor, step, legal_move, model, target_size, isRed

    step = 0		# For recording steps
    legal_move = True	# To decide if the move is legal or illegal
    isRed = True		# To decide which team moves now
    target_size = (640, 640)
    model = PieceDetector(weights_path='./weights.pt', imgsz=640, conf=0.05)
    os.makedirs(MOVE_CAPTURE_DIR, exist_ok=True)
    for name in os.listdir(MOVE_CAPTURE_DIR):
        path = os.path.join(MOVE_CAPTURE_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    # Initialize mysql
    # db = pymysql.connect("localhost", "root", "root", "chess")
    # cursor = db.cursor()
    # cursor.execute("DROP TABLE IF EXISTS chess")
    # sql1 = """CREATE TABLE chess (
    # 			Id INT AUTO_INCREMENT,
    # 			STEP CHAR(4) NOT NULL,
    # 			PRIMARY KEY(Id)
    # 		)"""
    # cursor.execute(sql1)
    # print('SQL Initialized.')

    # Initialize grid width
    frame0 = cv2.imread('./Test_Image/Step 0.png', 0)
    img_circle = cv2.HoughCircles(frame0,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=22)[0]
    begin_point = img_circle[np.sum(img_circle, axis=1).tolist().index(min(np.sum(img_circle, axis=1).tolist()))]
    end_point = img_circle[np.sum(img_circle, axis=1).tolist().index(max(np.sum(img_circle, axis=1).tolist()))]
    GRID_WIDTH_HORI = (end_point[0] - begin_point[0])/8
    GRID_WIDTH_VERTI = (end_point[1] - begin_point[1])/9
    init_board_state('./Test_Image/Step 0.png')
    print('Recognition Initialized.\n')

# def PiecePrediction(model, img, target_size, top_n=3):
# 	x = cv2.resize(img, target_size)
# 	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# 	x = x / 255
# 	x = np.expand_dims(x, axis=0)
# 	preds = model.predict_classes(x)
# 	return label_type[int(preds)]

# def PiecePrediction(model, piece, target_size):
#     img = cv2.resize(piece, target_size)
#     img = img.astype('float32') / 255.0

#     # tuỳ model training là input dạng gì, project này thường dùng (H, W, 1)
#     # nên ta reshape về (1, H, W, 1)
#     if img.ndim == 2:  # grayscale
#         x = img.reshape((1, img.shape[0], img.shape[1], 1))
#     else:  # lỡ là 3 kênh
#         x = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

#     # Dự đoán xác suất / logits
#     preds = model.predict(x)        # shape: (1, num_classes)

#     # Tự lấy class lớn nhất thay cho predict_classes
#     predict_category = int(np.argmax(preds, axis=1)[0])

#     return predict_category

# def PiecePrediction(model, img, target_size, top_n=3):
#     x = cv2.resize(img, target_size)
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#     x = x.astype('float32') / 255.0
#     x = np.expand_dims(x, axis=0)

#     preds = model.predict(x)
#     idx = int(np.argmax(preds, axis=1)[0])
#     return label_type[idx]

def PiecePrediction(model, img, target_size, top_n=3):
    return model.predict_piece(img, default_label='grid')


def crop_with_margin(img, circle, margin_scale=1.6):
    cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
    m = max(2, int(r * margin_scale))
    h_img, w_img = img.shape[:2]
    x1, x2 = max(0, cx - m), min(w_img, cx + m)
    y1, y2 = max(0, cy - m), min(h_img, cy + m)
    return img[y1:y2, x1:x2]


def point_to_board_xy(point):
    bx = int(np.around((point[0] - begin_point[0]) / GRID_WIDTH_HORI))
    by = int(np.around((point[1] - begin_point[1]) / GRID_WIDTH_VERTI))
    return bx, by


def board_xy_to_pixel(cell):
    return (
        int(np.around(begin_point[0] + cell[0] * GRID_WIDTH_HORI)),
        int(np.around(begin_point[1] + cell[1] * GRID_WIDTH_VERTI)),
    )


def is_in_palace(color, bx, by):
    if not (3 <= bx <= 5):
        return False
    if color == 'r':
        return 7 <= by <= 9
    if color == 'b':
        return 0 <= by <= 2
    return False


def correct_label_by_position(label, bx, by, score):
    # Advisers must stay inside the palace in Xiangqi.
    # If detected outside palace, remap to soldier to reduce r_bing -> r_shi confusion.
    if label == 'r_shi' and (not is_in_palace('r', bx, by)):
        return 'r_bing'
    if label == 'b_shi' and (not is_in_palace('b', bx, by)):
        return 'b_zu'
    return label


def detect_board_map(frame, prev_map=None, return_conf=False):
    detections = model.predict_detections(frame, conf=BOARD_DETECT_CONF)
    board_map = {}

    for det in detections:
        label = det['label']
        if label == 'grid' or label not in dic:
            continue

        bx, by = point_to_board_xy((det['cx'], det['cy']))
        if not (0 <= bx <= 8 and 0 <= by <= 9):
            continue

        key = (bx, by)
        score = det['conf']
        label = correct_label_by_position(label, bx, by, score)
        if key not in board_map or score > board_map[key][1]:
            board_map[key] = (label, score)

    if prev_map:
        stabilized = {}
        stabilized_conf = {}
        for key, (label, score) in board_map.items():
            prev_label = prev_map.get(key)
            is_soldier_adviser_flip = (prev_label, label) in {
                ('r_bing', 'r_shi'),
                ('r_shi', 'r_bing'),
                ('b_zu', 'b_shi'),
                ('b_shi', 'b_zu'),
            }

            # If label flips at the same cell with weak confidence, keep previous label.
            if prev_label and prev_label in dic and prev_label != label and score < LABEL_SWITCH_CONF_MIN:
                stabilized[key] = prev_label
            elif prev_label and prev_label in dic and is_soldier_adviser_flip and score < SOLDIER_ADVISER_SWITCH_CONF_MIN:
                stabilized[key] = prev_label
            else:
                stabilized[key] = label
            stabilized_conf[key] = score

        if return_conf:
            return stabilized, stabilized_conf
        return stabilized

    label_map = {k: v[0] for k, v in board_map.items()}
    if return_conf:
        conf_map = {k: v[1] for k, v in board_map.items()}
        return label_map, conf_map
    return label_map


def init_board_state(step0_path):
    global board_state_map
    board_state_map = {}

    step0_img = cv2.imread(step0_path)
    if step0_img is None:
        return

    board_state_map = detect_board_map(step0_img)


def save_move_snapshot(frame, step_idx, piece_name, begin, end, legal=True):
    canvas = frame.copy()

    start_pt = (
        int(np.around(begin_point[0] + begin[0] * GRID_WIDTH_HORI)),
        int(np.around(begin_point[1] + begin[1] * GRID_WIDTH_VERTI)),
    )
    end_pt = (
        int(np.around(begin_point[0] + end[0] * GRID_WIDTH_HORI)),
        int(np.around(begin_point[1] + end[1] * GRID_WIDTH_VERTI)),
    )

    arrow_color = (0, 255, 0) if legal else (0, 0, 255)
    cv2.circle(canvas, start_pt, 6, (255, 255, 255), -1)
    cv2.arrowedLine(canvas, start_pt, end_pt, arrow_color, 3, tipLength=0.25)

    status = 'LEGAL' if legal else 'ILLEGAL'
    text = f"Step {step_idx + 1}: {piece_name} {begin}->{end} [{status}]"

    cv2.putText(
        canvas,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if legal else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    filename = f"step_{step_idx + 1:03d}_{piece_name.replace(' ', '_')}_{status}.png"
    cv2.imwrite(os.path.join(MOVE_CAPTURE_DIR, filename), canvas)




def savePath(beginPoint, endPoint, piece):
    global legal_move	# For indicating error movement
    begin = point_to_board_xy(beginPoint)
    #print(beginPoint, endPoint)
    end = begin
    updown = int(np.around(abs(beginPoint[1]-endPoint[1])/GRID_WIDTH_VERTI))
    leftright = int(np.around(abs(beginPoint[0]-endPoint[0])/GRID_WIDTH_HORI))
    #print(updown, leftright)
    predict_category = PiecePrediction(model, piece, target_size)
    if predict_category == 'grid':
        predict_category = board_state_map.get(begin, 'grid')

    if predict_category == 'grid':
        variety = 'unknown'
        color = 'u'
    else:
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

    board_state_map[begin] = 'grid'
    board_state_map[end] = predict_category

    piece_name = dic.get(predict_category, 'Unknown piece')
    print('{} moved from point {} to point {}'.format(piece_name, begin, end))

    if not ENFORCE_MOVE_RULES:
        text = str(int(begin[0])) + str(int(begin[1])) + str(int(end[0])) + str(int(end[1]))
        return text, predict_category, piece_name, begin, end

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
    return text, predict_category, piece_name, begin, end

def findPoint(point, pointset):
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

def CalculateTrace(pre_img, cur_img, x, y, w, h):
    # Input loca = [x, y, w, h], return all circle center inside the rectangular to pointSet
    pointSet = []
    beginPoint = None
    endPoint = None
    pre_img_gray = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
    cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    pre_img_circle = cv2.HoughCircles(pre_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]
    cur_img_circle = cv2.HoughCircles(cur_img_gray,cv2.HOUGH_GRADIENT,1,40,param1=100,param2=20,minRadius=18,maxRadius=18)[0]

    for j in range(int(np.around(h/GRID_WIDTH_VERTI))):
        for i in range(int(np.around(w/GRID_WIDTH_HORI))):
            pointSet.append([y + (j+0.5)*GRID_WIDTH_VERTI, x + (i+0.5)*GRID_WIDTH_HORI])
    for p in pointSet:
        if beginPoint is not None and endPoint is not None:		# Already find beginPoint and endPoint, exit
            break
        flag1, p1 = findPoint(p, pre_img_circle)
        flag2, p2 = findPoint(p, cur_img_circle)
        if len(pre_img_circle)-len(cur_img_circle) == 1:	# 发生了吃子
            if flag1 == True and flag2 == False:
                beginPoint = p1
            elif flag1 == True and flag2 == True:
                pre_piece = crop_with_margin(pre_img, p1)
                cur_piece = crop_with_margin(cur_img, p2)
                if PiecePrediction(model, pre_piece, target_size) != PiecePrediction(model, cur_piece, target_size):
                    endPoint = p2
        elif len(pre_img_circle) == len(cur_img_circle):	#没有发生棋子减少情况
            if flag1 == True and flag2 == False:
                beginPoint = p1
            elif flag1 == False and flag2 == True:
                endPoint = p2
    if beginPoint is not None and endPoint is not None:
        piece = crop_with_margin(pre_img, beginPoint)
    else:
        return None, None, None
    return beginPoint, endPoint, piece

def changeDetection(previous_step, current_step, visual = False):
    current_frame_gray = cv2.cvtColor(current_step, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_step, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    frame_diff = cv2.medianBlur(frame_diff, 5)
    ret, frame_diff = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_OTSU)
    frame_diff = cv2.medianBlur(frame_diff, 5)
    x, y, w, h = cv2.boundingRect(frame_diff)
    #### For Test ####
    if visual:
        cv2.rectangle(frame_diff, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.imshow('', frame_diff)
        cv2.waitKey(20)
    #### For Test ####
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
        cat1 = PiecePrediction(model, piece1, target_size)
        cat2 = PiecePrediction(model, piece2, target_size)
        dict1[(coordinate[0], coordinate[1])] = cat1
        dict2[(coordinate[0], coordinate[1])] = cat2
    return operator.eq(dict1, dict2)


def is_legal_move(begin, end, predict_category):
    if predict_category == 'grid':
        return True
    

    updown = abs(begin[1] - end[1])
    leftright = abs(begin[0] - end[0])
    variety = predict_category.split('_')[-1]
    color = predict_category.split('_')[0]

    legal = True
    if variety in ['ma']:
        legal = (updown == 1 and leftright == 2) or (updown == 2 and leftright == 1)
    elif variety in ['xiang']:
        legal = (updown == 2 and leftright == 2)
    elif variety in ['shi']:
        legal = (updown == 1 and leftright == 1)
    elif variety in ['jiang', 'shuai']:
        legal = (updown == 1 and leftright == 0) or (updown == 0 and leftright == 1)
    elif variety in ['ju', 'pao']:
        legal = updown == 0 or leftright == 0
    elif variety in ['bing']:
        legal = not (begin[1] < end[1] or (begin[1] >= 5.0 and begin[0] != end[0]) or (begin[1] - end[1] > 1))
    elif variety in ['zu']:
        legal = not (begin[1] > end[1] or (begin[1] <= 4.0 and begin[0] != end[0]) or (end[1] - begin[1] > 1))

    if isRed and color == 'b':
        legal = False
        print('It''s red team''s turn to move')
    if (not isRed) and color == 'r':
        legal = False
        print('It''s black team''s turn to move')

    return legal


def _score_move_candidate(piece_label: str, begin: tuple, end: tuple) -> float:
    """
    Tính điểm cho cặp (begin → end) dựa trên luật di chuyển của quân cờ.
    Điểm cao hơn = khả năng là nước đi đúng cao hơn.
    Trả về -inf nếu là nước đứng yên (dx=dy=0).
    """
    if '_' not in piece_label:
        return 0.0
    variety = piece_label.split('_')[-1]
    dx = abs(end[0] - begin[0])
    dy = abs(end[1] - begin[1])

    if dx == 0 and dy == 0:
        return float('-inf')

    if variety == 'ma':
        return 10.0 if (dx == 1 and dy == 2) or (dx == 2 and dy == 1) else -5.0
    if variety == 'xiang':
        return 10.0 if dx == 2 and dy == 2 else -5.0
    if variety == 'shi':
        return 10.0 if dx == 1 and dy == 1 else -5.0
    if variety in ('jiang', 'shuai'):
        return 10.0 if dx + dy == 1 else -5.0
    if variety in ('ju', 'pao'):
        return 10.0 if (dx == 0 or dy == 0) else -5.0
    if variety in ('bing', 'zu'):
        # Kiểm tra lỏng (không biết chiều đi vì không có color context ở đây)
        return 5.0 if dx + dy <= 2 else -3.0
    return 3.0  # Loại quân không rõ


def infer_move_from_states(prev_map: dict, curr_map: dict):
    """
    So sánh 2 trạng thái bàn cờ để suy ra nước đi.
    Đánh giá tất cả cặp (nguồn × đích) bằng _score_move_candidate(),
    chọn cặp điểm cao nhất — tránh lấy bừa removed[0]/added[0] như code cũ.

    Xử lý được:
      - Nước đi thường (ô nguồn mất quân, ô đích mọc quân mới)
      - Nước ăn quân (ô nguồn mất quân, ô đích đổi sang loại quân mới)
      - Nhiễu detection (nhiều ô thay đổi cùng lúc)
    """
    vacated = [k for k in prev_map if k not in curr_map]
    occupied = [k for k in curr_map if k not in prev_map]
    changed = [k for k in prev_map if k in curr_map and prev_map[k] != curr_map[k]]

    if not vacated:
        return None, None, None

    # Sanity check: nếu quá nhiều ô biến mất cùng lúc thì không phải nước đi thật
    # mà là lỗi calibration grid hoặc nhiễu YOLO toàn bộ frame.
    if len(vacated) > 2:
        return None, None, None

    dst_candidates = occupied + changed
    if not dst_candidates:
        return None, None, None

    best_begin, best_end, best_label = None, None, None
    best_score = float('-inf')

    for begin in vacated:
        piece_label = prev_map[begin]
        for end in dst_candidates:
            move_score = _score_move_candidate(piece_label, begin, end)
            # Phạt khoảng cách Manhattan lớn (ưu tiên nước đi gần hơn để giảm nhiễu)
            dist_penalty = (abs(end[0] - begin[0]) + abs(end[1] - begin[1])) * 0.15
            total = move_score - dist_penalty
            if total > best_score:
                best_score = total
                best_begin, best_end, best_label = begin, end, piece_label

    # Score âm nghĩa quân cờ không thể đi đến đích theo luật → bỏ qua.
    if best_score < 0:
        return None, None, None

    return best_begin, best_end, best_label

def PiecesChangeDetection(current_step):
    global legal_move
    previous_step = cv2.imread('./Test_Image/Step %d.png' % step)
    x, y, w, h = changeDetection(previous_step, current_step, False)
    if w * h < 30 * 30:
        return 0

    prev_map = dict(board_state_map)

    # Luôn dùng fresh detection thay vì voter để tránh false positive do lệch
    # calibration giữa init frame và frame hiện tại.
    curr_map, curr_conf_map = detect_board_map(current_step, prev_map=prev_map, return_conf=True)

    begin, end, predict_category = infer_move_from_states(prev_map, curr_map)
    if begin is None or end is None:
        return 0

    # Dùng nhãn quân từ ô nguồn trong trạng thái cũ (đáng tin hơn).
    rule_piece_label = prev_map.get(begin, 'grid')
    if rule_piece_label == 'grid':
        rule_piece_label = predict_category

    piece_name = dic.get(rule_piece_label, 'Unknown piece')
    move_conf = float(curr_conf_map.get(end, 0.0))

    # Kiểm tra hợp lệ TRƯỚC khi cập nhật trạng thái bàn cờ.
    legal_move = is_legal_move(begin, end, rule_piece_label) if ENFORCE_MOVE_RULES else True

    # Cập nhật board state để theo dõi vị trí thực tế trên bàn cờ vật lý.
    board_state_map.clear()
    board_state_map.update(curr_map)
    board_voter.clear()

    print('{} moved from {} to {} (conf={:.3f}, legal={})'.format(
        piece_name, begin, end, move_conf, legal_move
    ))

    if legal_move:
        cv2.imwrite('./Test_Image/Step %d.png' % (step + 1), current_step)
        save_move_snapshot(current_step, step, piece_name, begin, end, legal=True)
        return 1  # Hợp lệ → main loop flips isRed, tăng step

    # Bất hợp lệ: lưu ảnh bàn cờ hiện tại (board đã thay đổi thật) để tránh
    # lặp phát hiện cùng nước đi bất hợp lệ ở frame tiếp theo.
    cv2.imwrite('./Test_Image/Step %d.png' % step, current_step)
    save_move_snapshot(current_step, step, piece_name, begin, end, legal=False)
    legal_move = True
    return 0  # Trả 0 → main loop KHÔNG tăng step, KHÔNG flip isRed

# if __name__ == '__main__':
# 	# Initialize camera
# 	# cap = cv2.VideoCapture("http://admin:admin@%s:8081/" % ip)
# 	cap = cv2.VideoCapture('./Sources/test55.mp4')
# 	# cap = cv2.VideoCapture(0)
# 	if cap.isOpened():
# 		for j in range(20):
# 			cap.read()
# 		ret, current_frame = cap.read()
# 		current_frame = current_frame[0:480, 0:480]
# 		# current_frame = cv2.resize(current_frame, (480, 480))
# 		cv2.imwrite('./Test_Image/Step 0.png', current_frame)
# 	else:
# 		exit('Camera is not open.')
# 	print('Camera Initialized.')
# 	previous_frame = current_frame
# 	Initialization()
# 	while (cap.isOpened()):
# 		x, y, w, h = changeDetection(current_frame, previous_frame)
# 		if (x == 0 and y == 0 and w == 480 and h == 480):
# 			num = PiecesChangeDetection(current_frame)
# 			if num == 1:
# 				step += 1
# 				isRed = bool(1 - isRed)
# 			elif num == 0: 
# 				pass
# 		previous_frame = current_frame.copy()
# 		ret, current_frame = cap.read()
# 		if not ret:
# 			break
# 		current_frame = current_frame[0:480, 0:480]
# 		# current_frame = cv2.resize(current_frame, (480, 480))
# 		cv2.rectangle(current_frame, ad.begin, (ad.begin[0] + 400, ad.begin[1] + 400), (255, 255, 255), 2)
# 		cv2.imshow('', current_frame)
# 		# cv2.waitKey(1)
# 		key = cv2.waitKey(1) & 0xFF
# 		if key == ord('q'):
# 			break


# 	cap.release()
# 	cv2.destroyAllWindows()
# 	# db.close()

# if __name__ == '__main__':
#     # Mở webcam (thử 0, nếu không được thì thử 1, 2...)
#     # cap = cv2.VideoCapture('./Sources/test3.mp4')
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#     if not cap.isOpened():
#         exit('Camera is not open.')

#     print("Camera mở rồi.")
#     print("Đặt bàn cờ vào khung hình, chỉnh cho rõ nét.")
#     print("Bấm phím 's' để CHỤP Step 0, 'q' để thoát.")

#     step0 = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Không đọc được frame từ camera.")
#             break

#         # Resize/crop về đúng 480x480 (tuỳ bạn đang làm gì)
#         # frame = cv2.resize(frame, (480, 480))
#         frame = frame[0:480, 0:480]

#         cv2.imshow('Preview - nhấn s để chụp Step 0', frame)
#         key = cv2.waitKey(50) & 0xFF

#         if key == ord('s'):
#             step0 = frame.copy()
#             cv2.imwrite('./Test_Image/Step 0.png', step0)
#             print("Đã chụp và lưu /Test_Image/Step 0.png")
#             break
#         elif key == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit(0)

#     if step0 is None:
#         cap.release()
#         cv2.destroyAllWindows()
#         exit("Không có Step 0 để khởi tạo.")

#     print('Camera Initialized.')
#     previous_frame = step0.copy()

#     # Gọi Initialization (đang đọc lại từ file Step 0.png)
#     Initialization()

#     # ----- Vòng lặp chính -----
#     while cap.isOpened():
#         ret, current_frame = cap.read()
#         if not ret:
#             break

#         current_frame = current_frame[0:480, 0:480]

#         x, y, w, h = changeDetection(current_frame, previous_frame)
#         if (x == 0 and y == 0 and w == 480 and h == 480):
#             num = PiecesChangeDetection(current_frame)
#             if num == 1:
#                 step += 1
#                 isRed = bool(1 - isRed)

#         previous_frame = current_frame.copy()

#         cv2.rectangle(current_frame, ad.begin, (ad.begin[0] + 400, ad.begin[1] + 400), (255, 255, 255), 2)
#         cv2.imshow('Chinese Chess', current_frame)
#         key = cv2.waitKey(50) & 0xFF
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':


#     # Mở webcam (nếu 0 không được thì thử 1,2,...)
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     # cap = cv2.VideoCapture('./Sources/test.avi')

#     if not cap.isOpened():
#         exit('Camera is not open.')

#     print("Camera mở rồi.")
#     print("Đặt bàn cờ vào bên trong Ô VUÔNG trên màn hình.")
#     print("Bấm phím 's' để CHỤP Step 0, 'q' để thoát.")

#     step0 = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Không đọc được frame từ camera.")
#             break

#         h, w = frame.shape[:2]

#         # Kích thước ô vuông (tối đa 480, nhưng không lớn hơn frame)
#         side = min(480, w, h)

#         # Cắt và vẽ ô vuông ở giữa khung hình
#         x0 = (w - side) // 2
#         y0 = (h - side) // 2
#         x1 = x0 + side
#         y1 = y0 + side

#         # Vẽ KHUNG Ô VUÔNG để ông canh bàn cờ
#         preview = frame.copy()
#         cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)

#         cv2.imshow("Align board & press 's' to capture Step 0", preview)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('s'):
#             # Cắt đúng vùng ô vuông làm Step 0 (KHÔNG vẽ khung vào ảnh lưu)
#             step0 = frame[y0:y1, x0:x1].copy()
#             cv2.imwrite('./Test_Image/Step 0.png', step0)
# #           print("Đã chụp và lưu /Test_Image/Step 0.png")
#             break
#         elif key == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit(0)

#     if step0 is None:
#         cap.release()
#         cv2.destroyAllWindows()
#         exit("Không có Step 0 để khởi tạo.")

#     print('Camera Initialized.')

#     # ---- từ đây trở xuống: dùng step0 làm frame đầu, giống logic cũ ----
#     previous_frame = step0.copy()

#     # Hàm Initialization của ông vẫn đọc ./Test_Image/Step 0.png
#     Initialization()

#     # Vòng lặp chính
#     while cap.isOpened():
#         ret, current_frame = cap.read()
#         if not ret:
#             break

#         # Cũng cắt đúng vùng ô vuông giống lúc chụp Step 0
#         h, w = current_frame.shape[:2]
#         side = min(480, w, h)
#         x0 = (w - side) // 2
#         y0 = (h - side) // 2
#         x1 = x0 + side
#         y1 = y0 + side
#         current_frame = current_frame[y0:y1, x0:x1]

#         x, y, w_box, h_box = changeDetection(current_frame, previous_frame)
#         if (x == 0 and y == 0 and w_box == side and h_box == side):
#             num = PiecesChangeDetection(current_frame)
#             if num == 1:
#                 step += 1
#                 isRed = bool(1 - isRed)

#         previous_frame = current_frame.copy()

#         # Vẽ lại ô vuông cho vui (không bắt buộc, vì frame đã là 480x480 rồi)
#         cv2.rectangle(current_frame, (0, 0), (side - 1, side - 1), (0, 255, 0), 2)

#         cv2.imshow('Chinese Chess', current_frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('./Sources/test.avi')
    cap = cv2.VideoCapture('./Sources/test9.mp4')

    if not cap.isOpened():
        exit('Camera is not open.')

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 1 or source_fps > 240:
        source_fps = 30.0
    STEP0_STABLE_FRAMES_REQUIRED = max(8, int(source_fps * STEP0_STABLE_SECONDS_REQUIRED))
    STABLE_FRAMES_REQUIRED = max(4, int(source_fps * STABLE_SECONDS_REQUIRED))
    MOVE_DEBOUNCE_FRAMES = max(3, int(source_fps * 0.25))
    frame_wait_ms = max(1, int(1000.0 / source_fps))

    print("Camera mở rồi.")
    print("Đặt bàn cờ vào trong KHUNG VUÔNG (I/J/K/L để di chuyển, +/- để đổi kích thước, F để fit tối đa).")
    print("Đang chờ khung hình ổn định để tự động chụp Step 0 (2-3 giây), bấm 'q' để thoát.")

    step0 = None
    os.makedirs('./Test_Image', exist_ok=True)
    init_prev_gray = None
    init_stable_count = 0
    roi = None

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Không đọc được frame từ camera.")
            break

        h, w = raw_frame.shape[:2]
        if roi is None:
            side = int(min(w, h) * 0.95)
            x = (w - side) // 2
            y = (h - side) // 2
            roi = normalize_roi(x, y, side, w, h)

        view = raw_frame.copy()
        x0, y0, side = roi
        x1, y1 = x0 + side, y0 + side
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 255), 2)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        cv2.line(view, (cx, y0), (cx, y1), (255, 255, 255), 1)
        cv2.line(view, (x0, cy), (x1, cy), (255, 255, 255), 1)

        current_frame = extract_process_frame(raw_frame, roi)
        if current_frame is None:
            break

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if init_prev_gray is not None:
            delta = cv2.absdiff(gray, init_prev_gray)
            score = cv2.countNonZero(cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY)[1])
            if score < STEP0_STABLE_DIFF_THRESHOLD:
                init_stable_count += 1
            else:
                init_stable_count = 0

        init_prev_gray = gray

        remaining = max(0, STEP0_STABLE_FRAMES_REQUIRED - init_stable_count)
        cv2.putText(
            view,
            f"Auto Step0 in: {remaining}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            f"I/J/K/L: move ROI, +/-: size, F: fit max | ROI: {side}x{side}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Canh khung - tu dong chup Step 0", view)
        key = cv2.waitKey(frame_wait_ms) & 0xFF

        if init_stable_count >= STEP0_STABLE_FRAMES_REQUIRED:
            step0 = current_frame.copy()
            cv2.imwrite('./Test_Image/Step 0.png', step0)
            print("Đã tự động chụp Step 0, shape =", step0.shape)
            break

        if key in (ord('i'), ord('I')):
            roi = normalize_roi(x0, y0 - ROI_MOVE_STEP, side, w, h)
            init_stable_count = 0
        elif key in (ord('k'), ord('K')):
            roi = normalize_roi(x0, y0 + ROI_MOVE_STEP, side, w, h)
            init_stable_count = 0
        elif key in (ord('j'), ord('J')):
            roi = normalize_roi(x0 - ROI_MOVE_STEP, y0, side, w, h)
            init_stable_count = 0
        elif key in (ord('l'), ord('L')):
            roi = normalize_roi(x0 + ROI_MOVE_STEP, y0, side, w, h)
            init_stable_count = 0
        elif key in (ord('+'), ord('=')):
            roi = normalize_roi(x0 - ROI_SIZE_STEP // 2, y0 - ROI_SIZE_STEP // 2, side + ROI_SIZE_STEP, w, h)
            init_stable_count = 0
        elif key in (ord('-'), ord('_')):
            roi = normalize_roi(x0 + ROI_SIZE_STEP // 2, y0 + ROI_SIZE_STEP // 2, side - ROI_SIZE_STEP, w, h)
            init_stable_count = 0
        elif key in (ord('f'), ord('F')):
            fit_side = int(min(w, h) * 0.98)
            fit_x = (w - fit_side) // 2
            fit_y = (h - fit_side) // 2
            roi = normalize_roi(fit_x, fit_y, fit_side, w, h)
            init_stable_count = 0

        if key == ord('q'):
            break

    if step0 is None:
        cap.release()
        cv2.destroyAllWindows()
        exit("Không có Step 0 để khởi tạo.")

    print('Camera Initialized.')
    previous_frame = step0.copy()
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    stable_count = 0
    motion_active = False
    frame_index = 0
    detect_cooldown = 0

    Initialization()

    # --- Vòng lặp chính phía sau giữ như cũ ---
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_index += 1

        h, w = raw_frame.shape[:2]
        roi = normalize_roi(roi[0], roi[1], roi[2], w, h)
        current_frame = extract_process_frame(raw_frame, roi)
        if current_frame is None:
            break

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(gray, prev_gray)
        motion_score = cv2.countNonZero(cv2.threshold(frame_delta, MOTION_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1])

        if detect_cooldown > 0:
            detect_cooldown -= 1

        # --- Nhận kết quả từ background detection thread (non-blocking) ---
        try:
            num = _detect_result_q.get_nowait()
            if num == 1:
                step += 1
                isRed = bool(1 - isRed)
                detect_cooldown = MOVE_DEBOUNCE_FRAMES
        except _q.Empty:
            pass

        # --- Motion / stability tracking ---
        if motion_score < STABLE_DIFF_THRESHOLD:
            stable_count += 1
        else:
            motion_active = True
            stable_count = 0
            board_voter.clear()

        # --- Trigger detection async khi đủ điều kiện ---
        with _detect_lock:
            can_detect = not _detect_active

        if detect_cooldown == 0 and motion_active and stable_count >= STABLE_FRAMES_REQUIRED and can_detect:
            with _detect_lock:
                _detect_active = True
            frame_copy = current_frame.copy()
            threading.Thread(
                target=_detection_worker, args=(frame_copy,), daemon=True
            ).start()
            motion_active = False
            stable_count = 0

        # Fast-move fallback: kích hoạt async detection khi đang chuyển động.
        if ENABLE_DETECT_DURING_MOTION and detect_cooldown == 0 and motion_active \
                and frame_index % DETECT_DURING_MOTION_EVERY == 0 and can_detect:
            with _detect_lock:
                _detect_active = True
            frame_copy = current_frame.copy()
            threading.Thread(
                target=_detection_worker, args=(frame_copy,), daemon=True
            ).start()
            motion_active = False
            stable_count = 0

        previous_frame = current_frame.copy()
        prev_gray = gray

        # Hiển thị thông tin trạng thái lên màn hình
        display = current_frame.copy()
        status_text = 'MOVING' if motion_active else f'STABLE {stable_count}/{STABLE_FRAMES_REQUIRED}'
        turn_text = 'Red' if isRed else 'Black'
        color = (0, 0, 255) if motion_active else (0, 255, 0)
        cv2.putText(display, f'Step {step} | {turn_text} | {status_text}',
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.imshow('Chinese Chess Tracker', display)
        key = cv2.waitKey(frame_wait_ms) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()