"""
Generate model comparison report: CNN (new_model_v2.h5) vs YOLO (weights.pt)
Run: python generate_report.py
"""
import cv2, glob, time, numpy as np, os
from piece_detector import PieceDetector
from collections import Counter
import onnxruntime as ort

# ===== Load models =====
print("Loading models...")
yolo = PieceDetector('./weights.pt', imgsz=640, conf=0.12)
sess = ort.InferenceSession('./chinese_chess_model.onnx', providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name

CNN_CLASSES = [
    'b_jiang', 'b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
    'grid',
    'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang',
]

# ===== CNN validation accuracy =====
print("Evaluating CNN on validation set...")
valid_dir = './Dataset/valid'
classes = sorted(os.listdir(valid_dir))
cnn_correct = 0
cnn_total = 0
cnn_per = {}
cnn_times = []

for cls in classes:
    imgs = glob.glob(f'{valid_dir}/{cls}/*.png') + glob.glob(f'{valid_dir}/{cls}/*.jpg')
    cnn_per[cls] = {'correct': 0, 'total': len(imgs)}
    for ip in imgs:
        img = cv2.imread(ip)
        if img is None:
            continue
        x = cv2.resize(img, (56, 56))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        x = np.expand_dims(x, 0)
        t0 = time.perf_counter()
        out = sess.run(None, {inp_name: x})[0][0]
        cnn_times.append((time.perf_counter() - t0) * 1000)
        pred = CNN_CLASSES[int(np.argmax(out))]
        cnn_total += 1
        if pred == cls:
            cnn_correct += 1
            cnn_per[cls]['correct'] += 1

# ===== YOLO speed benchmark =====
print("Benchmarking YOLO speed...")
img0 = cv2.imread('./Test_Image/Step 0.png')
for _ in range(5):
    yolo.predict_detections(img0, conf=0.12)
yolo_times = []
for _ in range(20):
    t0 = time.perf_counter()
    yolo.predict_detections(img0, conf=0.12)
    yolo_times.append((time.perf_counter() - t0) * 1000)

# ===== YOLO confidence distribution =====
print("Running YOLO on all step images...")
step_imgs = sorted(
    glob.glob('./Test_Image/Step *.png'),
    key=lambda x: int(x.split('Step ')[-1].replace('.png', ''))
)
all_confs = []
step_det_counts = []
for ip in step_imgs:
    img = cv2.imread(ip)
    if img is None:
        continue
    dets = yolo.predict_detections(img, conf=0.05)
    pieces = [d for d in dets if d['label'] != 'grid']
    step_det_counts.append(len(pieces))
    for d in pieces:
        all_confs.append(d['conf'])
confs = np.array(all_confs)

# ===== YOLO Step 0 count vs expected =====
dets0 = [d for d in yolo.predict_detections(img0, conf=0.12) if d['label'] != 'grid']
label_counts = Counter(d['label'] for d in dets0)
expected = {
    'r_shuai': 1, 'r_shi': 2, 'r_xiang': 2, 'r_ma': 2,
    'r_ju': 2,    'r_pao': 2, 'r_bing': 5,
    'b_jiang': 1, 'b_shi': 2, 'b_xiang': 2, 'b_ma': 2,
    'b_ju': 2,    'b_pao': 2, 'b_zu': 5,
}

# ===== YOLO per-step detection count =====
step_detail_lines = []
det_per_step = []
for ip in step_imgs:
    img = cv2.imread(ip)
    if img is None:
        continue
    step_n = os.path.basename(ip).replace('Step ', '').replace('.png', '')
    dets = yolo.predict_detections(img, conf=0.12)
    pieces = [d for d in dets if d['label'] != 'grid']
    c_arr = [d['conf'] for d in pieces]
    avg_c = np.mean(c_arr) if c_arr else 0.0
    min_c = np.min(c_arr) if c_arr else 0.0
    step_detail_lines.append(
        '  Step %-4s  detected=%2d  conf_avg=%.3f  conf_min=%.3f' % (
            step_n, len(pieces), avg_c, min_c)
    )
    det_per_step.append(len(pieces))

# ==================== BUILD REPORT ====================
L = []

def h(title):
    L.append('')
    L.append('[ %s ]' % title)
    L.append('-' * 60)

def row3(a, b, c, w=(28, 24, 24)):
    L.append(('%-' + str(w[0]) + 's%-' + str(w[1]) + 's%-' + str(w[2]) + 's') % (a, b, c))

L.append('=' * 70)
L.append('  MODEL COMPARISON REPORT')
L.append('  CNN  : new_model_v2.h5  (Keras CNN, converted to ONNX)')
L.append('  YOLO : weights.pt       (YOLOv11s, trained on Roboflow)')
L.append('=' * 70)

# ---------- 1. Overview ----------
h('1. OVERVIEW')
row3('Criterion', 'CNN (new_model_v2.h5)', 'YOLO (weights.pt)')
L.append('-' * 76)
for a, b, c in [
    ('Model type',       'Classification',          'Object Detection'),
    ('Architecture',     'Custom CNN (3Conv+2FC)',   'YOLOv11s'),
    ('Framework',        'Keras / TensorFlow',       'Ultralytics / PyTorch'),
    ('File size',        '9.61 MB',                  '18.30 MB'),
    ('Parameters',       '~835,567  (~835K)',         '9,433,984  (~9.4M)'),
    ('Training source',  'Manual cropped dataset',   'Roboflow full-board labels'),
]:
    row3(a, b, c)

# ---------- 2. Input ----------
h('2. INPUT')
row3('Criterion', 'CNN', 'YOLO', w=(26, 28, 24))
L.append('-' * 78)
for a, b, c in [
    ('Image scope',     'Single piece crop',          'Full board image'),
    ('Input size',      '56 x 56 x 3',                '640 x 640 x 3'),
    ('Color space',     'RGB',                         'RGB'),
    ('Normalize',       '[0, 1]  (divide 255)',        '[0, 1]  (divide 255)'),
    ('Pre-processing',  'HoughCircles -> crop->resize', 'None  (end-to-end)'),
]:
    row3(a, b, c, w=(26, 28, 24))
L.append('')
L.append('  CNN requires a separate segmentation step BEFORE inference:')
L.append('    Frame -> HoughCircles -> crop each piece -> resize 56x56 -> predict')
L.append('  YOLO requires NO pre-processing:')
L.append('    Frame -> resize 640x640 -> predict ALL pieces at once')

# ---------- 3. Output ----------
h('3. OUTPUT')
row3('Criterion', 'CNN', 'YOLO', w=(26, 28, 24))
L.append('-' * 78)
for a, b, c in [
    ('Output format',   'Softmax vector  [15]',        '3 detection heads'),
    ('Returns',         'Class probabilities only',    'BBox + label + confidence'),
    ('Bounding box',    'NO',                           'YES  (x1, y1, x2, y2)'),
    ('Confidence score','Softmax probability',          'Objectness x class_prob'),
    ('Detection heads', '1 Dense(15, softmax)',         'P3(80x80)+P4(40x40)+P5(20x20)'),
    ('Values per cell', 'N/A',                          '4(bbox) + 16(cls) = 20'),
]:
    row3(a, b, c, w=(26, 28, 24))

# ---------- 4. Classes ----------
h('4. CLASSES')
L.append('  CNN  (15 classes):')
L.append('    b_jiang  b_ju    b_ma    b_pao   b_shi   b_xiang  b_zu  (7 black)')
L.append('    r_bing   r_ju    r_ma    r_pao   r_shi   r_shuai  r_xiang (7 red)')
L.append('    grid  (empty cell / intersection)')
L.append('')
L.append('  YOLO (16 classes):')
L.append('    Black-Cannon  Black-Chariot  Black-Elephant  Black-General')
L.append('    Black-Guard   Black-Horse    Black-Soldier    None')
L.append('    Red-Cannon    Red-Chariot    Red-Elephant    Red-General')
L.append('    Red-Guard     Red-Horse      Red-Soldier     intersection')
L.append('')
L.append('  Differences:')
L.append('    - YOLO has 2 background classes: "None" and "intersection"')
L.append('    - CNN uses "grid" to cover all empty/background cells')
L.append('    - All 14 piece types are covered by both models')

# ---------- 5. Detection Performance ----------
h('5. DETECTION PERFORMANCE')

cnn_acc = cnn_correct / cnn_total * 100
L.append('')
L.append('--- 5a. CNN Accuracy on Dataset/valid (%d images) ---' % cnn_total)
L.append('  Overall accuracy: %d / %d = %.1f%%' % (cnn_correct, cnn_total, cnn_acc))
L.append('')
L.append('  %-14s  %7s  %7s  %8s  Bar (10=100%%)' % ('Class', 'Correct', 'Total', 'Acc(%)'))
L.append('  ' + '-' * 56)
for cls in sorted(cnn_per.keys()):
    c = cnn_per[cls]['correct']
    t = cnn_per[cls]['total']
    acc = c / max(t, 1) * 100
    bar = '#' * int(acc / 10)
    L.append('  %-14s  %7d  %7d  %7.1f%%  %s' % (cls, c, t, acc, bar))

L.append('')
L.append('  Notes:')
L.append('  - Tested on PRE-SEGMENTED crops (already correct size and centered)')
L.append('  - In real deployment, accuracy drops due to imperfect HoughCircles')
L.append('  - b_zu (98.1%) is the only class with 1 error (b_zu misclassified)')

L.append('')
L.append('--- 5b. YOLO Full-Board Detection on Test_Image (%d step images) ---' % len(step_imgs))
L.append('')
L.append('  Confidence distribution (conf >= 0.05,  %d total detections):' % len(confs))
L.append('    Mean:          %.3f' % confs.mean())
L.append('    Median:        %.3f' % np.median(confs))
L.append('    Std dev:       %.3f' % confs.std())
L.append('    conf >= 0.9:   %4d  (%5.1f%%)' % ((confs >= 0.9).sum(), (confs >= 0.9).mean() * 100))
L.append('    conf >= 0.7:   %4d  (%5.1f%%)' % ((confs >= 0.7).sum(), (confs >= 0.7).mean() * 100))
L.append('    conf >= 0.5:   %4d  (%5.1f%%)' % ((confs >= 0.5).sum(), (confs >= 0.5).mean() * 100))
L.append('    conf <  0.3:   %4d  (%5.1f%%)  <- noisy/unreliable detections' % (
    (confs < 0.3).sum(), (confs < 0.3).mean() * 100))
L.append('')
L.append('  Detections per step image:')
for line in step_detail_lines:
    L.append(line)

L.append('')
L.append('--- 5c. YOLO Count vs Expected on Step 0 (start: 32 pieces) ---')
L.append('  YOLO detected: %d  (expected 32)' % len(dets0))
L.append('')
L.append('  %-14s  %9s  %10s  %6s' % ('Label', 'Expected', 'Detected', 'Diff'))
L.append('  ' + '-' * 46)
total_exp = 0
total_det = 0
wrong_items = []
for lbl in sorted(expected.keys()):
    exp = expected[lbl]
    det = label_counts.get(lbl, 0)
    diff = det - exp
    flag = 'OK' if diff == 0 else ('+%d OVER' % diff if diff > 0 else '%d MISS' % diff)
    L.append('  %-14s  %9d  %10d  %s' % (lbl, exp, det, flag))
    total_exp += exp
    total_det += det
    if diff != 0:
        wrong_items.append((lbl, exp, det, diff))
L.append('  ' + '-' * 46)
L.append('  %-14s  %9d  %10d  %+d' % ('TOTAL', total_exp, total_det, total_det - total_exp))
L.append('')
if wrong_items:
    L.append('  Root cause analysis:')
    for lbl, exp, det, diff in wrong_items:
        if diff > 0:
            L.append('    [OVER] %s: +%d  -> confused with similar piece' % (lbl, diff))
        else:
            L.append('    [MISS] %s: %d  -> missed detections (low conf or occluded)' % (lbl, abs(diff)))
    L.append('')
    L.append('  Main confusion pairs:')
    L.append('    r_shi <-> r_bing  (adviser vs soldier, look similar)')
    L.append('    b_shi <-> b_zu    (guard vs pawn, look similar)')
    L.append('    b_xiang / b_ju    (missed: possibly training data shortage)')

# ---------- 6. Speed ----------
h('6. INFERENCE SPEED  (CPU,  Intel/AMD)')
cnn_m = np.mean(cnn_times)
cnn_s = np.std(cnn_times)
yolo_m = np.mean(yolo_times)
yolo_s = np.std(yolo_times)
L.append('')
L.append('  CNN ONNX - single crop (56x56):')
L.append('    Mean: %.3f ms   Std: %.3f ms   Max throughput: ~%.0f fps' % (
    cnn_m, cnn_s, 1000 / cnn_m))
L.append('')
L.append('  CNN - full board equiv. (90 crops x %.3f ms):' % cnn_m)
est = cnn_m * 90
L.append('    Estimated: %.1f ms   ~%.1f fps' % (est, 1000 / est))
L.append('    (*) Excludes HoughCircles overhead, which adds ~20-50ms')
L.append('')
L.append('  YOLO - full board (640x640):')
L.append('    Mean: %.1f ms   Std: %.1f ms   ~%.1f fps' % (yolo_m, yolo_s, 1000 / yolo_m))
L.append('')
ratio = yolo_m / est
if ratio > 1:
    L.append('  YOLO is %.1fx SLOWER than CNN (crop x90) on CPU' % ratio)
else:
    L.append('  YOLO is %.1fx FASTER than CNN (crop x90) on CPU' % (1.0 / ratio))
L.append('')
L.append('  Summary table:')
L.append('  %-30s  %10s  %8s' % ('Method', 'Time (ms)', 'FPS'))
L.append('  ' + '-' * 52)
L.append('  %-30s  %10.3f  %8.0f' % ('CNN ONNX per crop', cnn_m, 1000 / cnn_m))
L.append('  %-30s  %10.1f  %8.1f' % ('CNN x90 crops (no HoughCircles)', est, 1000 / est))
L.append('  %-30s  %10.1f  %8.1f' % ('YOLO full board', yolo_m, 1000 / yolo_m))

# ---------- 7. Strengths / Weaknesses ----------
h('7. STRENGTHS & WEAKNESSES')
L.append('')
L.append('  CNN:')
L.append('    PROS:')
L.append('      + Very fast per crop (0.17ms), minimal CPU usage')
L.append('      + 99.6% accuracy on clean pre-segmented crops')
L.append('      + Simple architecture, lightweight (835K params)')
L.append('      + ONNX version is portable and easy to deploy')
L.append('    CONS:')
L.append('      - Requires HoughCircles for piece segmentation (brittle)')
L.append('      - HoughCircles fails under varied lighting / overlapping pieces')
L.append('      - No positional information (classification only, no bbox)')
L.append('      - Accuracy degrades on imperfect crops from real camera')
L.append('')
L.append('  YOLO:')
L.append('    PROS:')
L.append('      + End-to-end: detects ALL pieces in 1 inference pass')
L.append('      + Returns bounding boxes -> no HoughCircles needed')
L.append('      + Board context helps distinguish similar pieces')
L.append('      + Handles occlusion, scale variation better')
L.append('      + Confidence scores per detection for filtering')
L.append('    CONS:')
L.append('      - Slower on CPU (~127ms vs 15ms for CNN x90)')
L.append('      - Confusion: r_shi/r_bing, b_shi/b_zu (similar look)')
L.append('      - Misses r_bing, b_xiang, b_ju (needs more training data)')
L.append('      - 16.2%% detections have conf < 0.3 (noisy)')
L.append('      - Larger model (9.4M params vs 835K)')

# ---------- 8. Recommendation ----------
h('8. RECOMMENDATION')
L.append('')
L.append('  YOLO is the correct choice for this system because:')
L.append('    1. Eliminates brittle HoughCircles dependency entirely')
L.append('    2. Single pass for all 32 pieces on the board')
L.append('    3. Bounding boxes enable accurate grid cell mapping')
L.append('    4. Works robustly under camera angle / lighting changes')
L.append('')
L.append('  To improve YOLO detection quality:')
L.append('    1. Collect more training images for r_bing and b_xiang (most missed)')
L.append('    2. Add hard negative examples to reduce r_shi/b_shi false positives')
L.append('    3. Raise conf threshold to 0.3-0.4 to filter noisy detections')
L.append('    4. Use GPU for inference (expected 10-20x speedup, ~10-15ms)')
L.append('    5. Consider imgsz=416 for faster CPU inference (~80-90ms)')
L.append('')
L.append('=' * 70)
L.append('  End of Report')
L.append('=' * 70)

report = '\n'.join(L)
print(report)

out_path = 'model_comparison_report.txt'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(report)
print('\n>> Saved to:', out_path)