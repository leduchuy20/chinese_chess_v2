import os

import cv2
import numpy as np
from ultralytics import YOLO


CANONICAL_LABELS = {
    "b_jiang",
    "b_ju",
    "b_ma",
    "b_pao",
    "b_shi",
    "b_xiang",
    "b_zu",
    "r_bing",
    "r_ju",
    "r_ma",
    "r_pao",
    "r_shi",
    "r_shuai",
    "r_xiang",
    "grid",
}


LABEL_ALIASES = {
    "b_general": "b_jiang",
    "b_king": "b_jiang",
    "black_general": "b_jiang",
    "black_king": "b_jiang",
    "b_rook": "b_ju",
    "b_chariot": "b_ju",
    "black_rook": "b_ju",
    "black_chariot": "b_ju",
    "b_knight": "b_ma",
    "black_knight": "b_ma",
    "b_horse": "b_ma",
    "black_horse": "b_ma",
    "b_cannon": "b_pao",
    "black_cannon": "b_pao",
    "b_guard": "b_shi",
    "black_guard": "b_shi",
    "b_advisor": "b_shi",
    "b_adviser": "b_shi",
    "black_advisor": "b_shi",
    "black_adviser": "b_shi",
    "b_elephant": "b_xiang",
    "black_elephant": "b_xiang",
    "b_bishop": "b_xiang",
    "black_bishop": "b_xiang",
    "b_soldier": "b_zu",
    "b_pawn": "b_zu",
    "black_soldier": "b_zu",
    "black_pawn": "b_zu",
    "r_general": "r_shuai",
    "r_king": "r_shuai",
    "red_general": "r_shuai",
    "red_king": "r_shuai",
    "r_rook": "r_ju",
    "r_chariot": "r_ju",
    "red_rook": "r_ju",
    "red_chariot": "r_ju",
    "r_knight": "r_ma",
    "red_knight": "r_ma",
    "r_horse": "r_ma",
    "red_horse": "r_ma",
    "r_cannon": "r_pao",
    "red_cannon": "r_pao",
    "r_guard": "r_shi",
    "red_guard": "r_shi",
    "r_advisor": "r_shi",
    "r_adviser": "r_shi",
    "red_advisor": "r_shi",
    "red_adviser": "r_shi",
    "r_elephant": "r_xiang",
    "red_elephant": "r_xiang",
    "r_bishop": "r_xiang",
    "red_bishop": "r_xiang",
    "r_soldier": "r_bing",
    "r_pawn": "r_bing",
    "red_soldier": "r_bing",
    "red_pawn": "r_bing",
    "empty": "grid",
    "none": "grid",
    "background": "grid",
    "board": "grid",
    "intersection": "grid",
    "cross": "grid",
}


class PieceDetector:
    def __init__(
        self,
        weights_path: str = "./weights.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
    ):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Khong tim thay model: {weights_path}")
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def _resolve_name(self, class_id: int) -> str:
        names = self.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        key = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
        if key in CANONICAL_LABELS:
            return key
        return LABEL_ALIASES.get(key, key)

    def predict_piece(self, crop: np.ndarray, default_label: str = "grid") -> str:
        if crop is None or crop.size == 0:
            return default_label

        # YOLO detect models are typically trained on larger context. Upscale tiny crops
        # to improve detection stability for per-piece classification.
        h, w = crop.shape[:2]
        min_side = min(h, w)
        if min_side < 160:
            scale = max(1, int(np.ceil(160 / max(1, min_side))))
            crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        results = self.model.predict(
            source=crop,
            imgsz=self.imgsz,
            conf=min(self.conf, 0.05),
            iou=self.iou,
            verbose=False,
            device="cpu",
        )

        if not results:
            return default_label

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return default_label

        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        ranked_indices = np.argsort(-confs)
        best_grid_label = default_label

        for idx in ranked_indices:
            raw_label = self._resolve_name(int(class_ids[idx]))
            normalized = self.normalize_label(raw_label)

            if normalized not in CANONICAL_LABELS:
                continue

            if normalized != "grid":
                return normalized

            best_grid_label = normalized

        return best_grid_label

    def predict_detections(self, image: np.ndarray, conf: float | None = None) -> list[dict]:
        if image is None or image.size == 0:
            return []

        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            verbose=False,
            device="cpu",
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[dict] = []
        for box, score, cls_id in zip(xyxy, confs, class_ids):
            raw_label = self._resolve_name(int(cls_id))
            label = self.normalize_label(raw_label)
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "label": label,
                    "raw_label": raw_label,
                    "conf": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "cx": float((x1 + x2) * 0.5),
                    "cy": float((y1 + y2) * 0.5),
                }
            )

        return detections

    def predict_board_state(
        self,
        image: np.ndarray,
        begin_pt,
        grid_w: float,
        grid_h: float,
        conf: float | None = None,
    ) -> tuple[dict, dict]:
        """
        Single YOLO inference trên toàn bộ ảnh bàn cờ.
        Snap tâm mỗi detection về ô lưới (bx, by) gần nhất.
        Trả về (label_map, conf_map) — nhanh hơn nhiều so với gọi predict_piece() 90 lần.

        Args:
            image:     Ảnh bàn cờ (BGR numpy array).
            begin_pt:  Toạ độ pixel (x, y) của góc trên-trái bàn cờ.
            grid_w:    Khoảng cách pixel giữa 2 cột lưới liền nhau.
            grid_h:    Khoảng cách pixel giữa 2 hàng lưới liền nhau.
            conf:      Ngưỡng confidence; None = dùng self.conf.

        Returns:
            label_map: {(bx, by): label_str}
            conf_map:  {(bx, by): float_confidence}
        """
        if image is None or image.size == 0:
            return {}, {}

        detections = self.predict_detections(image, conf=conf)
        board: dict = {}
        for det in detections:
            label = det['label']
            if label == 'grid':
                continue
            bx = int(round((det['cx'] - begin_pt[0]) / grid_w))
            by = int(round((det['cy'] - begin_pt[1]) / grid_h))
            if not (0 <= bx <= 8 and 0 <= by <= 9):
                continue
            key = (bx, by)
            if key not in board or det['conf'] > board[key][1]:
                board[key] = (label, float(det['conf']))

        label_map = {k: v[0] for k, v in board.items()}
        conf_map = {k: v[1] for k, v in board.items()}
        return label_map, conf_map
