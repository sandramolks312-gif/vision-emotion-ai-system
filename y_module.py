# y_module.py

from ultralytics import YOLO

model = YOLO("yolov8m.pt")


def detect_objects(frame):
    """
    Runs YOLOv8 on the frame.

    Key fixes vs old version:
    - conf raised to 0.55  : filters weak partial-body detections
    - iou  raised to 0.60  : allows nearby people without merging them
    - min box width/height : drops thin sliver detections (< 40px wide)
    - custom NMS threshold : 0.50 stops merging distinct people

    Returns:
        person_boxes : list of (x1, y1, x2, y2, conf)
        phone_count  : int
    """
    results = model(frame, conf=0.55, iou=0.60)

    person_boxes_raw = []
    phone_count      = 0

    h_frame, w_frame = frame.shape[:2]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_w = x2 - x1
                box_h = y2 - y1

                # Skip tiny / collapsed detections
                if box_w < 40 or box_h < 60:
                    continue

                # Skip boxes that are unrealistically narrow (slivers)
                aspect = box_h / max(box_w, 1)
                if aspect > 6.0:   # taller than 6x its width = sliver
                    continue

                person_boxes_raw.append((x1, y1, x2, y2, conf))

            elif label == "cell phone":
                phone_count += 1

    # NMS with generous IoU so side-by-side people are kept separate
    person_boxes = remove_duplicates(person_boxes_raw, iou_thresh=0.50)

    return person_boxes, phone_count


def remove_duplicates(boxes, iou_thresh=0.50):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept  = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]
    return kept


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    inter_x1   = max(ax1, bx1)
    inter_y1   = max(ay1, by1)
    inter_x2   = min(ax2, bx2)
    inter_y2   = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area     = (ax2 - ax1) * (ay2 - ay1)
    b_area     = (bx2 - bx1) * (by2 - by1)
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0