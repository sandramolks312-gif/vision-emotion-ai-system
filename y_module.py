# y_module.py

from ultralytics import YOLO

model = YOLO("yolov8m.pt")


def detect_objects(frame):
    results = model(frame, conf=0.50, iou=0.35)  # higher conf = fewer false detections

    person_boxes = []
    phone_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2, conf))

            if label == "cell phone":
                phone_count += 1

    person_count = len(remove_duplicates(person_boxes, iou_thresh=0.35))

    return person_count, phone_count


def remove_duplicates(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_thresh]
    return kept


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0