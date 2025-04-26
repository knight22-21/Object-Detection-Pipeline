import cv2
import time
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.change_detection import ChangeDetector
from src.utils import draw_boxes
from collections import defaultdict
import csv


def process_video(input_path, output_path, display=False):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = ObjectDetector()
    tracker = ObjectTracker()
    changer = ChangeDetector()

    frame_count = 0
    start_time = time.time()

    log_file = open("data/detections_log.csv", mode='w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "object_id", "class_name", "x1", "y1", "x2", "y2", "status"])
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes, confs = detector.detect(frame)
        detections = []
        for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)
            detection = ([x, y, w, h], float(conf), int(cls))
            detections.append(detection)

        tracked_objects = tracker.update(detections, frame)
        current_ids = [obj[0] for obj in tracked_objects]
        new_ids, missing_ids = changer.detect_changes(current_ids)

        for obj_id, box, cls_id in tracked_objects:
            x1, y1, x2, y2 = map(int, box)
            class_name = detector.model.names[int(cls_id)]
            status = ""
            if obj_id in new_ids:
                status = "new"
            elif obj_id in missing_ids:
                status = "missing"
            log_writer.writerow([frame_count, obj_id, class_name, x1, y1, x2, y2, status])
        frame_count += 1

        frame = draw_boxes(frame, tracked_objects)

        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25


        for i, nid in enumerate(new_ids):
            msg = f"New Object: ID {nid}"
            text_size = cv2.getTextSize(msg, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 60 - i * line_height
            cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)


        for i, mid in enumerate(missing_ids):
            msg = f"Missing Object: ID {mid}"
            text_size = cv2.getTextSize(msg, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 30 - i * line_height
            cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

        # FPS Calculation
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if display:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Count objects by class
        class_counts = defaultdict(int)
        for _, _, cls_id in tracked_objects:
            class_counts[cls_id] += 1

        # Overlay class-wise counts on frame
        y_offset = 30
        for cls_id, count in class_counts.items():
            label = f"{detector.model.names[int(cls_id)]}: {count}"
            cv2.putText(frame, label, (10, height - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 20


        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    average_fps = frame_count / total_time
    log_file.close()

    print(f"[INFO] Average FPS: {average_fps:.2f}")
