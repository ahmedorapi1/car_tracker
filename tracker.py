import cv2
import math
from detector import ObjectDetection

cap = cv2.VideoCapture('highway.mp4')
od = ObjectDetection()

count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    x, y, _ = frame.shape
    roi = frame[400:, :650]
    (class_ids, scores, boxes) = od.detect(roi)

    center_points_cur_frame = []
    for (x1, y1, x2, y2) in boxes.astype(int):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_points_cur_frame.append((cx, cy))

    if count <= 2:
        for pt in center_points_cur_frame:
            already_assigned = any(math.hypot(pt[0]-p[0], pt[1]-p[1]) < 50 for p in tracking_objects.values())
            if not already_assigned:
                tracking_objects[track_id] = pt
                track_id += 1
    else:
        # Match existing tracks to current centers
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, prev_pt in tracking_objects_copy.items():
            object_exists = False
            closest_pt = None
            closest_dist = 1e9

            for pt in center_points_cur_frame_copy:
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < closest_dist:
                    closest_dist = distance
                    closest_pt = pt

            # Update track if close enough
            if closest_pt is not None and closest_dist < 60:
                tracking_objects[object_id] = closest_pt
                object_exists = True
                if closest_pt in center_points_cur_frame:
                    center_points_cur_frame.remove(closest_pt)  # claim it

            # Remove lost tracks
            if not object_exists:
                tracking_objects.pop(object_id, None)

        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1


    for (x1, y1, x2, y2) in boxes.astype(int):
        cv2.rectangle(frame, (x1, y1+400), (x2, y2+400), (0, 255, 0), 2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy+400), 3, (255, 0, 0), -1)

    # Draw track IDs at their current center
    for object_id, (cx, cy) in tracking_objects.items():
        cv2.putText(frame, f'ID {object_id}', (cx - 10, cy+400 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy+400), 4, (255, 255, 0), -1)

    cv2.imshow("frame", frame)

    center_points_prev_frame = center_points_cur_frame
    count += 1
    key = cv2.waitKey(30)
    if key & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
