import math

class SmartIOUTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=10, max_distance=60):
        self.next_id = 0
        self.tracked = {}  # id: {'box': (x1,y1,x2,y2), 'missed': 0}
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _center_dist(self, box1, box2):
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        return math.hypot(cx1 - cx2, cy1 - cy2)

    def update(self, detections):
        new_tracked = {}
        matched_ids = set()

        for det in detections:
            best_match = None
            best_iou = 0
            best_dist = float('inf')

            for tid, data in self.tracked.items():
                iou_score = self._iou(det, data['box'])
                dist = self._center_dist(det, data['box'])

                if iou_score >= self.iou_threshold and dist < self.max_distance and tid not in matched_ids:
                    if iou_score > best_iou or (iou_score == best_iou and dist < best_dist):
                        best_match = tid
                        best_iou = iou_score
                        best_dist = dist

            if best_match is not None:
                new_tracked[best_match] = {'box': det, 'missed': 0}
                matched_ids.add(best_match)
            else:
                new_tracked[self.next_id] = {'box': det, 'missed': 0}
                self.next_id += 1

        # Add unmatched previous trackers
        for tid, data in self.tracked.items():
            if tid not in new_tracked:
                data['missed'] += 1
                if data['missed'] <= self.max_disappeared:
                    new_tracked[tid] = data

        self.tracked = new_tracked
        return {tid: data['box'] for tid, data in self.tracked.items()}
