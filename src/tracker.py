from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=50, n_init=3)

    def update(self, detections, frame=None):
        # Pass the frame to allow DeepSort to compute embeddings
        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.det_class
            tracked_objects.append((track_id, ltrb, class_id))
        return tracked_objects
