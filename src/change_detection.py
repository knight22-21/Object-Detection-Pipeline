class ChangeDetector:
    def __init__(self, cooldown_frames=30):
        self.previous_ids = set()
        self.new_id_cooldown = {}
        self.missing_id_cooldown = {}
        self.cooldown_frames = cooldown_frames

    def detect_changes(self, current_ids):
        current_ids = set(current_ids)

        # Detect new and missing
        new_ids = current_ids - self.previous_ids
        missing_ids = self.previous_ids - current_ids

        # Update cooldown dicts
        for nid in new_ids:
            self.new_id_cooldown[nid] = self.cooldown_frames
        for mid in missing_ids:
            self.missing_id_cooldown[mid] = self.cooldown_frames

        # Decrease counters
        self.new_id_cooldown = {k: v - 1 for k, v in self.new_id_cooldown.items() if v > 1}
        self.missing_id_cooldown = {k: v - 1 for k, v in self.missing_id_cooldown.items() if v > 1}

        self.previous_ids = current_ids

        return list(self.new_id_cooldown.keys()), list(self.missing_id_cooldown.keys())
