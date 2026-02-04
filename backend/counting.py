"""
Counting Module
Handles line crossing logic.
"""

class LineCounter:
    def __init__(self, line_y_percent=50):
        self.line_y_percent = line_y_percent
        self.line_y = 0  # To be calculated based on frame height
        
        self.in_count = 0
        self.out_count = 0
        
        # Keep track of previous positions: {track_id: prev_cy}
        self.previous_positions = {}
        
        # Prevent re-counting immediately (optional, debounce)
        self.counted_ids = set() 

    def set_line_position(self, y_percent, frame_height):
        """Update line position."""
        self.line_y_percent = y_percent
        self.line_y = int(frame_height * (y_percent / 100.0))

    def update(self, tracks):
        """
        Update counts based on track movements.
        
        Args:
            tracks: List of Track objects (with track_id, tlbr)
        """
        current_ids = set()
        
        for track in tracks:
            if not track.is_activated:
                continue
                
            tid = track.track_id
            current_ids.add(tid)
            
            x1, y1, x2, y2 = track.tlbr
            cy = int((y1 + y2) / 2)
            
            if tid in self.previous_positions:
                prev_cy = self.previous_positions[tid]
                
                # Check crossing
                # IN: Top -> Bottom (prev < line < curr)
                if prev_cy < self.line_y and cy >= self.line_y:
                    if tid not in self.counted_ids: # Basic duplicate prevention, though ID persistence handles most
                        self.in_count += 1
                        print(f"[Counter] ID {tid} IN (Prev: {prev_cy} -> Curr: {cy} | Line: {self.line_y})")
                
                # OUT: Bottom -> Top (prev > line > curr)
                elif prev_cy > self.line_y and cy <= self.line_y:
                    if tid not in self.counted_ids:
                        self.out_count += 1
                        print(f"[Counter] ID {tid} OUT (Prev: {prev_cy} -> Curr: {cy} | Line: {self.line_y})")
            
            # Update previous position
            self.previous_positions[tid] = cy
            
        # Clean up missing tracks
        # We can implement a cleanup strategy if memory is an issue, 
        # but for FYP this is likely fine or we can remove IDs not seen for X frames.
        # simple cleanup:
        for tid in list(self.previous_positions.keys()):
            if tid not in current_ids:
                del self.previous_positions[tid]

    def get_counts(self):
        return {"in": self.in_count, "out": self.out_count}

    def reset(self):
        self.in_count = 0
        self.out_count = 0
        self.previous_positions.clear()
        self.counted_ids.clear()
