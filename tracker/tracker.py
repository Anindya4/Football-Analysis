from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import os
import sys
import cv2
sys.path.append('../')
from utils import center_of_bbox, width_of_bbox
import pandas as pd
class Tracker: 
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
        track_activation_threshold=0.1,    # This was 'track_thresh' - lowered to match your YOLO conf
        lost_track_buffer=30,              # Keep default
        minimum_matching_threshold=0.8,    # Keep default  
        frame_rate=25,                     # Match your video fps
        minimum_consecutive_frames=1       # Keep default
        )

    def interpolate_ball_position(self, ball_positions):
        print(f"Total frames: {len(ball_positions)}")
        
        # First, let's see what frames have data
        frames_with_data = []
        for i, frame in enumerate(ball_positions):
            if frame:  # Non-empty frame
                frames_with_data.append(i)
        
        print(f"Frames with data: {frames_with_data[:10]}...")  # Show first 10
        
        # Look at the actual structure of frames with data
        if frames_with_data:
            sample_frame_idx = frames_with_data[0]
            sample_frame = ball_positions[sample_frame_idx]
            print(f"Sample frame {sample_frame_idx}: {sample_frame}")
            print(f"Sample frame keys: {list(sample_frame.keys()) if isinstance(sample_frame, dict) else 'Not a dict'}")
            
            # If it's a dict, look at each key's value
            if isinstance(sample_frame, dict):
                for key, value in sample_frame.items():
                    print(f"  Key '{key}': {value}")
                    if isinstance(value, dict) and 'bbox' in value:
                        print(f"    bbox: {value['bbox']}")
        
        # Now let's see what the extraction actually gives us
        ball_positions_extracted = [x.get(0,{}).get('bbox',[]) for x in ball_positions]
        
        # Check what we actually extracted
        non_empty_extractions = [pos for pos in ball_positions_extracted if pos]
        print(f"Non-empty extractions: {len(non_empty_extractions)}")
        if non_empty_extractions:
            print(f"Sample extraction: {non_empty_extractions[0]}")
        
        # Don't create DataFrame yet, just return original for now
        return ball_positions
        
    def detect_frames(self, frames):
        batch_size= 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracker(self, frames, read_from_stub=False, stub_path=None):
        print("[DEBUG] get_object_tracker() is running")
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        
        detections = self.detect_frames(frames)
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            
            #convert to supervison detection formate:
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            
            #convert goalkeeper to player:
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv['player']
            
            #Track objects:
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]              
                
                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)
        
        return tracks    
    
    # def get_object_tracker(self, frames, read_from_stub=False, stub_path=None):
    #     print("[DEBUG] get_object_tracker() is running")
    #     if read_from_stub and stub_path is not None and os.path.exists(stub_path):
    #         with open(stub_path,'rb') as f:
    #             tracks = pickle.load(f)
    #         return tracks
        
    #     detections = self.detect_frames(frames)
        
    #     tracks={
    #         "players":[],
    #         "referees":[],
    #         "ball":[]
    #     }
        
    #     for frame_num, detection in enumerate(detections):
    #         cls_names = detection.names
    #         cls_names_inv = {v:k for k,v in cls_names.items()}
            
    #         # DEBUG: Check early frames for ball detections
    #         if frame_num < 10:
    #             ball_count = 0
    #             if detection.boxes is not None:
    #                 for box in detection.boxes:
    #                     if cls_names[int(box.cls.item())] == "ball":
    #                         ball_count += 1
    #             print(f"[DEBUG] Frame {frame_num}: {ball_count} balls detected by YOLO")
            
    #         #convert to supervison detection formate:
    #         detection_supervision = sv.Detections.from_ultralytics(detection)
            
    #         # DEBUG: Check supervision conversion
    #         if frame_num < 10:
    #             ball_detections_sv = detection_supervision[detection_supervision.class_id == cls_names_inv['ball']]
    #             print(f"[DEBUG] Frame {frame_num}: {len(ball_detections_sv)} balls after supervision conversion")
            
    #         #convert goalkeeper to player:
    #         for object_idx, class_id in enumerate(detection_supervision.class_id):
    #             if cls_names[class_id] == "goalkeeper":
    #                 detection_supervision.class_id[object_idx] = cls_names_inv['player']
            
    #         #Track objects:
    #         detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
    #         # DEBUG: Check tracking results
    #         if frame_num < 10:
    #             ball_tracks = 0
    #             for frame_detection in detection_with_tracks:
    #                 cls_id = frame_detection[3]
    #                 if cls_id == cls_names_inv["ball"]:
    #                     ball_tracks += 1
    #             print(f"[DEBUG] Frame {frame_num}: {ball_tracks} balls after tracking")
            
    #         tracks["players"].append({})
    #         tracks["referees"].append({})
    #         tracks["ball"].append({})
            
    #         for frame_detection in detection_with_tracks:
    #             bbox = frame_detection[0].tolist()
    #             cls_id = frame_detection[3]
    #             track_id = frame_detection[4]              
                
    #             if cls_id == cls_names_inv["player"]:
    #                 tracks["players"][frame_num][track_id] = {"bbox": bbox}
    #             elif cls_id == cls_names_inv["referee"]:
    #                 tracks["referees"][frame_num][track_id] = {"bbox": bbox}
    #             elif cls_id == cls_names_inv["ball"]:
    #                 tracks["ball"][frame_num][track_id] = {"bbox": bbox}
    #                 print(f"[DEBUG] Frame {frame_num}: Ball tracked with ID {track_id}")
        
    #     if stub_path is not None:
    #         with open(stub_path, 'wb') as f:
    #             pickle.dump(tracks,f)
        
    #     return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = center_of_bbox(bbox)
        width = width_of_bbox(bbox)
        
        #draw the circle:
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=240,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
            )


        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED              
            )
        
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2
            )    
        return frame  
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame
    
    def draw_annotations(self, video_frames, tracks):
        print("[DEBUG] draw_annotations() is running")
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referees_dict = tracks["referees"][frame_num]
            
            #Draw Players:
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color , track_id )
            
            #Draw referees:
            for _, refeere in referees_dict.items():
                frame = self.draw_ellipse(frame, refeere["bbox"], (0,255,0))
            
            #Draw Ball:
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,255))
            
            output_video_frames.append(frame)
        
        return output_video_frames
        
     

    
    
    
    
    
    
    
    
    
