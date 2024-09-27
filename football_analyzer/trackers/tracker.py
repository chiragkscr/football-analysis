from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import measure_distance, get_bbox_width, get_center_of_bbox, get_foot_position
import cv2
import numpy as np
import pandas as pd

#class Tracker which has all the methods to store, return tracks/frames
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    #Uses yolo for Detection 
    def detect_frames(self, frames):
        batch_size= 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size], conf=.1 )
            detections+=detections_batch
            
        return detections

    def add_positions_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object=='ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position


    #returns detected objects with trackletst
    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None) :

        #checks if its should be loaded from stubs
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f :
                tracks = pickle.load(f)
            return tracks
        
        #calling the above function
        detections = self.detect_frames(frames)

        tracks = {
            'players':[],  #{0:{bbox:[0,0,0,0], 1: {bboox:[0000]}}} so this for one frame, and its gonna go until we finish the frame
            'refrees':[],
            'ball':[]

        }


        for frame_num, detection in enumerate(detections):
            
            cls_names = detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}

                #convert to supervision detection format

            detection_supervision = sv.Detections.from_ultralytics(detection)

                #convert goalkeeper to player object 
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] =='goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            #using tracker to track objects in the frame
            detection_with_tracks= self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['refrees'].append({})
            tracks['ball'].append({})


            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}

                if cls_id ==cls_names_inv['referee']:
                    tracks['refrees'][frame_num][track_id] = {'bbox':bbox}

                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}
 
                # print(detection_with_tracks)

        #dumping the detection tracks to pickle file
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    #function to draw ellipse around the player 
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)


        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
 
        return frame


    #function to draw traingle above the ball
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_controls):
        #draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850),(1900,970),(255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_controls_till_frame = team_ball_controls[:frame_num+1]
        #Get the number of time each team has the ball
        team1_num_frames = team_ball_controls_till_frame[team_ball_controls_till_frame==1].shape[0]
        team2_num_frames = team_ball_controls_till_frame[team_ball_controls_till_frame==2].shape[0]

        team1 = team1_num_frames/(team1_num_frames+team2_num_frames)
        team2 = team2_num_frames/(team2_num_frames+team1_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
        cv2.putText(frame, f"Team 2 Ball Control: {team2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)


        return frame

    #function to draw all the annotations on the frame
    def draw_annotations(self,video_frames, tracks, team_ball_control):

        output_video_frames= []

        for frame_num, frame in enumerate(video_frames):
            
        
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["refrees"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
            

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(255,0,0))


            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))
            
            #Draw team possesion 
            frame = self.draw_team_ball_control(frame, frame_num,team_ball_control )


            output_video_frames.append(frame)



        return output_video_frames
        
    def interpolate_ball_positions(self, ball_positions):
        #Extract the ball's bounding box for each frame, default to empty dict if missing
        ball_positions = [x.get(1,{}).get('bbox',{}) for x in ball_positions]

        #Convert the list of bounding boxes to a DataFrame for interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #interpolate missing values
        df_ball_positions = df_ball_positions.interpolate(limit_direction='both')
        df_ball_positions = df_ball_positions.bfill() #filling missing values at the beginig if needed
        df_ball_positions = df_ball_positions.ffill() #Fill missing values at the end if needed

        #convert DataFrame bakc to a list of dictinaries with ball positions
        ball_positions = [{1:{"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        
    
        