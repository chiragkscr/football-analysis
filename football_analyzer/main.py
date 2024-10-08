from utils import read_video, save_video
from trackers import Tracker
from team_assignerr import TeamAssigner
import cv2
from player_ball_assignser import PlayerBallAssigner
import numpy as np
import torch
from camera_movement_estimator import CameraMovementEstimator
torch.cuda.set_device(0)


def main():
    #read video
    video_frames = read_video(r"football_analyzer\input_vidoes\08fd33_4.mp4")

    


    #initialize Trackers
    tracker = Tracker(r"football_analyzer\models\last.pt") 
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=r'football_analyzer\stubs\track_stubs1.pkl')

    
    #gete object position
    tracker.add_positions_to_tracks(tracks)

    #camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stubs=False, stub_path=r'football_analyzer\stubs\camera_syubs.pkl')
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    #interpolate ball poistions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    
    # #save cropped image of a player
    # for track_id, player in tracks['players'][0].items():
        # bbox= player['bbox']
        # frame = video_frames[0]
        # cv2.imshow("frame", frame)
        # #crop bbox from frame
        # cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # #save  the cropped image
        # cv2.imwrite(f"output_videos/cropped_img.jpg", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        



    #Assign player teams

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_players_team(video_frames[frame_num],
                                                  track['bbox'],
                                                  player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team] 

    #Assign ball aquisition 
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox ']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball']  = True 
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    # #Draw output
    # #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    #draw camera movement 
    output_video_frames = camera_movement_estimator.draw_camera(output_video_frames, camera_movement_per_frame )

    #Save video
    save_video(output_video_frames, 'output_video.avi')
    


if __name__ =="__main__":
    main()