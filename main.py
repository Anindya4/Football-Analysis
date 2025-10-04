from utils import read_video, save_video
import cv2
from tracker import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerTeamAssigner
from camera_movement_estimator import CameraMovmentEst



def main():
    #read video:
    video_frames = read_video('./Input/input_4.mp4')
    
    #initialize the tracker:
    tracker = Tracker('models/best_v8x.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path='./stubs/new_ip4_track_stub.pkl')
    
    #Get object position:
    tracker.get_position_to_tracks(tracks)
    
    # Camera momvent estimator:
    camera_movement_estimator = CameraMovmentEst(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    
    #Interpolate the ball position:
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
     # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerTeamAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.asign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1])   
    team_ball_control = np.array(team_ball_control)    
        
    #frames with ball:
    # ball_frames_with_detection = [i for i, frame in enumerate(tracks["ball"]) if frame]
    # print("Frames with ball detected:", ball_frames_with_detection)

    
    #Draw Output:
    ##draw object tracks:
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    ## draw camera movements:
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    
    #save video:
    save_video(output_video_frames, 'output_videos/output_video_ip4.mp4')
    
    
if __name__ == '__main__':
    main()
