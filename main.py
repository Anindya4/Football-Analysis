from utils import read_video, save_video
import cv2
from tracker import Tracker
from team_assigner import TeamAssign

def main():
    
    #read video:
    video_frames = read_video('./Input/input_3.mp4')
    
    #initialize the tracker:
    tracker = Tracker('models/best_v8x.pt')
    
    tracks = tracker.get_object_tracker(video_frames,
                                        read_from_stub=True,
                                        stub_path='./stubs/new_ip3_track_stub.pkl')
    
    #Interpolate the ball position:
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])
    
    # Assign Player Teams:
    team_assigner = TeamAssign()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    
    # save cropped images of a player:
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]
        
        #crop bbox from frame:
        cropped_img = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
        
        # save the cropped image:
        cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_img)
        break
    
    #frames with ball:
    ball_frames_with_detection = [i for i, frame in enumerate(tracks["ball"]) if frame]
    print("Frames with ball detected:", ball_frames_with_detection)

    #Draw Outpur:
    ##draw object tracks:
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    
    
    #save video:
    save_video(output_video_frames, 'output_videos/output_video_ip3.mp4')
    
    
if __name__ == '__main__':
    main()
