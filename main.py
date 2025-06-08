from utils import read_video, save_video
import cv2
from tracker import Tracker


def main():
    
    #read video:
    video_frames = read_video('D:\Programing\Football-Analysis\Input\input_3.mp4')
    
    #initialize the tracker:
    tracker = Tracker('models/best_v8x.pt')
    
    tracks = tracker.get_object_tracker(video_frames,
                                        read_from_stub=True,
                                        stub_path='./stubs/ip3_track_stub.pkl')
    
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
