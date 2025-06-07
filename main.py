from utils import read_video, save_video

from tracker import Tracker


def main():
    
    #read video:
    video_frames = read_video('D:\Programing\Object Detection\Input\input_3.mp4')
    
    #initialize the tracker:
    tracker = Tracker('models/best_v8x.pt')
    
    tracks = tracker.get_object_tracker(video_frames,
                                        read_from_stub=True,
                                        stub_path='./stubs/ip3_track_stub.pkl')
    
    
    ball_frames_with_detection = [i for i, frame in enumerate(tracks["ball"]) if frame]
    print("Frames with ball detected:", ball_frames_with_detection)

    #Draw Outpur:
    ##draw object tracks:
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    
    
    #save video:
    save_video(output_video_frames, 'output_videos/output_video_ip3.mp4')
    
    
if __name__ == '__main__':
    main()
