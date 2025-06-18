from ultralytics import YOLO

model = YOLO('models/best_v8x.pt')

results = model.predict('D:\Programing\Football-Analysis\Input\input_4.mp4', save= True, conf=0.1)
print(results[0])
print("..................................................................................................................")

for box in results[0].boxes:
    print(box)

# from ultralytics import YOLO

# model = YOLO('models/best_v8x.pt')

# print("=== YOUR CUSTOM MODEL CLASSES ===")
# print("Class names mapping:", model.names)

# # Test on the specific frames where you know balls are detected
# results = model.predict('D:/Programing/Football-Analysis/Input/input_3.mp4', save=True, conf=0.1)

# print("\n=== FRAMES 4-6 DETECTIONS ===")
# for frame_idx in [4, 5, 6]:
#     if frame_idx < len(results):
#         print(f"\nFrame {frame_idx}:")
#         if results[frame_idx].boxes is not None:
#             for box in results[frame_idx].boxes:
#                 class_id = int(box.cls.item())
#                 confidence = box.conf.item()
#                 class_name = model.names[class_id]
#                 bbox = box.xyxy[0].tolist()
#                 print(f"  Class: '{class_name}' (ID: {class_id}), Conf: {confidence:.3f}, BBox: {bbox}")