from ultralytics import YOLO

model = YOLO('models/best_v8x.pt')

results = model.predict('D:\Programing\Object Detection\Input\input_4.mp4', save= True)
print(results[0])
print("..................................................................................................................")

for box in results[0].boxes:
    print(box)