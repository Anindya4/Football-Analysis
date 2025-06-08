from ultralytics import YOLO

model = YOLO('models/best_v8x.pt')

results = model.predict('D:\Programing\Football-Analysis\Input\input_3.mp4', save= True)
print(results[0])
print("..................................................................................................................")

for box in results[0].boxes:
    print(box)