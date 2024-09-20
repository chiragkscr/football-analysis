from ultralytics import YOLO

model = YOLO('yolov8x.pt')

model.track("input_vidoes/08fd33_4.mp4")