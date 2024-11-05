import cv2

from ultralytics import solutions

def rescaleFrame(frame, scale=0.75):
    width = 640# int(frame.shape[1] * scale)
    height = 480# int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame,dimensions, interpolation = cv2.INTER_AREA)


cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Define region points
line_points = [(400, 20), (400, 1080)]
region_points = [(370, 0), (370, 480), (270, 480), (270, 0)]
horizontal_line = [(20, 800), (1920, 800)]

# Video writer
video_writer = cv2.VideoWriter("Output_Videos/object_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="Models/epoch_100_best.pt",
    device='gpu'
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    try:
        im0 = rescaleFrame(im0)
    except:
        print("End of the video")
        break
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    total_count = counter.in_count + counter.out_count
    cv2.putText(im0, f"Total count: {total_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print(f"Total count: {total_count}")
    print(f"w: {w}, h: {h}")
    #cv2.imshow(im0)
    video_writer.write(im0)
    if cv2.waitKey(15) & 0xFF == 27:
        break

cap.release()
video_writer.release()
print(fps)
print(f"w: {w}, h: {h}")
cv2.destroyAllWindows()