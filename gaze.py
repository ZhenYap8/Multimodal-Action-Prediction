import cv2
import csv
import os

video_path = "1.mov"
output_csv = "labels.csv"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise ValueError("Could not read FPS from video.")

frame_interval = int(round(fps))
current_frame = 0
clicks = []
data = []


while True:
    user_input = input("Is this video picking up tennis or orange? (t/o): ").strip().lower()
    if user_input == "t":
        video_target = "tennis"
        break
    elif user_input == "o":
        video_target = "orange"
        break
    else:
        print("Invalid input. Please type 't' for tennis or 'o' for orange.")

def click_event(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < 4:
            clicks.append((x, y))

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_event)

print("Click order:")
print("1: gaze (red)")
print("2: hand (blue)")
print("3: tennis ball (green)")
print("4: orange ball (orange)")
print("Controls:")
print("  n = save current frame and go to next sampled frame")
print("  r = reset clicks on current frame")
print("  q = quit and save everything collected so far")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    if not ret:
        print("Reached end of video.")
        break

    while True:
        display = frame.copy()

        if len(clicks) >= 1:
            cv2.circle(display, clicks[0], 8, (0, 0, 255), -1)
            cv2.putText(
                display,
                "gaze",
                (clicks[0][0] + 10, clicks[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        if len(clicks) >= 2:
            cv2.circle(display, clicks[1], 8, (255, 0, 0), -1)
            cv2.putText(
                display,
                "hand",
                (clicks[1][0] + 10, clicks[1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        if len(clicks) >= 3:
            cv2.circle(display, clicks[2], 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                "tennis",
                (clicks[2][0] + 10, clicks[2][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if len(clicks) >= 4:
            cv2.circle(display, clicks[3], 8, (0, 165, 255), -1)
            cv2.putText(
                display,
                "orange",
                (clicks[3][0] + 10, clicks[3][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2
            )

        cv2.putText(
            display,
            f"Frame: {current_frame}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("frame", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicks = []
            print(f"Reset clicks for frame {current_frame}")

        elif key == ord("n"):
            if len(clicks) < 4:
                print("Please make 4 clicks first: gaze, hand, tennis ball, orange ball.")
                continue

            gaze = clicks[0]
            hand = clicks[1]
            tennis = clicks[2]
            orange = clicks[3]

            data.append([
                video_path,
                current_frame,
                gaze[0], gaze[1],
                hand[0], hand[1],
                tennis[0], tennis[1],
                orange[0], orange[1],
                video_target
            ])

            print(
                f"Saved frame {current_frame}: "
                f"gaze={gaze}, hand={hand}, tennis={tennis}, orange={orange}, target={video_target}"
            )

            clicks = []
            current_frame += frame_interval
            break

        elif key == ord("q"):
            print("Quitting and saving collected labels.")
            cap.release()
            cv2.destroyAllWindows()

            file_exists = os.path.isfile(output_csv)

            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "video",
                        "frame",
                        "gaze_x", "gaze_y",
                        "hand_x", "hand_y",
                        "tennis_x", "tennis_y",
                        "orange_x", "orange_y",
                        "target"
                    ])

                writer.writerows(data)

            print(f"Saved labels to {output_csv} (appended)")
            raise SystemExit

cap.release()
cv2.destroyAllWindows()

file_exists = os.path.isfile(output_csv)

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "video",
            "frame",
            "gaze_x", "gaze_y",
            "hand_x", "hand_y",
            "tennis_x", "tennis_y",
            "orange_x", "orange_y",
            "target"
        ])

    writer.writerows(data)

print(f"Saved labels to {output_csv} (appended)")