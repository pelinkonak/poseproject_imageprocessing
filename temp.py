import cv2
import mediapipe as mp

# Mediapipe pose modülü ve çizim yardımcıları
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark isimleri (Mediapipe Landmark listesinden)
LANDMARK_NAMES = {
    0: "Nose",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle"
    # İstersen daha fazlasını ekleyebilirsin
}

# Video dosya yolu
vidpath = r"C:\Users\pelin\OneDrive\Masaüstü\bad (3).mp4"
  

# Video yakalama nesnesi
vidcap = cv2.VideoCapture(vidpath)
if not vidcap.isOpened():
    print("Video dosyası açılamadı!")
    exit()

# Pencere boyutları
winwidth = 540
winheight = 500

# Mediapipe Pose nesnesi
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        # BGR -> RGB dönüşümü
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Pose tahmini
        results = pose.process(rgb_frame)

        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Eğer landmarklar varsa iskelet çiz ve etiketle
        if results.pose_landmarks:
            # İskeleti çiz
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Landmark koordinatlarını al
            h, w, _ = frame.shape
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id in LANDMARK_NAMES:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Etiketleri çerçeveye yaz
                    cv2.putText(frame, LANDMARK_NAMES[id], (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Görüntüyü pencere boyutuna göre yeniden boyutlandır
        resized_frame = cv2.resize(frame, (winwidth, winheight))

        # Görüntüyü göster
        cv2.imshow('Pose Estimation and Labeling', resized_frame)

        # 'q' ile çıkış
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
vidcap.release()
cv2.destroyAllWindows()