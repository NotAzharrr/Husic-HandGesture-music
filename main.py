import cv2
import mediapipe as mp
import os
from playsound import playsound
import threading

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)

# Chord and note map
gesture_map = {
    (0,1,0,0,0): ("C Major Chord", "C Note", "Index", "c_majorr.wav", "c_note.wav"),
    (0,1,1,0,0): ("D Minor Chord", "D Note", "Index + Middle", "d_chord.wav", "d_note.wav"),
    (1,0,0,0,0): ("E Minor Chord", "E Note", "Thumb", "e_chord.wav", "e_note.wav"),
    (1,1,0,0,0): ("F Major Chord", "F Note", "Thumb + Index", "f_chord.wav", "f_note.wav"),
    (0,0,0,0,1): ("G Major Chord", "G Note", "Pinky", "g_chord.wav", "g_note.wav"),
    (0,1,0,0,1): ("A Minor Chord", "A Note", "Index + Pinky", "a_chord.wav", "a_note.wav"),
    (1,1,1,1,1): ("B Dim Chord", "B Note", "All Fingers", "b_chord.wav", "b_note.wav"),
}

# Currently playing state to prevent repeated playback
playing = {"Right": None, "Left": None}

# Play sound in a separate thread
def play_sound(file_path):
    if os.path.exists(file_path):
        threading.Thread(target=playsound, args=(file_path,), daemon=True).start()

# Detect which fingers are up
def fingers_up(hand_landmarks, hand_label):
    fingers = []
    # Thumb
    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
    else:
        fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return tuple(int(f) for f in fingers)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            fingers = fingers_up(hand_landmark, hand_label)

            # Get mapped chord and note info
            gesture_info = gesture_map.get(fingers, None)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Show info above hand
            x = int(hand_landmark.landmark[0].x * frame.shape[1])
            y = int(hand_landmark.landmark[0].y * frame.shape[0]) - 30

            if gesture_info:
                chord, note, gesture_name, chord_file, note_file = gesture_info
                if hand_label == "Right":
                    text = f"{hand_label} Hand: {chord} ({gesture_name})"
                    if playing[hand_label] != chord:
                        play_sound(f"sounds/{chord_file}")
                        playing[hand_label] = chord
                else:
                    text = f"{hand_label} Hand: {note} ({gesture_name})"
                    if playing[hand_label] != note:
                        play_sound(f"sounds/{note_file}")
                        playing[hand_label] = note
                cv2.putText(frame, text, (x - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            else:
                playing[hand_label] = None  # Reset if no valid gesture

    # Show the frame
    cv2.imshow("Chord & Note Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
