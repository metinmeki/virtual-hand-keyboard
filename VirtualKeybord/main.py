import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller

# --- Mediapipe Hands Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Keyboard Controller ---
keyboard = Controller()

# --- Video Capture Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 580)

# --- Create window with fixed size ---
cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Virtual Keyboard", 1000, 580)

# --- Colors and Fonts ---
COLOR_BUTTON = (70, 70, 70)
COLOR_HIGHLIGHT = (0, 120, 215)
COLOR_PRESSED = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Button Class ---
class Button:
    def __init__(self, pos, text, size=(70, 70)):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img, is_highlighted=False, is_pressed=False):
        x, y = self.pos
        w, h = self.size
        color = COLOR_BUTTON
        if is_pressed:
            color = COLOR_PRESSED
        elif is_highlighted:
            color = COLOR_HIGHLIGHT

        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)

        text_size = cv2.getTextSize(self.text, FONT, 1.5, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y), FONT, 1.5, COLOR_TEXT, 2)

# --- Keyboard Layout ---
keys_upper = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "SP"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "APR"]
]

keys_lower = [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "CL"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "SP"],
    ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "APR"]
]

# --- Create buttons with padding to avoid clipping ---
def create_buttons(keys):
    btn_list = []
    padding_x = 20
    padding_y = 20
    btn_spacing = 80
    for row_idx, row in enumerate(keys):
        for col_idx, key in enumerate(row):
            pos_x = padding_x + btn_spacing * col_idx
            pos_y = padding_y + btn_spacing * row_idx
            btn_list.append(Button((pos_x, pos_y), key))
    return btn_list

buttons_upper = create_buttons(keys_upper)
buttons_lower = create_buttons(keys_lower)

# --- Helper Functions ---
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

x_vals = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y_vals = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coeffs = np.polyfit(x_vals, y_vals, 2)

def switch_mode(current_mode):
    return 1 - current_mode

# --- Main Loop ---
app_mode = 0  # 0: uppercase, 1: lowercase
delay_counter = 0
typed_text = ""

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)  # Mirror for natural use
    frame = cv2.resize(frame, (1000, 580))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_buttons = buttons_upper if app_mode == 0 else buttons_lower

    # Draw buttons first
    for button in current_buttons:
        button.draw(frame)

    # Draw text input box
    cv2.rectangle(frame, (20, 250), (850, 400), (230, 230, 230), cv2.FILLED)
    cv2.putText(frame, typed_text, (30, 320), FONT, 2, (0, 0, 0), 3)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))

    if landmarks:
        try:
            x5, y5 = landmarks[5][1], landmarks[5][2]
            x17, y17 = landmarks[17][1], landmarks[17][2]
            dist = calculate_distance(x5, y5, x17, y17)

            A, B, C = coeffs
            distance_cm = A * dist ** 2 + B * dist + C

            if 20 < distance_cm < 50:
                x_index_tip, y_index_tip = landmarks[8][1], landmarks[8][2]
                x_index_pip, y_index_pip = landmarks[6][1], landmarks[6][2]
                x_middle_tip, y_middle_tip = landmarks[12][1], landmarks[12][2]

                cv2.circle(frame, (x_index_tip, y_index_tip), 20, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x_middle_tip, y_middle_tip), 20, (255, 0, 255), cv2.FILLED)

                if y_index_pip > y_index_tip:
                    for button in current_buttons:
                        xb, yb = button.pos
                        wb, hb = button.size
                        if xb < x_index_tip < xb + wb and yb < y_index_tip < yb + hb:
                            button.draw(frame, is_highlighted=True)

                            finger_dist = calculate_distance(x_index_tip, y_index_tip, x_middle_tip, y_middle_tip)

                            if finger_dist < 50 and delay_counter == 0:
                                button.draw(frame, is_pressed=True)
                                key = button.text

                                if key == "SP":
                                    typed_text += " "
                                    keyboard.press(" ")
                                elif key == "CL":
                                    typed_text = typed_text[:-1]
                                    keyboard.press('\b')
                                elif key == "APR":
                                    app_mode = switch_mode(app_mode)
                                else:
                                    typed_text += key
                                    keyboard.press(key)

                                delay_counter = 1
                                break
        except Exception:
            pass

    if delay_counter > 0:
        delay_counter += 1
        if delay_counter > 10:
            delay_counter = 0

    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
