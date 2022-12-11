import mediapipe as mp
import time
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('model_final.h5')

mpHands = mp.solutions.hands

mhands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

translate = {
    -1: 'No Gesture',
    2: 'like',
    # 3: 'One',
    # 1: 'One',
    # 29: 'One',
    13: 'Hi',
    21: 'Hi',
    22: 'Hi',
    6: 'Stop',
    20: 'Stop',
    #     5: 'A lot',
    #     23: 'A lot',
    #     4: 'few',
    #     26: 'few',
}

cap = cv2.VideoCapture(0)
l = []
t = time.perf_counter()

colors = []

gesture = translate[-1]
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = mhands.process(imgRGB)

    hands = []
    rects = []
    if results.multi_hand_landmarks:
        for handLMs in results.multi_hand_landmarks:
            x0, y0, x1, y1 = 1, 1, 0, 0
            for lm in handLMs.landmark:
                # redddddd BRG red last
                color = img[int(lm.y), int(lm.x)]
                colors.append(color)

                if x0 >= lm.x:
                    x0 = lm.x
                if y0 >= lm.y:
                    y0 = lm.y
                if x1 <= lm.x:
                    x1 = lm.x
                if y1 <= lm.y:
                    y1 = lm.y
            p0 = x0, y0 = (int(x0 * w) - 20, int(y0 * h) - 20)
            p1 = x1, y1 = (int(x1 * w) + 20, int(y1 * h) + 20)

            y1 += 80
            w, h = x1 - x0, y1 - y0

            if w > h:
                yadd = (w - h) // 2
                y0 = int(y0 - yadd)
                y1 = int(y1 + yadd)

            elif w < h:
                xadd = (h - w) // 2
                x0 = int(x0 - xadd)
                x1 = int(x1 + xadd)

            hands.append(img[y0:y1 + 1, x0:x1 + 1])
            rects.append((p0, p1))

            cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
        try:
            for hand in hands:
                himg = hand
                b, g, r = cv2.split(himg)
                colors = colors[:-10]
                bgr_mean = np.mean(colors, 0).astype(np.uint8).reshape(3)

                th = cv2.inRange(r, int(bgr_mean[2] - 100), int(bgr_mean[2] + 100))
                th = th & cv2.inRange(g, int(bgr_mean[0] - 100), int(bgr_mean[0] + 100))

                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))
                th = cv2.resize(th, (64, 64))

        except:
            ...
        himg = hand
        try:
            himg = cv2.resize(himg, (64, 64))
        except:
            continue

        himg[np.where(th == 0)] = [100, 100, 100]
        himg = cv2.cvtColor(himg, cv2.COLOR_BGR2GRAY)
        himg = cv2.merge((himg, himg, himg))

        res = model(himg.reshape(-1, 64, 64, 3))
        am = np.argmax(res)
        l.append(am)

    try:
        if time.perf_counter() - t > 0.5:
            l = [translate[i] for i in l if i in translate] + [translate[-1]]
            gesture = max(l, key=l.count)
            l = []
            t = time.perf_counter()

    except:
        ...

    h, w, c = img.shape
    cv2.putText(img, gesture, (int(w / 2), int(h * 8 / 10)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
