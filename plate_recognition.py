# -*- coding: utf-8 -*-
"""
GRZ Recognition — fast-alpr + MQTT + GPIO
Raspberry Pi, без GUI (headless)
Setup:  pip install fast-alpr pillow paho-mqtt onnxruntime RPi.GPIO
Run:    python3 claude.py
Stop:   Ctrl+C
grz.txt — whitelist, one plate per line, any case/language
"""

import cv2
import os
import re
import time
import json
import threading
from datetime import datetime
from collections import deque, Counter

from fast_alpr import ALPR
import paho.mqtt.client as mqtt_lib
from PIL import Image, ImageFont
import numpy as np

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[GPIO] RPi.GPIO не найден — GPIO отключён")

# =================== SETTINGS ===================
SCAN_INTERVAL    = 2.0
OPEN_DISPLAY_SEC = 3.0
COOLDOWN_SEC     = 10.0
SAVE_DIR         = "detected_plates"
LOG_FILE         = "scanned_plates.txt"
WHITELIST_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grz.txt")
WHITELIST_RELOAD = 30.0

# ── GPIO ─────────────────────────────────────
SERVO_PIN      = 18    # SG90 сигнал на GPIO 18 (физический пин 12)
SERVO_OPEN     = 7.5   # duty cycle для 90° (открыто)
SERVO_CLOSED   = 2.5   # duty cycle для 0° (закрыто)
SERVO_FREQ     = 50    # частота PWM для сервопривода
RF_PIN         = 27    # MX-RM-5V DATA на GPIO 27 (физический пин 13)
RF_MIN_PULSES  = 50    # минимум переходов для детекции нажатия
TRIG_PIN       = 23    # HC-SR04 TRIG на GPIO 23 (физический пин 16)
ECHO_PIN       = 24    # HC-SR04 ECHO на GPIO 24 (физический пин 18)
SONAR_MAX_DIST = 200   # см — дальше этого считаем "пусто"
SONAR_WAIT_CAR = 30.0  # сек — ждём машину после открытия
SONAR_CLOSE_DELAY = 5.0  # сек — ждём после проезда машины

# ── MQTT (Home Assistant) ────────────────────
MQTT_ENABLED         = True
MQTT_HOST            = "127.0.0.1"
MQTT_PORT            = 1883
MQTT_USER            = ""
MQTT_PASSWORD        = ""
MQTT_TOPIC_EVENT     = "grz/event"
MQTT_TOPIC_MANUAL    = "grz/manual"     # ручное открытие с HA
MQTT_TOPIC_CLOSE     = "grz/close"      # ручное закрытие с HA
MQTT_TOPIC_WHITELIST = "grz/whitelist"  # публикация списка
MQTT_TOPIC_CMD       = "grz/cmd"        # команды: add/delete/list
# ================================================

LATIN_TO_CYR = str.maketrans({
    'A':'А','B':'В','E':'Е','K':'К','M':'М','H':'Н',
    'O':'О','P':'Р','C':'С','T':'Т','Y':'У','X':'Х',
    'a':'А','b':'В','e':'Е','k':'К','m':'М','h':'Н',
    'o':'О','p':'Р','c':'С','t':'Т','y':'У','x':'Х',
})

os.makedirs(SAVE_DIR, exist_ok=True)


# ─── GPIO / Servo ───

servo_pwm = None

def gpio_init():
    global servo_pwm
    if not GPIO_AVAILABLE:
        return
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(SERVO_CLOSED)
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # отключаем сигнал чтобы не дергался
    GPIO.setup(RF_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.1)
    print(f"[GPIO] Сервопривод инициализирован на пине {SERVO_PIN}, RF на пине {RF_PIN}, УЗ датчик TRIG={TRIG_PIN} ECHO={ECHO_PIN}")

def sonar_distance() -> float:
    """Возвращает расстояние в см или 999 при ошибке."""
    if not GPIO_AVAILABLE:
        return 999.0
    try:
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        t = time.time()
        while GPIO.input(ECHO_PIN) == 0:
            if time.time() - t > 0.1: return 999.0
        t1 = time.time()
        while GPIO.input(ECHO_PIN) == 1:
            if time.time() - t1 > 0.1: return 999.0
        t2 = time.time()
        return (t2 - t1) * 34300 / 2
    except:
        return 999.0

def gpio_led(state: bool):
    global servo_pwm
    if not GPIO_AVAILABLE or servo_pwm is None:
        return
    if state:
        servo_pwm.ChangeDutyCycle(SERVO_OPEN)
        print("[SERVO] Открыто (90°)")
    else:
        servo_pwm.ChangeDutyCycle(SERVO_CLOSED)
        print("[SERVO] Закрыто (0°)")
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)  # отключаем сигнал

def gpio_cleanup():
    global servo_pwm
    if not GPIO_AVAILABLE:
        return
    if servo_pwm:
        servo_pwm.ChangeDutyCycle(SERVO_CLOSED)
        time.sleep(0.5)
        servo_pwm.stop()
    GPIO.cleanup()


# ─── RF приёмник (отдельный поток) ───

def rf_listener():
    global manual_open_requested, manual_close_requested
    if not GPIO_AVAILABLE:
        return
    print(f"[RF] Слушаю пульт на пине {RF_PIN}")
    last_val = GPIO.input(RF_PIN)
    burst_changes = 0
    burst_start = None
    last_change = time.time()

    while True:
        val = GPIO.input(RF_PIN)
        now = time.time()

        if val != last_val:
            last_val = val
            if burst_start is None:
                burst_start = now
            burst_changes += 1
            last_change = now

        # Если пауза > 0.15 сек — burst закончился
        if burst_start is not None and now - last_change > 0.15:
            duration = last_change - burst_start
            # Реальный сигнал: 50-300 переходов за 0.05-1.0 сек
            if RF_MIN_PULSES <= burst_changes <= 300 and 0.05 <= duration <= 1.0:
                print(f"[RF] Кнопка нажата ({burst_changes} переходов, {duration:.2f}с)")
                manual_open_requested = True
            burst_changes = 0
            burst_start = None

        time.sleep(0.001)


def normalize(text: str) -> str:
    s = re.sub(r'[^А-ЯA-Z0-9а-яa-z]', '', text)
    return s.translate(LATIN_TO_CYR).upper()


def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    row = list(range(len(b)+1))
    for c1 in a:
        new_row = [row[0]+1]
        for j, c2 in enumerate(b):
            new_row.append(min(row[j+1]+1, new_row[-1]+1, row[j]+(c1!=c2)))
        row = new_row
    return row[-1]

def match_whitelist(plate, whitelist, max_dist=2):
    if plate in whitelist:
        return plate, True
    best_plate, best_dist = None, max_dist + 1
    for w in whitelist:
        if abs(len(plate) - len(w)) > 2:
            continue
        d = levenshtein(plate, w)
        if d < best_dist:
            best_dist, best_plate = d, w
    if best_plate and best_dist <= max_dist:
        return best_plate, False
    return None, False


# ─── whitelist ───

def load_whitelist():
    if not os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, 'w', encoding='utf-8') as f:
            f.write("# По одному номеру на строку:\n# У777ЕТ123\n# А001ВС77\n")
        print(f"[INFO] Создан {WHITELIST_FILE}")
        return set()
    plates = set()
    with open(WHITELIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            s = normalize(line.strip())
            if s and not s.startswith('#'):
                plates.add(s)
    print(f"[INFO] Whitelist ({len(plates)}): {plates}")
    return plates

def save_whitelist(plates: set):
    with open(WHITELIST_FILE, 'w', encoding='utf-8') as f:
        f.write("# По одному номеру на строку:\n")
        for p in sorted(plates):
            f.write(p + "\n")
    print(f"[INFO] Whitelist сохранён: {plates}")


# ─── MQTT ───

# Глобальные переменные
manual_open_requested  = False
manual_close_requested = False
manual_last_time       = 0.0   # кулдаун ручного открытия
whitelist_cmd = None  # {"cmd": "add"/"delete", "plate": "..."}

def on_mqtt_message(client, userdata, msg):
    global manual_open_requested, manual_close_requested, whitelist_cmd, manual_last_time
    topic   = msg.topic
    payload = msg.payload.decode('utf-8', errors='ignore').strip()
    if topic == MQTT_TOPIC_MANUAL and payload == "OPEN":
        now = time.time()
        if now - manual_last_time > 5.0:
            manual_last_time = now
            print("[MQTT] Получена команда ручного открытия!")
            manual_open_requested = True
        else:
            print("[MQTT] Команда проигнорирована (кулдаун)")
    elif topic == MQTT_TOPIC_CLOSE and payload == "CLOSE":
        print("[MQTT] Получена команда ручного закрытия!")
        manual_close_requested = True
    elif topic == MQTT_TOPIC_CMD:
        try:
            data = json.loads(payload)
            cmd  = data.get("cmd", "")
            if cmd in ("add", "delete", "list"):
                whitelist_cmd = data
                print(f"[MQTT] Команда: {data}")
        except Exception as e:
            print(f"[MQTT] Ошибка парсинга команды: {e}")

def mqtt_connect():
    if not MQTT_ENABLED:
        return None
    try:
        client = mqtt_lib.Client(client_id="grz_recognition",
                                  callback_api_version=mqtt_lib.CallbackAPIVersion.VERSION2,
                                  clean_session=True)
        if MQTT_USER:
            client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        client.on_message = on_mqtt_message
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.subscribe(MQTT_TOPIC_MANUAL)
        client.subscribe(MQTT_TOPIC_CLOSE)
        client.subscribe(MQTT_TOPIC_CMD)
        client.loop_start()
        print(f"[MQTT] Подключён к {MQTT_HOST}:{MQTT_PORT}")
        print(f"[MQTT] Слушаю топики: {MQTT_TOPIC_MANUAL}, {MQTT_TOPIC_CLOSE}, {MQTT_TOPIC_CMD}")
        return client
    except Exception as e:
        print(f"[MQTT] Ошибка: {e} — продолжаем без MQTT")
        return None

def mqtt_publish(client, topic, payload: dict):
    if client is None:
        return
    try:
        client.publish(topic, json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print(f"[MQTT] Ошибка публикации: {e}")


# ─── file log ───

def log_to_file(plate: str, in_whitelist: bool):
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "WHITELIST" if in_whitelist else "UNKNOWN"
    line   = f"{ts}  {plate}  {status}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"  >> {line.strip()}")


# ─── main ───

def main():
    global manual_open_requested, manual_close_requested, whitelist_cmd
    gpio_init()
    whitelist = load_whitelist()

    print("Загрузка ALPR модели...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="global-plates-mobile-vit-v2-model",
    )
    print("Готово!")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Ошибка камеры"); return

    mqtt_client       = mqtt_connect()
    vote_buf          = deque(maxlen=5)
    last_scan         = 0.0
    last_logged       = {}
    last_wl_reload    = time.time()
    gate_open         = False       # шлагбаум открыт?
    gate_opened_at    = 0.0         # когда открылся
    car_detected      = False       # машина была обнаружена?
    car_passed_at     = 0.0         # когда машина проехала
    last_sonar        = 0.0

    # Запускаем RF в отдельном потоке
    rf_thread = threading.Thread(target=rf_listener, daemon=True)
    rf_thread.start()

    mqtt_publish(mqtt_client, MQTT_TOPIC_WHITELIST, {"plates": sorted(whitelist)})
    print("Скрипт запущен. Остановка: Ctrl+C")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()

            # Логика закрытия шлагбаума через УЗ датчик
            if gate_open:
                if now - last_sonar >= 0.3:
                    last_sonar = now
                    dist = sonar_distance()

                    if dist < SONAR_MAX_DIST:
                        # Машина под шлагбаумом
                        if not car_detected:
                            car_detected = True
                            print(f"[SONAR] Машина обнаружена ({dist:.0f} см)")
                        car_passed_at = 0.0
                    else:
                        # Путь свободен
                        if car_detected and car_passed_at == 0.0:
                            car_passed_at = now
                            print(f"[SONAR] Машина проехала, закрываем через {SONAR_CLOSE_DELAY} сек")

                    # Закрываем если машина проехала и прошло 5 сек
                    if car_detected and car_passed_at > 0 and now - car_passed_at >= SONAR_CLOSE_DELAY:
                        print("[SONAR] Закрываем шлагбаум (машина проехала)")
                        gpio_led(False)
                        gate_open = False
                        car_detected = False
                        car_passed_at = 0.0

                    # Закрываем если за 30 сек машина не появилась
                    elif not car_detected and now - gate_opened_at >= SONAR_WAIT_CAR:
                        print("[SONAR] Закрываем шлагбаум (машина не появилась за 30 сек)")
                        gpio_led(False)
                        gate_open = False

            # Обработка команд управления whitelist
            if whitelist_cmd:
                cmd  = whitelist_cmd.get("cmd")
                plate = normalize(whitelist_cmd.get("plate", ""))
                whitelist_cmd = None
                if cmd == "add" and plate:
                    whitelist.add(plate)
                    save_whitelist(whitelist)
                    print(f"[WL] Добавлен: {plate}")
                    mqtt_publish(mqtt_client, MQTT_TOPIC_WHITELIST, {"plates": sorted(whitelist)})
                elif cmd == "delete" and plate:
                    whitelist.discard(plate)
                    save_whitelist(whitelist)
                    print(f"[WL] Удалён: {plate}")
                    mqtt_publish(mqtt_client, MQTT_TOPIC_WHITELIST, {"plates": sorted(whitelist)})
                elif cmd == "list":
                    mqtt_publish(mqtt_client, MQTT_TOPIC_WHITELIST, {"plates": sorted(whitelist)})

            # Ручное открытие с Home Assistant / RF пульт
            if manual_open_requested and not gate_open:
                manual_open_requested = False
                print("OPEN (вручную)")
                gpio_led(True)
                gate_open = True
                gate_opened_at = now
                car_detected = False
                car_passed_at = 0.0
                log_to_file("MANUAL", True)
            else:
                manual_open_requested = False

            # Ручное закрытие (кнопка B пульта)
            if manual_close_requested and gate_open:
                manual_close_requested = False
                print("CLOSE (вручную)")
                gpio_led(False)
                gate_open = False
            else:
                manual_close_requested = False

            # Автоперезагрузка whitelist
            if now - last_wl_reload >= WHITELIST_RELOAD:
                new_wl = load_whitelist()
                if new_wl != whitelist:
                    whitelist = new_wl
                    print(f"[INFO] Whitelist обновлён: {whitelist}")
                    mqtt_publish(mqtt_client, MQTT_TOPIC_WHITELIST, {"plates": sorted(whitelist)})
                last_wl_reload = now

            # Сканирование
            if now - last_scan >= SCAN_INTERVAL:
                last_scan = now
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                results   = alpr.predict(frame)
                print(f"\n[{ts}] найдено: {len(results)}")

                for plate_obj in results:
                    bb     = plate_obj.detection.bounding_box
                    x1, y1 = int(bb.x1), int(bb.y1)
                    x2, y2 = int(bb.x2), int(bb.y2)
                    raw    = plate_obj.ocr.text if plate_obj.ocr else ""
                    conf   = plate_obj.ocr.confidence if plate_obj.ocr else 0.0
                    plate  = normalize(raw)

                    print(f"  raw:'{raw}' -> '{plate}'  conf:{conf:.2f}")

                    if not plate:
                        continue

                    # Voting
                    vote_buf.append(plate)
                    best, best_count = Counter(vote_buf).most_common(1)[0]
                    print(f"  votes: {dict(Counter(vote_buf))}  best:'{best}'x{best_count}")

                    if best_count < 2:
                        continue

                    confirmed = best

                    # Cooldown
                    m, _ = match_whitelist(confirmed, whitelist)
                    cooldown_key = m if m else confirmed
                    if now - last_logged.get(cooldown_key, 0) < COOLDOWN_SEC:
                        continue

                    matched, is_exact = match_whitelist(confirmed, whitelist)
                    in_wl  = matched is not None
                    log_key = matched if matched else confirmed
                    last_logged[log_key] = now

                    log_to_file(confirmed, in_wl)
                    # cv2.imwrite(f"{SAVE_DIR}/{confirmed}_{ts}.jpg", frame[y1:y2, x1:x2])

                    if in_wl:
                        suffix = f" (~{matched})" if not is_exact else ""
                        print(f"OPEN{suffix}")
                        gpio_led(True)
                        gate_open = True
                        gate_opened_at = now
                        car_detected = False
                        car_passed_at = 0.0
                        mqtt_publish(mqtt_client, MQTT_TOPIC_EVENT, {
                            "plate":   confirmed,
                            "matched": matched,
                            "status":  "OPEN",
                            "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    else:
                        print(f"UNKNOWN: {confirmed}")
                        mqtt_publish(mqtt_client, MQTT_TOPIC_EVENT, {
                            "plate":  confirmed,
                            "status": "UNKNOWN",
                            "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        cap.release()
        gpio_cleanup()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("Готово.")


if __name__ == "__main__":
    main()