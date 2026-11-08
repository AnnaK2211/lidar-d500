import csv
import queue
import struct
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import serial
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


PORT = '/dev/ttyUSB0'
BAUDRATE = 230400
SERIAL_TIMEOUT = 1

DATA_DIR = Path.home() / 'lidar_d500' / 'data'
SHOT_DIR = Path.home() / 'lidar_d500' / 'screenshots'
EMPTY_FILE = DATA_DIR / 'scan_20261106_103616.csv'

PACKET_HEADER = 0x54
PACKET_VER_LEN = 0x2C
PACKET_REST = 46
START_OFFSET = 3
END_OFFSET = 41
POINTS_OFFSET = 5
POINT_SIZE = 3
POINTS_PER_PACKET = 12
ANGLE_SCALE = 100
FULL_CIRCLE = 360
ANGLE_BINS = 360
MIN_DIST_MM = 50
MAX_DIST_MM = 8000
MM_IN_M = 1000

QUEUE_SIZE = 8
STAMPS_WINDOW = 20
QUEUE_TIMEOUT = 0.05

SIZE = 800
CENTER = SIZE // 2
PX = 1000
RING_RADII = (0.1, 0.2, 0.3, 0.4)

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THR = 15
HOUGH_LEN = 15
HOUGH_GAP = 20
POINT_RADIUS = 2
DILATE_KERNEL = 3
WALL_MASK_THICK = 11
GROUP_ANGLE_DEG = 10
GROUP_OFFSET_PX = 15
MIN_CORNER_DEG = 45
RIGHT_ANGLE = np.pi / 2
EPSILON = 1e-9
CORNER_TOL_DEG = 20

EPS_MM = 10
MIN_SAMPLES = 3
Z_X10 = 30
Z_SCALE = 10
PERSIST = 2
WALL_TOL = 0.015
STATIC_RATIO = 0.5
BG_SCANS = 10
MIN_SIGMA = 0.01
NEIGHBORS = (-1, 0, 1)
NOISE = -1

TRACKBARS = (
    ('eps mm', EPS_MM, 100),
    ('min_samples', MIN_SAMPLES, 10),
    ('z x10', Z_X10, 80),
    ('persist', PERSIST, 5),
    ('hough thr', HOUGH_THR, 150),
    ('min len', HOUGH_LEN, 300),
    ('max gap', HOUGH_GAP, 100)
)

COLOR_POINTS = (255, 128, 0)
COLOR_WALL = (0, 255, 0)
COLOR_WALL_FULL = (0, 140, 0)
COLOR_OBJECT = (0, 140, 255)
COLOR_NEW = (0, 0, 255)
COLOR_GONE = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_GRID = (70, 70, 70)
COLOR_GRID_TEXT = (120, 120, 120)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN = 'LiDAR D500'
WHITE = 255

scan_queue = queue.Queue(maxsize=QUEUE_SIZE)
stamps = deque(maxlen=STAMPS_WINDOW)
stop = threading.Event()


def to_xy(ang, dist):
    rad = np.radians(ang)
    return np.column_stack((dist * np.sin(rad), dist * np.cos(rad)))


def to_pixels(xy):
    return np.column_stack((
        CENTER + xy[:, 0] * PX,
        CENTER - xy[:, 1] * PX
    )).astype(int)


def load_wall_tree():
    if not EMPTY_FILE.exists():
        return None

    data = pd.read_csv(EMPTY_FILE)
    data = data[data['dist_mm'].between(MIN_DIST_MM, MAX_DIST_MM)]
    return cKDTree(to_xy(
        data['angle_deg'].to_numpy(),
        data['dist_mm'].to_numpy() / MM_IN_M
    ))


def push(scan):
    stamps.append(time.time())
    while True:
        try:
            scan_queue.put_nowait(scan)
            break
        except queue.Full:
            try:
                scan_queue.get_nowait()
            except queue.Empty:
                pass


def lidar_reader():
    ang, dist, previous = [], [], None

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=SERIAL_TIMEOUT) as ser:
            while not stop.is_set():
                if ser.read(1) != bytes([PACKET_HEADER]):
                    continue

                packet = ser.read(PACKET_REST)
                if len(packet) != PACKET_REST or packet[0] != PACKET_VER_LEN:
                    continue

                start = struct.unpack_from('<H', packet, START_OFFSET)[0] / ANGLE_SCALE
                end = struct.unpack_from('<H', packet, END_OFFSET)[0] / ANGLE_SCALE

                if previous is not None and start < previous and ang:
                    push((np.array(ang), np.array(dist) / MM_IN_M))
                    ang, dist = [], []

                previous = start
                step = ((end - start) % FULL_CIRCLE) / (POINTS_PER_PACKET - 1)

                for i in range(POINTS_PER_PACKET):
                    value = struct.unpack_from('<H', packet, POINTS_OFFSET + POINT_SIZE * i)[0]
                    if MIN_DIST_MM <= value <= MAX_DIST_MM:
                        ang.append((start + step * i) % FULL_CIRCLE)
                        dist.append(value)

    except serial.SerialException as error:
        print('LiDAR error:', error)
        stop.set()


def find_walls(px, py, threshold, min_len, gap):
    points = np.zeros((SIZE, SIZE), np.uint8)
    for x, y in zip(px, py):
        cv2.circle(points, (int(x), int(y)), POINT_RADIUS, WHITE, cv2.FILLED)

    points = cv2.dilate(points, np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8))
    lines = cv2.HoughLinesP(
        points, HOUGH_RHO, HOUGH_THETA, threshold,
        minLineLength=min_len, maxLineGap=gap
    )
    if lines is None:
        return []
    return [tuple(map(int, line)) for line in lines.reshape(-1, 4)]


def group_lines(lines):
    groups = []

    for segment in lines:
        x1, y1, x2, y2 = segment
        angle = np.arctan2(y2 - y1, x2 - x1) % np.pi
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        for group in groups:
            gx1, gy1, gx2, gy2, group_angle, members = group
            diff = abs(angle - group_angle)
            diff = min(diff, np.pi - diff)
            offset = abs(
                (gx2 - gx1) * (gy1 - my) -
                (gx1 - mx) * (gy2 - gy1)
            ) / (np.hypot(gx2 - gx1, gy2 - gy1) + EPSILON)

            if diff < np.radians(GROUP_ANGLE_DEG) and offset < GROUP_OFFSET_PX:
                members.append(segment)
                break
        else:
            groups.append((x1, y1, x2, y2, angle, [segment]))

    return groups


def wall_extents(groups):
    fits = []

    for group in groups:
        points = np.array([
            point
            for line in group[5]
            for point in ((line[0], line[1]), (line[2], line[3]))
        ], np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01
        ).ravel()
        direction = np.array([vx, vy])
        origin = np.array([x0, y0])
        t = (points - origin) @ direction
        fits.append((origin, direction, t.min(), t.max()))

    result = []
    for i, (origin, direction, lo, hi) in enumerate(fits):
        for j, (other, vector, _, _) in enumerate(fits):
            if i == j or abs(direction @ vector) > np.sin(np.radians(CORNER_TOL_DEG)):
                continue

            den = direction[0] * vector[1] - direction[1] * vector[0]
            if abs(den) < EPSILON:
                continue

            s = (
                (other[0] - origin[0]) * vector[1] -
                (other[1] - origin[1]) * vector[0]
            ) / den
            point = origin + s * direction

            if 0 <= point[0] < SIZE and 0 <= point[1] < SIZE:
                lo, hi = min(lo, s), max(hi, s)

        a, b = origin + lo * direction, origin + hi * direction
        result.append((int(a[0]), int(a[1]), int(b[0]), int(b[1])))

    return result


def wall_angle(groups):
    angles = [
        abs(groups[i][4] - groups[j][4])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    ]
    if not angles:
        return None

    angle = min(angles, key=lambda value: abs(RIGHT_ANGLE - min(value, np.pi - value)))
    angle = min(angle, np.pi - angle)
    return np.degrees(angle) if angle > np.radians(MIN_CORNER_DEG) else None


def cluster_boxes(xy, pixels, eps, min_samples):
    if len(xy) < min_samples:
        return [], len(xy)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    boxes = [
        cv2.boundingRect(pixels[labels == label].astype(np.int32))
        for label in set(labels) - {NOISE}
    ]
    return boxes, int(np.sum(labels == NOISE))


def make_background(buffer):
    bins = np.concatenate([item[0] for item in buffer])
    distances = np.concatenate([item[1] for item in buffer])
    mean = np.full(ANGLE_BINS, np.nan)
    sigma = np.full(ANGLE_BINS, np.nan)

    for angle_bin in range(ANGLE_BINS):
        values = distances[bins == angle_bin]
        if len(values):
            mean[angle_bin] = values.mean()
            sigma[angle_bin] = max(values.std(), MIN_SIGMA)

    return mean, sigma


def changes(ang, dist, background, threshold):
    mean, sigma = background
    bins = np.floor(ang).astype(int) % ANGLE_BINS

    z = np.stack([
        (dist - mean[(bins + shift) % ANGLE_BINS]) /
        sigma[(bins + shift) % ANGLE_BINS]
        for shift in NEIGHBORS
    ])
    z = np.where(np.isnan(z), z[1], z)

    new = np.isnan(z[1]) | (z.max(axis=0) < -threshold)
    gone = z.min(axis=0) > threshold
    return new, gone, mean[bins]


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def draw_box(img, bounds, pad, color, label, thickness):
    x, y, w, h = bounds
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(SIZE - 1, x + w + pad), min(SIZE - 1, y + h + pad)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img, label, (x1, max(y1 - 4, 12)),
        FONT, 0.45, color, 1
    )


def draw(ang, dist, params, state):
    img = np.zeros((SIZE, SIZE, 3), np.uint8)

    for radius in RING_RADII:
        r = int(radius * PX)
        cv2.circle(img, (CENTER, CENTER), r, COLOR_GRID, 1)
        cv2.putText(
            img, f'{radius:.1f} m',
            (CENTER + r + 3, CENTER - 3), FONT, 0.4, COLOR_GRID_TEXT, 1
        )

    cv2.drawMarker(img, (CENTER, CENTER), COLOR_TEXT, cv2.MARKER_TRIANGLE_UP, 12, 2)

    xy = to_xy(ang, dist)
    pixels = to_pixels(xy)
    valid = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < SIZE) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < SIZE)
    )
    ang, dist, xy, pixels = ang[valid], dist[valid], xy[valid], pixels[valid]
    px, py = pixels[:, 0], pixels[:, 1]
    img[py, px] = COLOR_POINTS

    if state['tree'] is not None:
        static = state['tree'].query(xy)[0] <= WALL_TOL
        use_tree = len(static) and static.mean() > STATIC_RATIO
    else:
        use_tree = False

    if use_tree:
        lines = find_walls(px[static], py[static], params['thr'], params['len'], params['gap'])
    else:
        lines = find_walls(px, py, params['thr'], params['len'], params['gap'])
        mask = np.zeros((SIZE, SIZE), np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(mask, (x1, y1), (x2, y2), WHITE, WALL_MASK_THICK)
        static = mask[py, px] > 0

    groups = group_lines(lines)
    for x1, y1, x2, y2 in wall_extents(groups):
        cv2.line(img, (x1, y1), (x2, y2), COLOR_WALL_FULL, 1)
    for line in lines:
        cv2.line(img, line[:2], line[2:], COLOR_WALL, 2)

    objects, noise = cluster_boxes(
        xy[~static], pixels[~static], params['eps'], params['ms']
    )

    new_boxes, gone_boxes = [], []
    n_new = n_gone = 0

    if state['bg'] is not None:
        new, gone, old_dist = changes(ang, dist, state['bg'], params['z'])

        new_boxes, _ = cluster_boxes(
            xy[new], pixels[new], params['eps'], params['ms']
        )

        gone_dist = old_dist[gone]
        gone_angles = ang[gone]
        finite = np.isfinite(gone_dist)
        old_xy = to_xy(gone_angles[finite], gone_dist[finite])
        old_pixels = to_pixels(old_xy)

        inside = (
            (old_pixels[:, 0] >= 0) & (old_pixels[:, 0] < SIZE) &
            (old_pixels[:, 1] >= 0) & (old_pixels[:, 1] < SIZE)
        )
        gone_boxes, _ = cluster_boxes(
            old_xy[inside], old_pixels[inside], params['eps'], params['ms']
        )

        state['streak']['new'] = state['streak']['new'] + 1 if new_boxes else 0
        state['streak']['gone'] = state['streak']['gone'] + 1 if gone_boxes else 0

        if state['streak']['new'] >= params['persist']:
            n_new = len(new_boxes)
        else:
            new_boxes = []

        if state['streak']['gone'] >= params['persist']:
            n_gone = len(gone_boxes)
        else:
            gone_boxes = []

        state['alarms'] += bool(new_boxes or gone_boxes)

    for bounds in objects:
        if not any(overlaps(bounds, b) for b in new_boxes + gone_boxes):
            draw_box(img, bounds, 4, COLOR_OBJECT, 'obj', 1)

    for bounds in new_boxes:
        draw_box(img, bounds, 7, COLOR_NEW, 'NEW', 2)
    for bounds in gone_boxes:
        draw_box(img, bounds, 7, COLOR_GONE, 'REMOVED', 2)

    if state['buffer'] is not None:
        status = f'collecting {len(state["buffer"])} / {BG_SCANS}'
    else:
        status = 'fixed' if state['bg'] is not None else 'none'

    times = np.diff(list(stamps))
    rps = 1 / times.mean() if len(times) else 0
    angle = wall_angle(groups)

    info = [
        f'{rps:.1f} rev/s   {len(ang)} points',
        f'walls {len(groups)}   objects {len(objects)}   changes {n_new} new / {n_gone} removed   noise {noise}',
        f'wall angle: {angle:.1f} deg' if angle is not None else 'wall angle: -',
        f'background: {status}   alarm frames: {state["alarms"]}',
        'b fix   r reset   s save   q quit'
    ]
    for i, line in enumerate(info):
        cv2.putText(img, line, (10, 20 + 20 * i), FONT, 0.5, COLOR_TEXT, 1)

    legend = [
        ('lidar points', COLOR_POINTS),
        ('walls (Hough)', COLOR_WALL),
        ('walls completed', COLOR_WALL_FULL),
        ('objects (DBSCAN)', COLOR_OBJECT),
        ('new', COLOR_NEW),
        ('removed', COLOR_GONE)
    ]
    for i, (label, color) in enumerate(legend):
        y = SIZE - 115 + 18 * i
        cv2.line(img, (10, y), (35, y), color, 3)
        cv2.putText(img, label, (42, y + 5), FONT, 0.45, COLOR_TEXT, 1)

    return img


def get_params():
    value = lambda name: cv2.getTrackbarPos(name, WIN)
    return {
        'eps': max(value('eps mm'), 1) / MM_IN_M,
        'ms': max(value('min_samples'), 1),
        'z': value('z x10') / Z_SCALE,
        'persist': max(value('persist'), 1),
        'thr': max(value('hough thr'), 1),
        'len': value('min len'),
        'gap': value('max gap')
    }


def save_scan(ang, dist, img):
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv2.imwrite(str(SHOT_DIR / f'screen_{timestamp}.png'), img)

    with open(DATA_DIR / f'scan_{timestamp}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['scan', 'angle_deg', 'dist_mm'])
        writer.writerows((0, a, round(d * MM_IN_M)) for a, d in zip(ang, dist))

    print(f'saved: screen_{timestamp}.png (screenshots) and scan_{timestamp}.csv (data)')


def main():
    state = {
        'tree': load_wall_tree(),
        'bg': None,
        'buffer': None,
        'alarms': 0,
        'streak': {'new': 0, 'gone': 0}
    }
    print('wall reference:', 'file' if state['tree'] is not None else 'none (Hough mask)')

    threading.Thread(target=lidar_reader, daemon=True).start()
    cv2.namedWindow(WIN)

    for name, value, maximum in TRACKBARS:
        cv2.createTrackbar(name, WIN, value, maximum, lambda _: None)

    frame = np.zeros((SIZE, SIZE, 3), np.uint8)
    last_scan = None

    try:
        while not stop.is_set():
            try:
                ang, dist = scan_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                ang = None

            if ang is not None:
                last_scan = (ang, dist)

                if state['buffer'] is not None:
                    bins = np.floor(ang).astype(int) % ANGLE_BINS
                    state['buffer'].append((bins, dist))
                    if len(state['buffer']) >= BG_SCANS:
                        state['bg'] = make_background(state['buffer'])
                        state['buffer'] = None
                        state['alarms'] = 0

                frame = draw(ang, dist, get_params(), state)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord('b'):
                state['bg'], state['buffer'], state['alarms'] = None, [], 0
                state['streak'] = {'new': 0, 'gone': 0}
            elif key == ord('r'):
                state['bg'], state['buffer'], state['alarms'] = None, None, 0
                state['streak'] = {'new': 0, 'gone': 0}
            elif key == ord('s') and last_scan is not None:
                save_scan(last_scan[0], last_scan[1], frame)

    finally:
        stop.set()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
