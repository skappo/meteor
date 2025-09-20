#!/usr/bin/env python3
"""
meteora_pipeline_pi5_v1.py

Proof-of-Concept: multi-thread/multiprocess pipeline for continuous night-sky capture,
detection of meteors, 30s event recording via hardware H.264 (ffmpeg), and timelapse stacking.

Designed for Raspberry Pi 5 (but works on other Linux SBCs).

Features:
- capture thread (supports Picamera2 or OpenCV VideoCapture)
- detection worker (frame-differencing + simple morphology) using multiprocessing
- event recorder (saves 30s H264 video via ffmpeg hardware encoder)
- timelapse worker (accumulates N frames, aligns with ECC, saves average stack)
- JSON metadata logging per event/timelapse
- rotating logs and safe shutdown

Usage:
- install dependencies: pip3 install opencv-python-headless numpy psutil pyexiv2
  (note: on Raspberry Pi prefer opencv built with V4L2 and libcamera; picamera2 optional)
- ensure ffmpeg is installed and can access hardware encoder (v4l2m2m or h264_omx depending on distro)
- run: python3 meteora_pipeline_pi5.py --config config.yaml

Caveats:
- This script is a PoC and will need tuning for your camera, lens and site.
- For astrometry/photometry use outputs as candidates and post-process on desktop.


meteora_pipeline_pi5_v2.py

PoC pipeline aggiornato per Raspberry Pi Camera v3 usando Picamera2 per acquisizione,
mantenendo OpenCV per detection, stacking e salvataggio.

| Parametro              | Valore suggerito    | Motivazione                                                    |
| ---------------------- | ------------------- | -------------------------------------------------------------- |
| **Resolution**         | 1920×1080 (Full HD) | Buon compromesso tra dettagli e carico CPU/SSD                 |
| **ExposureTime**       | 200–400 ms          | Espone abbastanza per le stelle, senza saturare meteore veloci |
| **AnalogueGain**       | 4–8                 | ISO medio-alto, aumenta sensibilità senza troppo rumore        |
| **AutoExposure**       | False               | Stabilizza luminosità, evita oscillazioni del cielo notturno   |
| **FPS (capture rate)** | 1–2 fps             | Adeguato con esposizioni lunghe; evita buffer overflow         |
| **White balance**      | Daylight / fixed    | Mantiene colori consistenti; evita fluttuazioni automatiche    |
| **Timelapse stack N**  | 10–20               | Migliora SNR; √N scaling del rumore, evita saturare RAM        |
| **Event buffer**       | 30 s                | Cattura intero passaggio meteora                               |
| **JPEG quality**       | 85                  | Compromesso tra spazio e qualità per salvataggi continui       |


meteora_pipeline_pi5_v3.py
Aggiunto start and stop

meteora_pipeline_pi5_v4.py
Aggiunto visualizzazione a schema di colori

meteora_pipeline_pi5_v5.py
Aggiunti metadati latitude e longitude nei file di output.

meteora_pipeline_pi5_v6.py
Calcolo automatico equivalent_exposure_s per eventi e timelapse.

meteora_pipeline_pi5_v7.py
Aggiunto parametro timelapse_interval_sec per gestire tempo tra stack.
meteora_pipeline_pi5_v71.py
gestito dentro timelapse_loop (non blocca più capture_loop).
meteora_pipeline_pi5_v72.py
Implementato stacking reale dei frame per timelapse.
meteora_pipeline_pi5_v73.py
Implementato stacking reale dei frame per timelapse (media, mediana, somma).
meteora_pipeline_pi5_v74.py
Sistemato save_event_metadata e ensure_dir
meteora_pipeline_pi5_v75.py
correct timelapse_loop
meteora_pipeline_pi5_v76.py
aggiunto save_json_atomic
meteora_pipeline_pi5_v77.py
modificata posizione save_json_atomic
aggiunto stack_frames
modificato timelapse
meteora_pipeline_pi5_v78.py
rimosso stack_frames duplicato alla fine della pipeline
meteora_pipeline_pi5_v79.py
Aggiunto system_monitor_loop opzionale per registrare temperatura CPU/SoC, memoria, disco, load average e GPU (se disponibile).
meteora_pipeline_pi5_v80.py
aggiunto daylight snapshot e controlli sulla qualità del cialo
meteora_pipeline_pi5_v81.py
Refactored:
__init__, _clear_queues, is_bypass_active, within_schedule, capture_loop
sky_monitor_loop, capture_daylight_snapshot, analyze_sky_conditions, wait_for_good_sky 
capture_threads_active, start_capture_threads, stop_capture_threads, start, stop
meteora_pipeline_pi5_v82.py
queue control before stop streads
meteora_pipeline_pi5_v83.py
bugs corrected
"""

import argparse
import atexit
import json
import logging
import math
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, time as dtime

import cv2
import numpy as np
import psutil

from picamera2 import Picamera2, Preview

# --------------------------- Configuration (tweak as needed) ---------------------------
DEFAULTS = {
    "capture": {
        "width": 1280,
        "height": 720,
        "fps": 2,
        "exposure_us": 200000,
        "gain": 8.0,
        "auto_exposure": False,
    },
    "detection": {
        "min_area": 50,
        "threshold": 25,
        "accumulate_alpha": 0.01,
        "event_buffer_sec": 30,
        "min_concurrent_frames": 1
    },
    "timelapse": {
        "stack_N": 20,
        "stack_align": True,
        "out_dir": "output/timelapse",
        "interval_sec": 0,
        "stack_method": "mean"
    },
    "events": {
        "out_dir": "output/events",
        "video_fps": 15,
        "video_width": 1280,
        "video_height": 720,
        "ffmpeg_encoder": "h264_v4l2m2m",
        "ffmpeg_bitrate": "1500k"
    },
    "monitor": {
        "interval_sec": 5,
        "out_file": "logs/system_monitor.csv"
    },
    "daylight": {
        "enabled": True,
        "time": "18:00",
        "out_dir": "output/daylight",
        "stddev_threshold": 15.0,
        "min_stars": 20,
        "retry_interval_min": 15,
        "check_interval_min": 30,
        "force_bypass": True        
    },
    "general": {
        "version": 83,
        "log_dir": "logs",
        "max_queue_size": 1000,
        "hostname": socket.gethostname(),
        "location": None,
        "latitude": 0.0,
        "longitude": 0.0,
        "start_time": "15:00",
        "end_time": "04:00"
    }
}

# --------------------------- Logging ---------------------------

class ColorLogFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",   # grigio
        logging.INFO: "\033[36m",    # ciano
        logging.WARNING: "\033[33m", # giallo
        logging.ERROR: "\033[31m",   # rosso
        logging.CRITICAL: "\033[41m" # rosso con sfondo
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorLogFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    logging.info("Logging started with color output to console")

# --------------------------- Camera capture using Picamera2 ---------------------------

class CameraCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.picam2 = Picamera2()
        main_config = self.picam2.create_still_configuration(main={"size": (cfg['width'], cfg['height'])})
        self.picam2.configure(main_config)

        if not cfg.get("auto_exposure", False):
            controls = {
                "ExposureTime": cfg.get("exposure_us", 200000),
                "AnalogueGain": cfg.get("gain", 8.0),
                "AwbEnable": False  # White balance fisso
            }
            self.picam2.set_controls(controls)
        else:
            self.picam2.set_controls({"AeEnable": True})

        self.picam2.start()
        time.sleep(2)

    def read(self):
        try:
            frame = self.picam2.capture_array()
            return frame
        except Exception as e:
            logging.warning("Picamera2 capture failed: %s", e)
            return None

    def release(self):
        self.picam2.stop()

# --------------------------- Pipeline class and workers ---------------------------

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        # --- REFACTORED THREADING AND STATE MANAGEMENT ---
        self.force_bypass_cfg = bool(self.cfg.get("daylight", {}).get("force_bypass", False))
        
        # Global stop signal for the entire application
        self.running = threading.Event()
        self.running.set()
        
        # Separate stop signal for the capture workers
        self.capture_running = threading.Event()
        self.capture_running.set()

        # Separate lists to manage different types of threads
        self.worker_threads = []
        self.control_threads = []
        # --- END REFACTOR ---

        maxq = cfg["general"].get("max_queue_size", 1000)
        self.acq_q = queue.Queue(maxsize=maxq)
        self.event_q = queue.Queue()
        self.timelapse_q = queue.Queue(maxsize=maxq)
        os.makedirs(cfg["events"]["out_dir"], exist_ok=True)
        os.makedirs(cfg["timelapse"]["out_dir"], exist_ok=True)
        os.makedirs(cfg["general"]["log_dir"], exist_ok=True)
        self.camera = CameraCapture(cfg["capture"])
        self.background = None
        self.prev_gray = None
        self.event_lock = threading.Lock()
        self.event_active = threading.Event()


    def _clear_queues(self):
        """Empties all data queues to prevent processing stale frames."""
        logging.info("Clearing all data queues...")
        for q in [self.acq_q, self.event_q, self.timelapse_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    # ----------------- bypass control -----------------
    def is_bypass_active(self):
        return bool(self.force_bypass_cfg)


    # ----------------- schedule control -----------------
    def within_schedule(self):
        now = datetime.now()
        start_h, start_m = map(int, self.cfg["general"]["start_time"].split(":"))
        end_h, end_m = map(int, self.cfg["general"]["end_time"].split(":"))
        start = dtime(start_h, start_m)
        end = dtime(end_h, end_m)
        current = now.time()
        if start <= end:
            return start <= current <= end
        else:
            return current >= start or current <= end
            
    def capture_loop(self):
        # IMPORTANT: This loop now checks the LOCAL capture_running event
        fps = self.cfg["capture"].get("fps", 2)
        interval = 1.0 / max(0.0001, fps)
        logging.info("Capture loop running at %.2f fps (interval %.3fs)", fps, interval)

        while self.running.is_set() and self.capture_running.is_set():
            if not self.within_schedule():
                time.sleep(10) # Shorter sleep while waiting for schedule
                continue
            
            ts = datetime.now(timezone.utc).isoformat()
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.2)
                continue
            
            try:
                # Put a copy for detection to avoid data races if needed later
                self.acq_q.put_nowait((ts, frame.copy()))
            except queue.Full:
                logging.warning("Acquisition queue full, dropping detection frame")

            try:
                # Timelapse queue gets its own copy
                self.timelapse_q.put_nowait((ts, frame.copy()))
            except queue.Full:
                try:
                    _ = self.timelapse_q.get_nowait()
                    self.timelapse_q.put_nowait((ts, frame.copy()))
                except queue.Empty:
                    pass
            
            time.sleep(interval)
        logging.info("Capture loop has exited.")

    # ----------------- stack -----------------
    def stack_frames(self, frames, align=True, method="mean"):
        """
        A robust, memory-aware method to align and stack a list of frames.
        It uses an efficient iterative approach for 'mean' and 'sum' methods,
        and a buffered approach for the memory-intensive 'median' method.

        Args:
            frames (list): A list of NumPy arrays (images).
            align (bool): If True, frames will be aligned to the first frame.
            method (str): The stacking method ('mean', 'median', 'sum').

        Returns:
            A stacked image as a NumPy array, or None if input is empty.
        """
        if not frames:
            return None

        h, w = frames[0].shape[:2]

        # --- PATH 1: Memory-efficient iterative calculation for MEAN and SUM ---
        if method in ["mean", "sum"]:
            accumulator = np.zeros((h, w, 3), dtype=np.float64)
            ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if align else None

            for i, f in enumerate(frames):
                frame_to_add = f
                if align and i > 0:
                    try:
                        warp_mode = cv2.MOTION_TRANSLATION
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
                        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (_, warp_matrix) = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
                        frame_to_add = cv2.warpAffine(f, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    except cv2.error as e:
                        logging.warning("Alignment failed for frame %d in stack, using unaligned frame: %s", i, e)
                
                accumulator += frame_to_add.astype(np.float64)
            
            if method == "mean":
                stacked_float = accumulator / len(frames)
            else:  # sum
                stacked_float = accumulator

        # --- PATH 2: Memory-intensive buffered calculation for MEDIAN ---
        elif method == "median":
            logging.warning("Median stacking is memory-intensive and may cause OOM errors on low-RAM systems.")
            aligned_frames = []
            ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if align else None

            for i, f in enumerate(frames):
                if align and i > 0:
                    try:
                        warp_mode = cv2.MOTION_TRANSLATION
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
                        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (_, warp_matrix) = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
                        aligned_frame = cv2.warpAffine(f, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                        aligned_frames.append(aligned_frame)
                    except cv2.error as e:
                        logging.warning("Alignment failed for frame %d in stack, using unaligned frame: %s", i, e)
                        aligned_frames.append(f) # Fallback to unaligned
                else:
                    # Append the first frame or all frames if not aligning
                    aligned_frames.append(f)
            
            stacked_float = np.median(aligned_frames, axis=0)
        
        else:
            logging.error("Unknown stack_method '%s'. Defaulting to a 'mean' stack.", method)
            # Default to the most common and safest method
            stacked_float = np.mean([f.astype(np.float64) for f in frames], axis=0)

        # Final conversion applies to all paths
        return stacked_float.clip(0, 255).astype(np.uint8)

    # ----------------- timelapse -----------------
    def timelapse_loop(self):
        cfg = self.cfg["timelapse"]
        stack_N = cfg.get("stack_N", 20)
        method = cfg.get("stack_method", "mean")
        align = cfg.get("stack_align", True)
        interval_sec = cfg.get("interval_sec", 0)
        out_dir = cfg["out_dir"]
        ensure_dir(out_dir)
        logging.info("Timelapse worker started (stack_N=%d, method=%s, align=%s)", stack_N, method, align)
        buffer = []

        # --- MERGED AND CORRECTED SINGLE LOOP ---
        while True:
            try:
                item = self.timelapse_q.get()

                if item is None:
                    logging.info("Timelapse loop received sentinel.")
                    if buffer:
                        logging.info("Processing final timelapse stack before exiting...")
                        frames_to_stack = [f for (_, f) in buffer]
                        stacked_image = self.stack_frames(frames_to_stack, align=align, method=method)
                        if stacked_image is not None:
                            tstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            out_name = f"timelapse_final_{tstamp}.png"
                            full_path = os.path.join(out_dir, out_name)
                            cv2.imwrite(full_path, stacked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                            self.save_timelapse_metadata(full_path)
                    logging.info("Draining complete, exiting.")
                    break # Exit the loop

                ts, frame = item
                buffer.append((ts, frame.copy()))
            except queue.Empty:
                continue

            if len(buffer) >= stack_N:
                frames_to_stack = [f for (_, f) in buffer]
                stacked_image = self.stack_frames(frames_to_stack, align=align, method=method)

                if stacked_image is not None:
                    tstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    out_name = f"timelapse_{tstamp}.png"
                    full_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(full_path, stacked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    logging.info("Saved timelapse stack to %s", full_path)
                    self.save_timelapse_metadata(full_path)
                
                buffer.clear()

                if interval_sec > 0:
                    # This logic for discarding is fine
                    end_time = time.time() + interval_sec
                    while time.time() < end_time:
                        try:
                            self.timelapse_q.get(timeout=0.5)
                        except queue.Empty:
                            time.sleep(0.1)

    def daylight_scheduler_loop(self):
        last_snapshot_date = None
        while self.running.is_set() and self.cfg["daylight"].get("enabled", True):
            now = datetime.now()
            target_h, target_m = map(int, self.cfg["daylight"]["time"].split(":"))
            target_time = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)

            if now >= target_time and (last_snapshot_date != now.date()):
                self.capture_daylight_snapshot()
                last_snapshot_date = now.date()

            time.sleep(30)  # controlla ogni 30 secondi

# ----------------- system monitor loop -----------------
    def system_monitor_loop(self):
        out_file = self.cfg["monitor"].get("out_file", "logs/system_monitor.csv")
        interval = self.cfg["monitor"].get("interval_sec", 5)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # scrivi header se il file è nuovo
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                f.write("timestamp,cpu_temp,cpu_percent,mem_used_mb,mem_percent,"
                        "disk_used_mb,disk_percent,gpu_percent,load1,load5,load15,uptime_sec\n")

    #    while self.running.is_set() and self.cfg["monitor"].get("enabled", True):
        while self.running.is_set():
            ts = datetime.now(timezone.utc).isoformat()

            # temperatura CPU/SoC
            temps = psutil.sensors_temperatures()
            cpu_temp = temps.get("cpu_thermal", [None])[0].current if "cpu_thermal" in temps else 0

            # uso CPU
            cpu_percent = psutil.cpu_percent(interval=None)

            # memoria
            mem = psutil.virtual_memory()
            mem_used_mb = mem.used / (1024*1024)

            # disco
            disk = psutil.disk_usage("/")
            disk_used_mb = disk.used / (1024*1024)

            # GPU usage (se disponibile)
            gpu_percent = 0
            try:
                gpu_stat = subprocess.check_output(["vcgencmd", "measure_utilisation"], text=True)
                parts = dict(p.split("=") for p in gpu_stat.strip().split())
                if "gpu" in parts:
                    gpu_percent = parts["gpu"].replace("%", "")
            except Exception:
                gpu_percent = 0

            # load average
            load1, load5, load15 = os.getloadavg()

            # uptime sistema
            uptime_sec = time.time() - psutil.boot_time()

            line = (f"{ts},{cpu_temp},{cpu_percent},{mem_used_mb:.1f},{mem.percent},"
                    f"{disk_used_mb:.1f},{disk.percent},{gpu_percent},{load1:.2f},"
                    f"{load5:.2f},{load15:.2f},{int(uptime_sec)}\n")
            with open(out_file, "a") as f:
                f.write(line)

            time.sleep(interval)

    # ----------------- metadata helper -----------------
    def get_metadata(self, n_frames=1):
        g = self.cfg["general"]
        exposure_us = self.cfg["capture"].get("exposure_us", 0)
        equivalent_exposure_s = (exposure_us * n_frames) / 1e6 if exposure_us > 0 else None
        return {
            "hostname": g.get("hostname"),
            "location": g.get("location"),
            "latitude": g.get("latitude"),
            "longitude": g.get("longitude"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exposure_us": exposure_us,
            "frames_stacked": n_frames,
            "equivalent_exposure_s": equivalent_exposure_s
        }

    def save_event_metadata(self, event_path):
        fps = self.cfg["capture"].get("fps", 1)
        event_buffer_s = self.cfg["detection"].get("event_buffer_sec", 30)
        n_frames = int(fps * event_buffer_s)
        meta = self.get_metadata(n_frames=n_frames)
#        json_path = event_path + ".json"
#        with open(json_path, "w") as f:
#            json.dump(meta, f, indent=2)
        json_path = event_path.rsplit('.', 1)[0] + ".json" # More robust way to get json path
        save_json_atomic(json_path, meta)
        logging.info("Saved event metadata to %s", json_path)

    def save_timelapse_metadata(self, img_path):
        n_frames = self.cfg["timelapse"].get("stack_N", 1)
        meta = self.get_metadata(n_frames=n_frames)
        meta["timelapse_interval_sec"] = self.cfg["timelapse"].get("interval_sec", 0)
        meta["stack_method"] = self.cfg["timelapse"].get("stack_method", "mean")
#        json_path = img_path + ".json"
#        with open(json_path, "w") as f:
#            json.dump(meta, f, indent=2)
        json_path = img_path.rsplit('.', 1)[0] + ".json" # More robust way to get json path
        save_json_atomic(json_path, meta)
        logging.info("Saved timelapse metadata to %s", json_path)

    def sky_monitor_loop(self):
        if self.is_bypass_active():
            logging.warning("Sky monitor is BYPASSED. Starting capture threads immediately.")
            self.start_capture_threads()
            # Loop indefinitely without taking action if bypassed
            while self.running.is_set():
                time.sleep(60)
            return

        logging.info("Sky monitor started. Waiting for good sky conditions to begin.")
        if not self.wait_for_good_sky():
            return # Exit if shutdown is requested during initial wait

        self.start_capture_threads()
        
        retry_min = self.cfg["daylight"].get("retry_interval_min", 15)
        check_min = self.cfg["daylight"].get("check_interval_min", 30)

        while self.running.is_set():
            # Wait for the next check interval
            for _ in range(check_min * 60):
                if not self.running.is_set(): break
                time.sleep(1)
            if not self.running.is_set(): break

            if self.capture_threads_active():
                # Wait for the next check interval
                for _ in range(check_min * 60):
                    if not self.running.is_set(): break
                    time.sleep(1)
                if not self.running.is_set(): break
                
                is_sky_ok = self.capture_daylight_snapshot()
                # ... (code for checking when threads are active is correct) ...
                if not is_sky_ok:
                    logging.warning("Sky monitor: Conditions worsened. Stopping and draining.")
                    self.stop_capture_threads()
                    
                    # Enter retry loop until conditions improve
                    if not self.wait_for_good_sky():
                        return # Exit if shutdown requested
                    
                    logging.info("Sky monitor: Conditions improved. Restarting capture.")
                    self.start_capture_threads()
            else:
                # If threads aren't active, it's because we are waiting for good sky
                logging.info("Sky monitor: Capture threads inactive, checking sky to start.")
                if self.wait_for_good_sky():
                    self.start_capture_threads()
                else:
                    # If shutdown was requested during wait, exit
                    if not self.running.is_set():
                        return

    def capture_daylight_snapshot(self):
        logging.info("Capturing sky condition snapshot...")
        night_cfg = self.cfg["capture"]
        is_sky_ok = False
        try:
            ensure_dir(self.cfg["daylight"]["out_dir"])
            
            # Switch to auto-exposure for daylight shot
            self.camera.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
            time.sleep(2) # Allow settings to stabilize

            frame = self.camera.read()
            if frame is None:
                logging.error("Failed to capture daylight snapshot frame.")
                return False # Treat camera failure as bad sky
            
            today = datetime.now().strftime("%Y-%m-%d_%H-%M")
            full_path = os.path.join(self.cfg["daylight"]["out_dir"], f"sky_conditions_{today}.jpg")
            cv2.imwrite(full_path, frame)

            results = self.analyze_sky_conditions(frame)
            meta = self.get_metadata(n_frames=1)
            meta.update(results)
            save_json_atomic(os.path.splitext(full_path)[0] + ".json", meta)
            logging.info("Saved daylight snapshot and analysis to %s", full_path)

            if (results["stddev"] >= self.cfg["daylight"]["stddev_threshold"] and
                results["stars"] >= self.cfg["daylight"]["min_stars"]):
                logging.info("Sky conditions: GOOD")
                is_sky_ok = True
            else:
                logging.warning("Sky conditions: BAD (stddev=%.2f, stars=%d)", results["stddev"], results["stars"])
                is_sky_ok = False
        
        except Exception as e:
            logging.exception("An error occurred during daylight snapshot: %s", e)
            return False
        
        finally:
            # CRITICAL: Always restore night settings
            self.camera.picam2.set_controls({
                "ExposureTime": night_cfg.get("exposure_us"),
                "AnalogueGain": night_cfg.get("gain"),
                "AwbEnable": False,
                "AeEnable": False
            })
            logging.info("Restored night capture settings.")
        
        return is_sky_ok

    def analyze_sky_conditions(self, img):
        # This method is correct, no changes needed, but ensure it's inside the Pipeline class
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, stddev_val = cv2.meanStdDev(gray)
        stddev_val = stddev_val[0][0]
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 2
        params.maxArea = 50
        detector = cv2.SimpleBlobDetector_create(params)
        # Use a more adaptive threshold for star detection
        _, thresh_img = cv2.threshold(gray, int(np.max(gray) * 0.7), 255, cv2.THRESH_BINARY)
        keypoints = detector.detect(thresh_img)
        return {"stddev": stddev_val, "stars": len(keypoints)}

    def wait_for_good_sky(self):
        if self.is_bypass_active():
            logging.warning("Bypassing sky check.")
            return True
        
        retry_min = self.cfg["daylight"].get("retry_interval_min", 15)
        while self.running.is_set():
            if self.capture_daylight_snapshot():
                return True
            logging.info("Retrying sky check in %d minutes.", retry_min)
            for _ in range(retry_min * 60):
                if not self.running.is_set(): return False
                time.sleep(1)
        return False

    def capture_threads_active(self):
        return any(t.is_alive() for t in self.worker_threads)

    def start_capture_threads(self):
        if self.capture_threads_active():
            logging.warning("Capture threads are already running. Ignoring start request.")
            return

        # Do not clear queues here, allow draining to continue if restarting
        self.capture_running.set()

        self.worker_threads = [
            threading.Thread(target=self.capture_loop, name="CaptureThread", daemon=True),
            threading.Thread(target=self.detection_loop, name="DetectionThread", daemon=True),
            threading.Thread(target=self.event_writer_loop, name="EventWriterThread", daemon=True),
            threading.Thread(target=self.timelapse_loop, name="TimelapseThread", daemon=True)
        ]
        for t in self.worker_threads:
            t.start()
        logging.info("All worker threads started.")

    # Ensure this method is INDENTED to be inside the Pipeline class
    def stop_capture_threads(self):
        if not self.capture_threads_active():
            logging.info("Capture threads are not running. Ignoring stop request.")
            return
            
        logging.info("Stopping capture thread and signaling workers to drain queues...")
        
        self.capture_running.clear()
        
        capture_thread = next((t for t in self.worker_threads if t.name == "CaptureThread"), None)
        if capture_thread:
            capture_thread.join(timeout=5)
        
        logging.info("Posting sentinels to worker queues.")
        self.acq_q.put(None)
        self.timelapse_q.put(None)

        logging.info("Waiting for worker threads to finish draining...")
        for t in self.worker_threads:
            if t.name != "CaptureThread" and t.is_alive():
                t.join(timeout=60)

        self.worker_threads = []
        logging.info("All worker threads have stopped cleanly after draining.")

    # ----------------- thread management -----------------
    def start(self):
        logging.info("Starting all pipeline services...")
        self.control_threads = [
            threading.Thread(target=self.system_monitor_loop, name="MonitorThread", daemon=True),
            threading.Thread(target=self.sky_monitor_loop, name="SkyMonitorThread", daemon=True)
        ]
        for t in self.control_threads:
            t.start()
        logging.info("Control threads started.")

    def stop(self):
        logging.info("Stopping all pipeline services...")
        self.running.clear()
        self.capture_running.clear() # Signal all loops to exit

        all_threads = self.worker_threads + self.control_threads
        for t in all_threads:
            if t.is_alive():
                t.join(timeout=5)
        
        try:
            self.camera.release()
        except Exception:
            pass
        logging.info("Pipeline stopped cleanly.")

    # ----------------- detection -----------------
    def detection_loop(self):
        logging.info("Detection loop started")
        # --- All setup remains the same ---
        alpha = self.cfg["detection"].get("accumulate_alpha", 0.01)
        thr = self.cfg["detection"].get("threshold", 25)
        min_area = self.cfg["detection"].get("min_area", 50)
        event_buffer_s = self.cfg["detection"].get("event_buffer_sec", 30)
        fps = self.cfg["capture"].get("fps", 2)
        buffer_frames = int(max(1, math.ceil(event_buffer_s * max(0.5, fps))))
        recent_frames = deque(maxlen=buffer_frames)

        # --- MERGED AND CORRECTED SINGLE LOOP ---
        while True:
            try:
                # Blocking get is efficient
                item = self.acq_q.get()

                # Sentinel check to exit the loop
                if item is None:
                    logging.info("Detection loop received sentinel. Draining complete, exiting.")
                    break

                ts, frame = item
            except queue.Empty:
                continue # Should not happen with blocking get, but safe to keep

            # --- ALL PROCESSING LOGIC IS NOW HERE ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)

            if self.background is None:
                self.background = gray.astype("float32")
                continue

            cv2.accumulateWeighted(gray, self.background, alpha)
            bg = cv2.convertScaleAbs(self.background)
            diff = cv2.absdiff(gray, bg)
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = any(cv2.contourArea(c) >= min_area for c in cnts)
            recent_frames.append((ts, frame.copy()))

            if detected:
                logging.info("Candidate event detected at %s (contours=%d)", ts, len(cnts))
                with self.event_lock:
                    for item_to_add in recent_frames:
                        try:
                            self.event_q.put_nowait(item_to_add)
                        except queue.Full:
                            logging.warning("Event queue full when pushing buffer")
                    self.event_active.set()

    # ----------------- event writer -----------------
    def event_writer_loop(self):
        logging.info("Event writer waiting for events")
        out_dir = self.cfg["events"]["out_dir"]
        ensure_dir(out_dir)
        fps = self.cfg["events"].get("video_fps", 15)
        width = self.cfg["events"].get("video_width", self.cfg["capture"]["width"])
        height = self.cfg["events"].get("video_height", self.cfg["capture"]["height"])
        encoder = self.cfg["events"].get("ffmpeg_encoder", "h264_v4l2m2m")
        bitrate = self.cfg["events"].get("ffmpeg_bitrate", "1500k")
        event_duration = self.cfg["detection"].get("event_buffer_sec", 30)

        while self.running.is_set() and self.capture_running.is_set():
            # wait for event flag
            if not self.event_active.wait(timeout=1.0):
                continue

            # collect frames for event_duration seconds
            frames = []
            start_t = time.time()
            while time.time() - start_t < event_duration:
                try:
                    ts, frame = self.event_q.get(timeout=1.0)
                    frames.append((ts, frame.copy()))
                except queue.Empty:
                    # no new frames; continue waiting until duration passes
                    continue

            if not frames:
                logging.info("Event flag set but no frames collected")
                self.event_active.clear()
                continue

            # write video via ffmpeg using pipe - feed raw BGR frames as RGB24
            t0 = frames[0][0]
            tstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base_name = f"event_{tstamp}"
            tmp_dir = os.path.join(out_dir, base_name + "_frames")
            ensure_dir(tmp_dir)

            # save frames as temporary jpeg (fast) and then make video with ffmpeg
            for i, (ts, frame) in enumerate(frames):
                fname = os.path.join(tmp_dir, f"f_{i:05d}.jpg")
                cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            video_path = os.path.join(out_dir, base_name + ".mp4")
            # ffmpeg command - use input images
            ff_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp_dir, "f_%05d.jpg"),
                "-c:v", encoder,
                "-b:v", bitrate,
                video_path
            ]
            logging.info("Encoding event video to %s (frames=%d)", video_path, len(frames))
            try:
#                subprocess.run(ff_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                result = subprocess.run(ff_cmd, check=True, capture_output=True, text=True)
                self.save_event_metadata(video_path)
                logging.info("Event video and metadata saved: %s", video_path)
            except subprocess.CalledProcessError as e:
#                logging.exception("ffmpeg failed: %s", e)
                logging.error("--- FFMPEG FAILED ---")
                logging.error("FFmpeg Command: %s", " ".join(e.cmd))
                logging.error("FFmpeg Exit Code: %d", e.returncode)
                logging.error("FFmpeg stderr:\n%s", e.stderr)
                logging.error("--- END FFMPEG ERROR ---")
            finally:
                # cleanup frames
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

            # clear flag
            self.event_active.clear()

# -------------------------------------------------------------------------------
# CLI and orchestrator
# -------------------------------------------------------------------------------

def save_json_atomic(path, data):
    try:
        temp_path = path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=4)
        os.rename(temp_path, path)
    except Exception as e:
        logging.error("Failed to save JSON atomically to %s: %s", path, e)

def ensure_dir(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser(description="Meteor detection pipeline PoC for Raspberry Pi 5")
    p.add_argument("--config", help="Path to JSON config file (optional)", default=None)
    return p.parse_args()


def load_config(path=None):
    cfg = DEFAULTS.copy()
    # deep merge simple
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_cfg = json.load(f)
        # merge dicts (one level)
        # merge ricorsivo
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Meteora Pipeline Pi5")
    parser.add_argument("--config", type=str, help="Path to config JSON", default=None)
    parser.add_argument("--duration", type=int, help="Duration in seconds (0 = infinite)", default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["general"]["log_dir"])

    pipeline = Pipeline(cfg)

    def handle_signal(sig, frame):
        logging.info("Signal received, stopping pipeline...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline.start()

    if args.duration > 0:
        logging.info("Running pipeline for %s seconds", args.duration)
        time.sleep(args.duration)
        pipeline.stop()
    else:
        logging.info("Running pipeline indefinitely (Ctrl+C to stop)")
        while True:
            time.sleep(60)


if __name__ == '__main__':
    main()
