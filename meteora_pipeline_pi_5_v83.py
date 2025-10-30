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

"""

import argparse
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
import cv2
import glob 
import numpy as np
import psutil
import io
import paramiko
import ntplib
import csv
import html
from collections import Counter
from zipfile import ZipFile
from enum import Enum, auto
from typing import Tuple
from collections import deque
from datetime import datetime, timezone, time as dtime
from logging.handlers import RotatingFileHandler
from picamera2 import Picamera2
from collections import namedtuple  # testing porpuse
from pydantic import BaseModel, Field, conint, confloat, constr, field_validator
from typing import Optional

try:
    import gpiod
    
    # The most reliable method is to directly try opening the device files
    # that are known to exist on Pi hardware.
    
    chip_path_to_open = None
    
    # Try the Pi 5 path first
    if os.path.exists('/dev/gpiochip4'):
        chip_path_to_open = '/dev/gpiochip4'
    # If not, fall back to the Pi 4
    elif os.path.exists('/dev/gpiochip0'):
        chip_path_to_open = '/dev/gpiochip0'
    
    # Didn't find any known chip path, raise an error to be caught below.
    if chip_path_to_open is None:
        raise FileNotFoundError("No known Raspberry Pi GPIO chip found (/dev/gpiochip4 or /dev/gpiochip0)")

    # Now, try to open the chip we found.
    GPIO_CHIP = gpiod.Chip(chip_path_to_open)
    
    IS_GPIO_AVAILABLE = True
    logging.info(f"GPIO support is enabled on chip '{chip_path_to_open}'.")

except (ImportError, FileNotFoundError, OSError) as e:
    IS_GPIO_AVAILABLE = False
    GPIO_CHIP = None
    logging.warning(f"GPIO support is DISABLED (Reason: {e}). Power monitor will not run.")

# --------------------------- Configuration ---------------------------
DEFAULTS = {
    "capture": {
        "width": 1280,
        "height": 720,
        "fps": 2,
        "exposure_us": 200000,
        "gain": 8.0,
        "auto_exposure": False,
        "red_gain": 2.0,
        "blue_gain": 1.5
    },
    "detection": {
        "min_area": 50,
        "threshold": 25,
        "accumulate_alpha": 0.01,
        "pre_event_sec": 4,
        "event_cooldown_sec": 8,
        "dark_frame_path": None
    },
    "timelapse": {
        "stack_N": 20,
        "stack_align": True,
        "interval_sec": 0,
        "stack_method": "mean",
        "min_features_for_aligment": 20
    },
    "timelapse_video": {
        "enabled": True,
        "video_fps": 10,
        "ffmpeg_encoder": "libx264",
        "ffmpeg_bitrate": "2000k"
    },
    "events": {
        "video_fps": 8,
        "ffmpeg_encoder": "libx264",
        "ffmpeg_bitrate": "1500k",
        "jpeg_quality": 85
    },
    "monitor": {
        "interval_sec": 5
    },
    "queue_monitor": {
        "enabled": True,
        "interval_sec": 30
    },
    "power_monitor": {
        "enabled": True,
        "pin": 17,
        "shutdown_delay_sec": 60
    },
    "led_status": {
        "enabled": True,
        "pin": 26
    },
    "daylight": {
        "enabled": True,
        "stddev_threshold": 15.0,
        "min_stars": 20,
        "check_interval_min": 30,
        "lux_threshold": 20.0
    },
    "janitor": {
        "monitor_path": "/",
        "log_rotation_mb": 10,
        "log_backup_count": 5,
        "csv_rotation_mb": 20,
        "csv_backup_count": 5
    },
    "dashboard": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5000
    },
    "heartbeat": {
        "enabled": True,
        "url": "https://hc-ping.com/b828f9df-eecb-4dd0-9fe6-112ea6382618",
        "interval_min": 60
    },
    "event_log": {
        "enabled": True
    },
    "health_monitor": {
        "log_rotation_mb": 5,
        "log_backup_count": 3
    },
    "sftp": {
        "enabled": False,
        "host": "sftp.myserver.net",
        "port": 22,
        "user": "myuser",
        "remote_dir": "/remote/path",
        "max_queue_size": 500
    },
    "ntp": {
        "enabled": True,
        "server": "pool.ntp.org",
        "sync_interval_hours": 6,
        "max_offset_sec": 2.0
    },
    "general": {
        "version": 151,
        "log_dir": "logs",
        "max_queue_size": 1000,
        "hostname": socket.gethostname(),
        "location": None,
        "latitude": 0.0,
        "longitude": 0.0,
        "max_camera_failures": 20,
        "max_restart_failures": 3,
        "idle_heartbeat_interval_min": 5,
        "power_monitor_pin": 17,
        "maintenance_timeout": 300,  
        "shutdown_time": None,
        "start_time": "15:00",
        "end_time": "06:00"
    }
}

# --------------------------- Configuration Schema (Pydantic) ---------------------------
#try:

class CaptureConfig(BaseModel):
    width: conint(gt=0,le=8000) = DEFAULTS["capture"]["width"]
    height: conint(gt=0,le=8000) = DEFAULTS["capture"]["height"]
    fps: confloat(gt=0,le=120) = DEFAULTS["capture"]["fps"]
    exposure_us: conint(gt=0,le=10000000) = DEFAULTS["capture"]["exposure_us"]
    gain: confloat(ge=0,le=16) = DEFAULTS["capture"]["gain"]
    auto_exposure: bool = DEFAULTS["capture"]["auto_exposure"]
    red_gain: confloat(gt=0,le=16) = DEFAULTS["capture"]["red_gain"]
    blue_gain: confloat(gt=0,le=16) = DEFAULTS["capture"]["blue_gain"]

class DetectionConfig(BaseModel):
    min_area: conint(gt=0,le=255) = DEFAULTS["detection"]["min_area"]
    threshold: conint(ge=0, le=255) = DEFAULTS["detection"]["threshold"]
    accumulate_alpha: confloat(gt=0, lt=1) = DEFAULTS["detection"]["accumulate_alpha"]
    pre_event_sec: conint(ge=0,le=5) = DEFAULTS["detection"]["pre_event_sec"]
    event_cooldown_sec: conint(ge=0) = DEFAULTS["detection"]["event_cooldown_sec"]
    dark_frame_path: Optional[str] = DEFAULTS["detection"]["dark_frame_path"]
    @field_validator('dark_frame_path', mode='before')
    def allow_empty_str_for_optional_path(cls, v):
        # If the input is an empty string, convert it to None before other validation.
        if v == '':
            return None
        return v

class TimelapseConfig(BaseModel):
    stack_N: conint(gt=0) = DEFAULTS["timelapse"]["stack_N"]
    stack_align: bool = DEFAULTS["timelapse"]["stack_align"]
    interval_sec: int = DEFAULTS["timelapse"]["interval_sec"]
    stack_method: str = DEFAULTS["timelapse"]["stack_method"]
    min_features_for_aligment: conint(ge=1) = DEFAULTS["timelapse"]["min_features_for_aligment"]

class TimelapseVideoConfig(BaseModel):
    enabled: bool = DEFAULTS["timelapse_video"]["enabled"]
    video_fps: conint(gt=0,le=30) = DEFAULTS["timelapse_video"]["video_fps"]
    ffmpeg_encoder: str = DEFAULTS["timelapse_video"]["ffmpeg_encoder"]
    ffmpeg_bitrate: str = DEFAULTS["timelapse_video"]["ffmpeg_bitrate"]

class EventsConfig(BaseModel):
    video_fps: conint(gt=0,le=30) = DEFAULTS["events"]["video_fps"]
    ffmpeg_encoder: str = DEFAULTS["events"]["ffmpeg_encoder"]
    ffmpeg_bitrate: str = DEFAULTS["events"]["ffmpeg_bitrate"]
    jpeg_quality: conint(ge=1, le=100) = DEFAULTS["events"]["jpeg_quality"]

class MonitorConfig(BaseModel):
    interval_sec: conint(gt=0,le=60) = DEFAULTS["monitor"]["interval_sec"]

class QueueMonitorConfig(BaseModel):
    enabled: bool = DEFAULTS["queue_monitor"]["enabled"]
    interval_sec: conint(gt=0,le=60) = DEFAULTS["queue_monitor"]["interval_sec"]

class PowerMonitorConfig(BaseModel):
    enabled: bool = DEFAULTS["power_monitor"]["enabled"]
    pin: int = DEFAULTS["power_monitor"]["pin"]
    shutdown_delay_sec: conint(ge=0) = DEFAULTS["power_monitor"]["shutdown_delay_sec"]

class LedStatusConfig(BaseModel):
    enabled: bool = DEFAULTS["led_status"]["enabled"]
    pin: int = DEFAULTS["led_status"]["pin"]

class DaylightConfig(BaseModel):
    enabled: bool = DEFAULTS["daylight"]["enabled"]
    stddev_threshold: float = DEFAULTS["daylight"]["stddev_threshold"]
    min_stars: conint(gt=10,le=50) = DEFAULTS["daylight"]["min_stars"]
    check_interval_min: conint(ge=1,le=60) = DEFAULTS["daylight"]["check_interval_min"]
    lux_threshold: float = DEFAULTS["daylight"]["lux_threshold"]

class JanitorConfig(BaseModel):
    monitor_path: str = DEFAULTS["janitor"]["monitor_path"]
    log_rotation_mb: conint(gt=5,le=20) = DEFAULTS["janitor"]["log_rotation_mb"]
    log_backup_count: conint(gt=0,le=10) = DEFAULTS["janitor"]["log_backup_count"]
    csv_rotation_mb: conint(gt=5,le=20) = DEFAULTS["janitor"]["csv_rotation_mb"]
    csv_backup_count: conint(gt=0,le=10) = DEFAULTS["janitor"]["csv_backup_count"]
    
class DashboardConfig(BaseModel):
    enabled: bool = DEFAULTS["dashboard"]["enabled"]
    host: str = DEFAULTS["dashboard"]["host"]
    port: conint(gt=1023, lt=65536) = DEFAULTS["dashboard"]["port"]

class HeartbeatConfig(BaseModel):
    enabled: bool = DEFAULTS["heartbeat"]["enabled"]
    url: str = DEFAULTS["heartbeat"]["url"]
    interval_min: conint(gt=10,le=60) = DEFAULTS["heartbeat"]["interval_min"]

class EventLogConfig(BaseModel):
    enabled: bool = DEFAULTS["event_log"]["enabled"]

class HealthMonitorConfig(BaseModel):
    log_rotation_mb: conint(gt=0) = DEFAULTS["health_monitor"]["log_rotation_mb"]
    log_backup_count: conint(gt=0) = DEFAULTS["health_monitor"]["log_backup_count"]

class SftpConfig(BaseModel):
    enabled: bool = DEFAULTS["sftp"]["enabled"]
    host: str = DEFAULTS["sftp"]["host"]
    port: int = DEFAULTS["sftp"]["port"]
    user: str = DEFAULTS["sftp"]["user"]
    remote_dir: str = DEFAULTS["sftp"]["remote_dir"]
    max_queue_size: int = DEFAULTS["sftp"]["max_queue_size"]

class NtpConfig(BaseModel):
    enabled: bool = DEFAULTS["ntp"]["enabled"]
    server: str = DEFAULTS["ntp"]["server"]
    sync_interval_hours: conint(ge=0) = DEFAULTS["ntp"]["sync_interval_hours"]
    max_offset_sec: float = DEFAULTS["ntp"]["max_offset_sec"]

class GeneralConfig(BaseModel):
    version: int = DEFAULTS["general"]["version"]
    log_dir: str = DEFAULTS["general"]["log_dir"]
    max_queue_size: conint(gt=0,le=1000) = DEFAULTS["general"]["max_queue_size"]
    hostname: str = DEFAULTS["general"]["hostname"]
    location: Optional[str] = DEFAULTS["general"]["location"]
    latitude: float = DEFAULTS["general"]["latitude"]
    longitude: float = DEFAULTS["general"]["longitude"]
    max_camera_failures: conint(gt=0) = DEFAULTS["general"]["max_camera_failures"]
    max_restart_failures: conint(gt=0,le=50) = DEFAULTS["general"]["max_restart_failures"]
    idle_heartbeat_interval_min: conint(gt=0) = DEFAULTS["general"]["idle_heartbeat_interval_min"]
    power_monitor_pin: int = DEFAULTS["general"]["power_monitor_pin"]
    maintenance_timeout: conint(gt=0) = DEFAULTS["general"]["maintenance_timeout"]
    shutdown_time: Optional[constr(pattern=r'^\d{2}:\d{2}$')] = DEFAULTS["general"]["shutdown_time"]
    start_time: constr(pattern=r'^\d{2}:\d{2}$') = DEFAULTS["general"]["start_time"]
    end_time: constr(pattern=r'^\d{2}:\d{2}$') = DEFAULTS["general"]["end_time"]
    @field_validator('location', 'shutdown_time', mode='before')
    def allow_empty_str_for_optionals(cls, v):
        # This single validator handles both 'location' and 'shutdown_time'
        if v == '':
            return None
        return v

class MainConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    timelapse: TimelapseConfig = Field(default_factory=TimelapseConfig)
    timelapse_video: TimelapseVideoConfig = Field(default_factory=TimelapseVideoConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    queue_monitor: QueueMonitorConfig = Field(default_factory=QueueMonitorConfig)
    power_monitor: PowerMonitorConfig = Field(default_factory=PowerMonitorConfig)
    led_status: LedStatusConfig = Field(default_factory=LedStatusConfig)
    daylight: DaylightConfig = Field(default_factory=DaylightConfig)
    janitor: JanitorConfig = Field(default_factory=JanitorConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    event_log: EventLogConfig = Field(default_factory=EventLogConfig)
    health_monitor: HealthMonitorConfig = Field(default_factory=HealthMonitorConfig)
    sftp: SftpConfig = Field(default_factory=SftpConfig)
    ntp: NtpConfig = Field(default_factory=NtpConfig)
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    
    @field_validator('*', mode='after')
    def check_stack_n_memory_usage(cls, v, values):
        """
        After all individual fields are validated, check if the configured
        stack_N value is safe for the system's available memory.
        """
        # This validator runs for every field, but we only need to execute
        # our logic once all the necessary data is available.
        # We check if 'capture' and 'timelapse' are present in the 'values.data' dict.
        if 'capture' in values.data and 'timelapse' in values.data:
            capture_cfg = values.data['capture']
            timelapse_cfg = values.data['timelapse']
            
            # 1. Estimate memory per frame (width * height * 3 bytes/pixel)
            bytes_per_frame = capture_cfg.width * capture_cfg.height * 3
            
            # 2. Get available system memory
            try:
                # Use a fresh check to get current available memory
                available_mem_bytes = psutil.virtual_memory().available
            except Exception:
                # If psutil fails, fallback to a safe, low default (e.g., 512MB)
                available_mem_bytes = 512 * 1024 * 1024
            
            # 3. Calculate a safe maximum stack size
            # We'll use a conservative limit: don't let the image buffer
            # consume more than 25% of the *available* RAM.
            safe_mem_for_stack = available_mem_bytes * 0.25
            safe_max_stack_n = int(safe_mem_for_stack / bytes_per_frame)
            
            # Failsafe: ensure the max is at least a reasonable number
            safe_max_stack_n = max(5, safe_max_stack_n)

            # 4. Perform the validation check
            if timelapse_cfg.stack_N > safe_max_stack_n:
                mem_per_frame_mb = bytes_per_frame / (1024*1024)
                configured_usage_mb = timelapse_cfg.stack_N * mem_per_frame_mb
                available_mem_mb = available_mem_bytes / (1024*1024)
                
                error_msg = (
                    f"Configuration failed: timelapse.stack_N ({timelapse_cfg.stack_N}) is too high for the available system RAM. "
                    f"Estimated usage: {configured_usage_mb:.0f} MB. "
                    f"Available RAM: {available_mem_mb:.0f} MB. "
                    f"Recommended maximum stack_N for this resolution is ~{safe_max_stack_n}."
                )
                raise ValueError(error_msg)
        
        return v # Must return the value for the next validator

#    IS_PYDANTIC_AVAILABLE = True
#except ImportError:
#    IS_PYDANTIC_AVAILABLE = False
#    MainConfig = None
#    print("WARNING: Pydantic library not found. Running without schema validation. `pip3 install pydantic`")

# --------------------------- Logging ---------------------------
class ThreadColorLogFormatter(logging.Formatter):
    """
    A custom logging formatter that assigns a unique color to each thread,
    in addition to coloring the log level.
    """
    RESET = "\033[0m"
    
    # A palette of high-contrast colors for thread names
    THREAD_COLORS = [
        # Standard
        "\033[36m",    # Cyan
        "\033[33m",    # Yellow
        "\033[35m",    # Magenta
        "\033[32m",    # Green
        "\033[34m",    # Blue
        "\033[31m",    # Red
        # Bright
        "\033[96m",    # Bright Cyan
        "\033[93m",    # Bright Yellow
        "\033[95m",    # Bright Magenta
        "\033[92m",    # Bright Green
        "\033[94m",    # Bright Blue
        "\033[91m",    # Bright Red
        # Bold
        "\033[36;1m",  # Bold Cyan
        "\033[33;1m",  # Bold Yellow
        "\033[35;1m",  # Bold Magenta
        "\033[32;1m",  # Bold Green
        "\033[34;1m",  # Bold Blue
    ]

    LEVEL_COLORS = {
        logging.DEBUG: "\033[90m",      # Grey
        logging.INFO: "\033[37m",       # White
        logging.WARNING: "\033[93;1m",  # Bright Yellow Bold
        logging.ERROR: "\033[91;1m",    # Bright Red Bold
        logging.CRITICAL: "\033[97;41m", # White on Red Background
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_color_map = {}
        self.lock = threading.Lock()

    def _get_thread_color(self, thread_name):
        # Assigns a persistent color to a thread name if not already assigned.
        with self.lock:
            if thread_name not in self.thread_color_map:
                new_color = self.THREAD_COLORS[len(self.thread_color_map) % len(self.THREAD_COLORS)]
                self.thread_color_map[thread_name] = new_color
            return self.thread_color_map[thread_name]

    def format(self, record):
        # Formats the log record with level and thread colors.
        level_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        thread_color = self._get_thread_color(record.threadName)

        # Create the formatted string with embedded ANSI color codes
        log_line = (
            f"\033[90m{self.formatTime(record, self.datefmt)}{self.RESET} "
            f"{level_color}[{record.levelname}]{self.RESET} "
            f"{thread_color}[{record.threadName}]{self.RESET} "
            f"{record.getMessage()}"
        )
        
        return log_line

    # --------------------------- Camera capture using Picamera2 ---------------------------
class CameraCapture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.picam2 = Picamera2()
        main_config = self.picam2.create_still_configuration(main={"size": (cfg['width'], cfg['height'])})
        self.picam2.configure(main_config)

        if not cfg.get("auto_exposure", False):
            # --- Red Gain, Blue Gain ---
            red_gain = cfg.get("red_gain", 2.0)
            blue_gain = cfg.get("blue_gain", 1.5)
            
            controls = {
                "ExposureTime": cfg.get("exposure_us", 200000),
                "AnalogueGain": cfg.get("gain", 8.0),
                "AwbEnable": False,  # Fixed white balance
                "ColourGains": [red_gain, blue_gain]
            }
#                logging.info(f"Setting manual camera controls: Exposure={controls['ExposureTime']}, Gain={controls['AnalogueGain']}, ColourGains={controls['ColourGains']}")
            self.picam2.set_controls(controls)
        else:
            self.picam2.set_controls({"AeEnable": True})

        self.picam2.start()
        time.sleep(2)
        logging.info("CameraCapture initialized successfully.")
            
    def read(self):
        try:
            frame = self.picam2.capture_array()
            return frame
        except Exception as e:
            logging.warning("Picamera2 capture failed: %s", e)
            return None

    def release(self):
        try:
            self.picam2.stop()
            logging.info("Camera stream stopped.")
            self.picam2.close()
            logging.info("Camera device closed and resources released.")
        except Exception as e:
            # Log an error but don't crash if the camera is already in a bad state
            logging.error(f"An error occurred during camera release: {e}")

    # --------------------------- SFTP connect and upload ---------------------------
class SFTPUploader:
    def __init__(self, cfg, pipeline_instance):
        self.cfg = cfg
        self.pipeline_instance = pipeline_instance
        self.upload_q = queue.Queue()
        self.stop_event = threading.Event()
        self.sftp_password = os.environ.get("SFTP_PASS")
        self.upload_q = queue.Queue(maxsize=cfg.get("max_queue_size", 500)) 
        
        if self.sftp_password:
            logging.info("SFTP password loaded from SFTP_PASS environment variable.")
        else:
            # If the password environment variable is missing, log an error.
            logging.error("SFTP_PASS environment variable is not set. SFTP uploads will fail.")

    def _ensure_remote_dir(self, sftp_client):
        remote_dir = self.cfg['remote_dir']
        try:
            # Check if the directory already exists
            sftp_client.stat(remote_dir)
            logging.info(f"Remote directory '{remote_dir}' already exists.")
        except FileNotFoundError:
            # If it doesn't exist, create it.
            logging.warning(f"Remote directory '{remote_dir}' not found. Attempting to create it.")
            try:
                # Create directories recursively
                dirs = remote_dir.strip('/').split('/')
                current_dir = ''
                for d in dirs:
                    current_dir = os.path.join(current_dir, d)
                    # For SFTP, paths should always use forward slashes
                    current_dir_posix = current_dir.replace(os.path.sep, '/') 
                    if not current_dir_posix.startswith('/'):
                        current_dir_posix = '/' + current_dir_posix
                    try:
                        sftp_client.stat(current_dir_posix)
                    except FileNotFoundError:
                        logging.info(f"Creating remote directory: {current_dir_posix}")
                        sftp_client.mkdir(current_dir_posix)
            except Exception as e:
                logging.error(f"Failed to create remote directory '{remote_dir}': {e}")
                # Re-raise the exception to cause the connection to fail.
                raise

    def connect(self):
        try:
            # Use the higher-level SSHClient which simplifies connection and has a timeout
            client = paramiko.SSHClient()
            # Automatically add the server's host key. For high-security environments,
            # you might want to load known host keys instead.
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())            
            
            client.connect(
                hostname=self.cfg['host'],
                port=self.cfg['port'],
                username=self.cfg['user'],
                password=self.sftp_password,
                timeout=30
            )

            sftp = client.open_sftp()
            logging.info("SFTP connection established.")
            
            self._ensure_remote_dir(sftp)
            
            # Return both the client and sftp object for proper cleanup later
            return client, sftp
        
        except Exception as e:
            # This will now catch timeout errors, authentication errors, and name resolution errors.
            msg = f"SFTP connection failed: {e}"
            logging.error(msg)
            self.pipeline_instance.log_health_event("ERROR", "SFTP_connection_failed", msg)
            return None, None

    def get_status(self):
        """Returns a tuple of (status_class, status_message) for the dashboard."""
        q_size = self.upload_q.qsize()
        
        # Check for power loss first, as it's the highest priority status
        with self.pipeline_instance.status_lock:
            if self.pipeline_instance.power_status != "OK":
                return ("err", f"Paused (Power Loss) - Queue: {q_size}")

        q_max = self.upload_q.maxsize
        if q_max == 0: return ("ok", f"Queue: {q_size}") # Should not happen, but safe

        q_percent = (q_size / q_max) * 100
        
        if q_percent >= 99:
            return ("err", f"Backlogged - Queue: {q_size}/{q_max} ({q_percent:.0f}%)")
        elif q_percent > 75:
            return ("warn", f"Filling - Queue: {q_size}/{q_max} ({q_percent:.0f}%)")
        else:
            return ("ok", f"Idle/Uploading - Queue: {q_size}")

    def worker_loop(self):
        ssh_client = None
        sftp = None
        was_paused_by_power_loss = False
        
        while not self.stop_event.is_set():

            # --- PRIORITY 1: CHECK FOR POWER LOSS ---
            if self.pipeline_instance.power_status != "OK":
                if not was_paused_by_power_loss:
                    logging.warning("SFTP Uploader: Power loss detected. Pausing all upload activity.")
                    was_paused_by_power_loss = True
                
                # If a connection exists, close it to be safe during a power outage.
                if sftp:
                    sftp.close()
                    sftp = None
                    
                self.stop_event.wait(timeout=10) 
                continue # Restart the loop to re-check power status.
            
            # If we are here, power is OK. Check if it was just restored.
            if was_paused_by_power_loss:
                logging.info("SFTP Uploader: Power has been restored. Resuming uploads.")
                was_paused_by_power_loss = False
                # sftp is already None, so the next block will force a reconnect.

            # --- PRIORITY 2: GET A CONNECTION (STATE 1) ---
            if sftp is None:
                logging.info("SFTP: No active connection. Attempting to connect...")
                ssh_client, sftp = self.connect()
                
                if sftp is None:
                    # Connection failed. Wait without touching the queue.
                    logging.warning("SFTP: Connection failed. Will retry in 60 seconds.")
                    self.stop_event.wait(timeout=60)
                    continue
                else:
                    logging.info("SFTP: Connection established. Starting to process upload queue.")

            # --- PRIORITY 3: PROCESS THE QUEUE (STATE 2) ---
            # Now that we know power is OK and we have a connection, we can get a file.
            try:
                local_path = self.upload_q.get(timeout=5)
                if local_path is None: continue

            except queue.Empty:
                # No files to upload.
                continue

            # Attempt the upload.
            try:
                remote_path = os.path.join(self.cfg['remote_dir'], os.path.basename(local_path))
                sftp.put(local_path, remote_path)
                logging.info(f"Uploaded: {os.path.basename(local_path)}")
                self.mark_as_uploaded(local_path)

            except Exception as e:
                # An error occurred during the upload. The connection is likely now broken.

                msg = f"Upload failed for {os.path.basename(local_path)}: {e}"
                logging.error(msg)
                
                self.pipeline_instance.log_health_event("ERROR", "SFTP_UPLOAD_FAIL", msg)
                
                logging.warning("SFTP connection may have been lost. Re-queueing file and forcing reconnect.")
                
                # Re-queue the file so it's not lost.
                self.upload_q.put(local_path) 
                
                # Close the broken connection and set sftp to None.
                # This will force to go back to STATE 1 on the next iteration.
                if sftp: sftp.close()
                if ssh_client: ssh_client.close()
                sftp, ssh_client = None, None
        
        # Final cleanup on shutdown.
        if sftp: 
            sftp.close()
            logging.info("SFTP connection closed.")
            
        if ssh_client: ssh_client.close()

    def mark_as_uploaded(self, file_path):
        # Marks a file's metadata as uploaded.
        json_path = os.path.splitext(file_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r+') as f:
                    meta = json.load(f)
                    meta["uploaded_at"] = datetime.now(timezone.utc).isoformat()
                    self.pipeline_instance.save_json_atomic(json_path, meta)
            except Exception as e:
                logging.error(f"Failed to mark {os.path.basename(json_path)} as uploaded: {e}")

    def stop(self):
            """Signals the SFTP worker to stop and clears its queues."""
            self.stop_event.set()
            # Post a sentinel to unblock the worker_loop if it's waiting on the queue
            try: self.upload_q.put_nowait(None)
            except queue.Full: pass
            
            # Clear any remaining items in the queues
            with self.upload_q.mutex: self.upload_q.queue.clear()
            with self.pipeline_instance.sftp_dispatch_q.mutex: self.pipeline_instance.sftp_dispatch_q.queue.clear()

    # -----------------
class SkyConditionStatus(Enum):
    # Represents the possible sky condition check result.
    CLEAR = auto()
    CLOUDY = auto()
    ERROR = auto()
    
    # --------------------------- Pipeline class and workers ---------------------------
class Pipeline:
    def __init__(self, cfg, config_path=None):

        self.cfg = cfg
        
        self.config_path = config_path or "config.json"  # config path
        # Get the base path from the janitor config. Default to the script's directory.
        base_path = self.cfg["janitor"].get("monitor_path", ".")    
        
        # Set the paths and files.
        self.events_out_dir = os.path.join(base_path, "output/events")
        self.timelapse_out_dir = os.path.join(base_path, "output/timelapse")
        self.daylight_out_dir = os.path.join(base_path, "output/daylight")
        self.calibration_out_dir = os.path.join(base_path, "output/calibration")
        self.general_log_dir = os.path.join(base_path, self.cfg["general"]["log_dir"])
        self.monitor_out_file = os.path.join(self.general_log_dir, "system_monitor.csv")
        self.event_log_out_file = os.path.join(self.general_log_dir, "events.csv")
        self.health_log_out_file = os.path.join(self.general_log_dir, "health_stats.csv")
        self.backlog_file_path = os.path.join(self.general_log_dir, "sftp_backlog.txt")
 
        os.makedirs(self.events_out_dir, exist_ok=True)
        os.makedirs(self.timelapse_out_dir, exist_ok=True)
        os.makedirs(self.general_log_dir, exist_ok=True)
        os.makedirs(self.calibration_out_dir, exist_ok=True)
        os.makedirs(self.daylight_out_dir, exist_ok=True)
 
        # Global stop signal for the entire application
        self.running = threading.Event()
        self.running.set()
        
        # Separate stop signal for the capture workers
        self.capture_running = threading.Event()
        self.capture_running.set()

        # Lists to manage different types of threads
        self.worker_threads = []
        self.control_threads = []

        maxq = cfg["general"].get("max_queue_size", 1000)
        self.acq_q = queue.Queue(maxsize=maxq)          # Queue for the aquisition frame
        self.event_q = queue.Queue()                    # Queue for the event frame
        self.timelapse_q = queue.Queue(maxsize=maxq)    # Queue for timelapse frame
        self.event_stack_q = queue.Queue()              # Queue for event stacks
        self.event_log_q = queue.Queue()                # Queue for the event logger
        self.health_q = queue.Queue()                   # Queue for health statistics
        self.calibration_q = queue.Queue(maxsize=100)   # Queue for the calibration image
        self.sftp_dispatch_q = queue.Queue()            # Queue for SFTP dispatcher
        
        self.last_calibration_image_path = None
        self.is_calibrating = threading.Event()
        self.is_calibrating.clear()
        self.event_counter = 0
            
        self.camera = CameraCapture(cfg["capture"])
        self.background = None
        self.camera_fatal_error = threading.Event()     # Set by capture_loop if fatal
        self.consecutive_restart_failures = 0
        self.config_reloaded_ack = threading.Event()    # Triggers for config reload

        # --- Initialize the SFTP ---
        self.sftp_uploader = None
        if self.cfg["sftp"].get("enabled", False):
            try:
                # Dependencies
                import paramiko 
                self.sftp_uploader = SFTPUploader(self.cfg["sftp"], self)
            except ImportError:
                logging.error("SFTP is enabled, but 'paramiko' library is not installed. Please run 'pip3 install paramiko'.")
            except Exception as e:
                logging.error(f"Failed to initialize SFTP uploader: {e}")
        
        # --- Initialize status LED ---
        self.led_line = None
        if IS_GPIO_AVAILABLE and self.cfg["led_status"].get("enabled", False):
            try:
                self.led_pin = self.cfg["led_status"]["pin"]
                
                # 1. Create a LineSettings object to define the line's configuration.
                settings = gpiod.LineSettings(
                    direction=gpiod.line.Direction.OUTPUT,
                    output_value=gpiod.line.Value.ACTIVE  # Set initial value to ON (booting)
                )
                
                # 2. Request the line with the specified settings.
                self.led_line = GPIO_CHIP.request_lines(
                    consumer="meteora-pipeline-led",
                    config={self.led_pin: settings}
                )
                logging.info(f"Status LED enabled on GPIO pin {self.led_pin}. LED is ON for startup sequence.")

            except Exception as e:
                logging.error(f"Failed to initialize status LED on pin {self.led_pin}: {e}", exc_info=True)
                if self.led_line: self.led_line.release() # Ensure release on failure
                self.led_line = None
        else:
            logging.info("Status LED is disabled.")

        # --- Load and validate the dark frame ---
        self.master_dark = None
        dark_path = self.cfg["detection"].get("dark_frame_path")
        if dark_path and os.path.exists(dark_path):
            try:
                dark_img = cv2.imread(dark_path)
                if dark_img is None:
                    raise ValueError("File could not be read by OpenCV.")

                # Ensure dark frame dimensions match capture dimensions
                cam_h, cam_w = self.cfg["capture"]["height"], self.cfg["capture"]["width"]
                dark_h, dark_w, _ = dark_img.shape
                if cam_h != dark_h or cam_w != dark_w:
                    raise ValueError(f"Dimension mismatch. Camera is {cam_w}x{cam_h}, but dark frame is {dark_w}x{dark_h}.")

                self.master_dark = dark_img
                logging.info("Successfully loaded and validated master dark frame from: %s", dark_path)
            except Exception as e:
                logging.error("Failed to load master dark frame from '%s': %s. Feature will be disabled.", dark_path, e)
                self.master_dark = None # Ensure it's disabled on failure
        elif dark_path:
            logging.warning("Master dark frame file not found at: %s. Proceeding without dark frame subtraction.", dark_path)
                              
        self.daylight_mode_active = threading.Event()
        self.daylight_mode_active.clear()               # Default to night mode (detection on)
        self.mode_check_lock = threading.Lock()         # Lock to prevent race conditions during mode checks
        self.weather_hold_active = threading.Event()
        self.weather_hold_active.clear()                # Start in good weather status
        self.event_in_progress = threading.Event()
        self.event_in_progress.clear()                  # Start in the "no event" state
        self.reload_config_signal = threading.Event()
        self.request_sky_analysis_trigger = threading.Event()
        self.status_lock = threading.Lock()             # A dedicated lock for status variables
        self.last_illuminance = "N/A"
        self.last_sky_conditions = {"stddev": "N/A", "stars": "N/A"}
        self.power_status = "OK"                        # Can be OK, LOST, or SHUTTING_DOWN
        self.last_heartbeat_status = "N/A"  
        self.last_calibration_error = None
        self.last_event_files = {"image": None, "video": None}
        self.session_start_time = None
        self.camera_lock = threading.Lock()
        self.in_maintenance_mode = threading.Event()
        self.in_maintenance_mode.clear()                # Start in normal operating mode
        self.watchdog_pause = threading.Event()
        self.emergency_stop_active = threading.Event()
        self.emergency_stop_active.clear()              # Set in the "not stopped" state
        self.initial_sftp_sweep_complete = threading.Event()
        self.initial_sftp_sweep_complete.clear()        # Start in the "not complete" state
        self.initial_janitor_check_complete = threading.Event()
        self.initial_janitor_check_complete.clear()     # Start in the "not complete" state
        self.system_ready = threading.Event() # Master "All Systems Go" signal
        self.maintenance_timeout_lock = threading.Lock()
        self.maintenance_timeout_until = 0              # A timestamp indicating when the timeout expires
        self.maintenance_timeout_duration_sec = cfg["general"].get("maintenance_timeout", 300)
        self._initial_cleanup()

    # -----------------
    def _initial_cleanup(self):
        # Scans and removes orphaned temporary directories.
        logging.info("Performing initial cleanup check...")
        event_dir = self.events_out_dir
        cleanup_count = 0
        if os.path.isdir(event_dir):
            try:
                with os.scandir(event_dir) as it:
                    for entry in it:
                        # Temporary directories are our specific target
                        if entry.is_dir() and entry.name.endswith('_frames'):
                            logging.warning("Found orphaned temp directory from previous run: %s. Deleting.", entry.name)
                            try:
                                shutil.rmtree(entry.path)
                                cleanup_count += 1
                            except Exception:
                                logging.exception("Error deleting orphaned directory %s", entry.path)
            except Exception:
                logging.exception("Error during initial cleanup scan.")
        if cleanup_count > 0:
            logging.info("Initial cleanup removed %d orphaned directories.", cleanup_count)
            
    # ----------------- Power Monitor -----------------
    def power_monitor_loop(self):
        '''
        Monitor a GPIO pin for power loss using the gpiod library.
        If a sustained outage is detected, it initiates
        a graceful shutdown and system halt.
        '''
        
        if not IS_GPIO_AVAILABLE:
            logging.warning("gpiod library not found. Power monitor is disabled.")
            return

        power_pin = self.cfg["power_monitor"].get("pin", 17)
        shutdown_delay_sec = self.cfg["power_monitor"].get("shutdown_delay_sec", 60)
        check_interval = 2  # Delay between check.
        
        line = None
        try:
            settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT,
                bias=gpiod.line.Bias.PULL_UP
            )
            
            line_request = GPIO_CHIP.request_lines(
                consumer="meteora-pipeline-power",
                config={power_pin: settings}
            )
            logging.info(f"Power monitor started on GPIO pin {power_pin} (shutdown delay: {shutdown_delay_sec}s).")
        except Exception:
            logging.exception("Failed to initialize GPIO pin %d for power monitor. The thread will now exit.", power_pin)
            if line_request: line_request.release()
            return

        power_lost = False
        power_loss_time = 0

        while self.running.is_set():
            try:
                # Assumes power is OK when the pin is HIGH (1) and LOST when LOW (0).
                is_power_ok = line_request.get_value(power_pin) == gpiod.line.Value.ACTIVE
                              
                if not is_power_ok and not power_lost:
                    # Power outage
                    power_lost = True
                    power_loss_time = time.time()
                    with self.status_lock:
                        self.power_status = "LOST"
                    logging.warning("Initial power loss detected. Starting %d second confirmation timer.", shutdown_delay_sec)
                    self.log_health_event("WARNING", "POWER_ERROR", "Initial power loss detected")

                elif is_power_ok and power_lost:
                    # Power was restored.
                    power_lost = False
                    with self.status_lock:
                        self.power_status = "OK"
                    logging.info("Power was restored. Cancelling shutdown timer.")

                elif not is_power_ok and power_lost:
                    if time.time() - power_loss_time > shutdown_delay_sec:
                        msg="!!! CONFIRMED POWER OUTAGE (lost for >%d sec) !!!", shutdown_delay_sec
#                        logging.critical("="*60)
#                        logging.critical("!!! CONFIRMED POWER OUTAGE (lost for >%d sec) !!!", shutdown_delay_sec)
#                        logging.critical("Initiating graceful shutdown and system halt.")
#                        logging.critical("="*60)
                        
                        with self.status_lock:
                            self.power_status = "SHUTTING DOWN"
                        self.perform_system_action("shutdown", reason=msg)
                        break 
                
                if self.running.wait(timeout=check_interval):
                    break

            except Exception:
                logging.exception("An error occurred in the power monitor loop. The thread will exit.")
                break # Exit the loop on error
        
        if line:
            line.release()

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

    # -----------------
    def capture_loop(self): 
        logging.info("Capture loop started. Waiting for the 'System Ready' signal...")
        
        # Wait until the main 'run' method signals that all threads are started.
        # This prevents this thread from producing data before consumers are ready.
        self.system_ready.wait()
        logging.info("'System Ready' signal received. Capture is now active.")

        first_run=True
        consecutive_failures = 0
        idle_loop_counter = 0

        while self.running.is_set() and self.capture_running.is_set():
            
            max_failures = self.cfg["general"].get("max_camera_failures", 20)
            heartbeat_min = self.cfg["general"].get("idle_heartbeat_interval_min", 5)
            idle_log_interval_loops = (heartbeat_min * 60) / 10
            fps = self.cfg["capture"].get("fps", 2)
            interval = 1.0 / max(0.0001, fps)            
            
            if first_run:
                logging.info("Capture loop started. Max failures: %d. Heartbeat: %d min.", max_failures, heartbeat_min)
                first_run=False
            
            if not self.within_schedule() and not self.is_calibrating.is_set():
                idle_loop_counter += 1
                if idle_loop_counter >= idle_log_interval_loops:
                    start_time = self.cfg["general"]["start_time"]
                    end_time = self.cfg["general"]["end_time"]
                    logging.info("System is idle, waiting for active schedule (%s - %s).", start_time, end_time)
                    idle_loop_counter = 0
                time.sleep(10)
                continue
            
            idle_loop_counter = 0
            ts = datetime.now(timezone.utc).isoformat()
            with self.camera_lock:
                frame = self.camera.read()

            if frame is None:
                consecutive_failures += 1
                logging.warning("Camera read failed (%d/%d consecutive failures).", consecutive_failures, max_failures)
                if consecutive_failures >= max_failures:
                    logging.critical("Camera has failed %d consecutive times. This is considered a fatal error.", max_failures)
                    # Don't stop the pipeline here. Simply exit this thread.
                    # The Scheduler will see that died and initiate recovery.
                    self.camera_fatal_error.set() # Fatal camera error
                    return # Terminate this thread
                time.sleep(1) # Wait a bit longer after a single failure
                continue
            
            # If a frame is read successfully, reset the counter
            consecutive_failures = 0
            
            try:
                self.acq_q.put_nowait((ts, frame.copy()))
            except queue.Full:
                logging.warning("Acquisition queue full, dropping frame")
                self.log_health_event("WARNING", "ACQUISITION_QUEUE", "Acquisition queue full.")

            time.sleep(interval)
        logging.info("Capture loop has exited.")

    # ----------------- stack -----------------
    def stack_frames(self, frames, align=True, method="mean", min_features_for_aligment=20):
        """
        A method to align and stack a list of frames.
        It uses an iterative approach for 'mean' and 'sum' methods,
        and a buffered approach for the memory-intensive 'median' method.

        method should be 'mean', 'median', 'sum'.
        """
        if not frames:
            return None

        def _has_enough_features(gray_img):
            # Quickly checks if an image has enough features to be worth aligning.
            # Use Shi-Tomasi corner detection to find star-like features
            corners = cv2.goodFeaturesToTrack(
                image=gray_img,
                maxCorners=150,
                qualityLevel=0.01,
                minDistance=10
            )
            return corners is not None and len(corners) >= min_features_for_aligment

        h, w = frames[0].shape[:2]

        # --- NEW: Memory Safety Guard for Median Stacking ---
        if method == "median":
            bytes_per_frame = h * w * 3
            total_mem_required = bytes_per_frame * len(frames)
            available_mem = psutil.virtual_memory().available
            
            # Use a conservative 50% threshold of available RAM for this single operation
            if total_mem_required > (available_mem * 0.5):
                logging.error(
                    f"MEMORY SAFETY HALT: Median stacking {len(frames)} frames would require "
                    f"~{total_mem_required / 1e6:.0f} MB, which is too risky. "
                    f"Falling back to 'mean' stacking for this operation."
                )
                self.log_health_event("ERROR", "OOM_PREVENTION", "Median stack aborted, fallback to mean.")
                method = "mean" # Force fallback

        # --- NEW: Logic to pre-validate the reference frame and prevent log spam ---
        perform_alignment = align
        ref_gray = None
        if perform_alignment:
            ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            if not _has_enough_features(ref_gray):
                logging.warning(
                    "Reference frame for stack has insufficient features. "
                    "Alignment will be skipped for this entire stack."
                )
                perform_alignment = False # Disable alignment for this specific run

        # --- PATH 1: Memory-efficient iterative calculation for MEAN and SUM ---
        if method in ["mean", "sum"]:
            accumulator = np.zeros((h, w, 3), dtype=np.float64)
            
            # # --- Logic to pre-validate the reference frame and prevent log spam ---
            # perform_alignment = align # Start with the user's preference
            # ref_gray = None
            # if perform_alignment:
                # ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                # if not _has_enough_features(ref_gray):
                    # logging.warning(
                        # "Reference frame for stack has insufficient features. "
                        # "Alignment will be skipped for this entire stack."
                    # )
                    # perform_alignment = False # Disable alignment for this run
            # # --- END LOGIC ---            

            for i, f in enumerate(frames):
                frame_to_add = f
                if perform_alignment and i > 0:
                    try:
                        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        
                        # --- Proactively skip alignment on bad frames ---
                        if not _has_enough_features(gray):
                            # logging.warning("Skipping alignment for a frame due to insufficient features (likely clouds).")
                            # By raising this cv2.error, we trigger the existing fallback logic cleanly.
                            raise cv2.error("Custom: Not enough features to align.")

                        # --- Relaxed criteria to improve convergence ---
                        warp_mode = cv2.MOTION_TRANSLATION
                        # Increased max iterations from 50 to 100, slightly lowered epsilon
#                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6) # Relaxed criteria
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        
                        (_, warp_matrix) = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
                        
                        frame_to_add = cv2.warpAffine(f, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    except cv2.error:
                        # This will now catch both ECC failures and our custom "not enough features" trigger
#                        logging.warning("Alignment failed for frame %d in stack, using unaligned frame: %s", i, e)
#                        self.log_health_event("WARNING", "ALIGMENT_ERROR", "Alignment failed (mean, sum).")
                        pass
                        
                accumulator += frame_to_add.astype(np.float64)
            
            if method == "mean":
                stacked_float = accumulator / len(frames)
            else:  # sum
                stacked_float = accumulator

        # --- PATH 2: Memory-intensive buffered calculation for MEDIAN ---
        elif method == "median":
#            logging.warning("Median stacking is memory-intensive and may cause OOM errors on low-RAM systems.")
            aligned_frames = []
#            ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if align else None

            for i, f in enumerate(frames):
                if align and i > 0:
                    try:
                        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        if not _has_enough_features(gray):
                            raise cv2.error("Custom: Not enough features.")

                        warp_mode = cv2.MOTION_TRANSLATION
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (_, warp_matrix) = cv2.findTransformECC(ref_gray, gray, warp_matrix, warp_mode, criteria)
                        aligned_frames.append(cv2.warpAffine(f, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
                    except cv2.error:
                        aligned_frames.append(f) # Fallback to unaligned
                else:
                    aligned_frames.append(f)
            
            stacked_float = np.median(aligned_frames, axis=0)
        
        else:
            logging.error("Unknown stack_method '{method}'. This should not happen.")
            # Default to the most common and safest method
            return none

        # Final conversion applies to all paths
        return stacked_float.clip(0, 255).astype(np.uint8)

    # ----------------- timelapse -----------------
    def timelapse_loop(self):
        first_run=True
        buffer = []

        while True:
            try:
                cfg = self.cfg["timelapse"]
                stack_N = cfg.get("stack_N", 20)
                method = cfg.get("stack_method", "mean")
                align = cfg.get("stack_align", True)
                interval_sec = cfg.get("interval_sec", 0)
                aligment_features = cfg.get("min_features_for_aligment", 20)   
                
                if first_run:
                    logging.info("Timelapse worker started (stack_N=%d, method=%s, align=%s)", stack_N, method, align)
                    first_run=False
                
                item = self.timelapse_q.get()

                if item is None:
                    logging.info("Timelapse loop received sentinel.")
                    if buffer:
                        logging.info("Processing final timelapse stack before exiting...")
                        frames_to_stack = [f for (_, f) in buffer]
                        stacked_image = self.stack_frames(frames_to_stack, align=align, method=method, min_features_for_aligment=aligment_features)
                        if stacked_image is not None:
                            tstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            out_name = f"timelapse_final_{tstamp}.png"
                            full_path = os.path.join(self.timelapse_out_dir, out_name)
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
                stacked_image = self.stack_frames(frames_to_stack, align=align, method=method, min_features_for_aligment=aligment_features)

                if stacked_image is not None:
                    tstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    out_name = f"timelapse_{tstamp}.png"
                    full_path = os.path.join(self.timelapse_out_dir, out_name)
                    cv2.imwrite(full_path, stacked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    logging.info("Saved timelapse stack to %s", full_path)
                    self.save_timelapse_metadata(full_path)

                    if self.sftp_uploader and self.running.is_set():
                        logging.info(f"Queueing {os.path.basename(full_path)} and its metadata for upload.")
                        self.sftp_dispatch_q.put(full_path)
                        self.sftp_dispatch_q.put(os.path.splitext(full_path)[0] + ".json")
                        
                buffer.clear()

                if interval_sec > 0:
                    # This logic for discarding is fine
                    end_time = time.time() + interval_sec
                    while time.time() < end_time:
                        try:
                            self.timelapse_q.get(timeout=0.5)
                        except queue.Empty:
                            time.sleep(0.1)

    # -----------------
    def start_timelapse_video_creation(self):
        """
        Launches the timelapse video creation process in a new, non-blocking
        daemon thread.
        """
        if not self.cfg["timelapse_video"].get("enabled", False):
            return

        logging.info("Scheduler has triggered end-of-session timelapse video creation.")
        # Run the potentially long video encoding process in a separate thread
        # so it doesn't block the scheduler or the main shutdown sequence.
        video_thread = threading.Thread(
            target=self._create_timelapse_video,
            name="TimelapseVideoThread",
            daemon=True # Daemon threads will not block program exit
        )
        video_thread.start()

    # -----------------
    def _create_timelapse_video(self):
        """
        Gathers all timelapse images from the last session and encodes them
        into a single MP4 video file using FFmpeg.
        """
        logging.info("Timelapse video creation thread started.")
        # Give other threads a moment to finish writing their last files
        time.sleep(10)

        if not self.cfg["timelapse_video"].get("enabled"):
            logging.info("Video creation is disabled")
            return

        if self.session_start_time is None:
            logging.warning("No session start time recorded. Cannot create timelapse video.")
            return

        session_start_ts = self.session_start_time.timestamp()

        # 1. Find all timelapse PNGs from the current session
        image_files = []
        for entry in os.scandir(self.timelapse_out_dir):
            if entry.is_file() and entry.name.startswith("timelapse_") and entry.name.endswith(".png"):
                try:
                    if entry.stat().st_mtime >= session_start_ts:
                        image_files.append(entry.path)
                except FileNotFoundError:
                    continue # File might have been deleted

        if len(image_files) < 2:
            logging.warning(f"Not enough timelapse images ({len(image_files)}) found for this session to create a video.")
            return

        # 2. Sort files chronologically (by filename, which contains the timestamp)
        image_files.sort()
        logging.info(f"Found {len(image_files)} timelapse images for video creation.")

        # 3. Create a temporary file list for FFmpeg (most reliable method)
        filelist_path = os.path.join(self.general_log_dir, "timelapse_filelist.txt")
        with open(filelist_path, 'w') as f:
            for path in image_files:
                # FFmpeg's concat demuxer needs a specific format
                f.write(f"file '{os.path.abspath(path)}'\n")

        # 4. Construct and run the FFmpeg command
        try:
            cfg = self.cfg["timelapse_video"]
            tstamp = self.session_start_time.strftime("%Y%m%d")
            output_path = os.path.join(self.timelapse_out_dir, f"timelapse_video_{tstamp}.mp4")

            ff_cmd = [
                "ffmpeg",
                "-y",                   # Overwrite output file if it exists
                "-f", "concat",         # Use the concatenation demuxer
                "-safe", "0",           # Allow absolute paths in the file list
                "-r", str(cfg["video_fps"]), # Input framerate (how fast to read images)
                "-i", filelist_path,    # The input file list
                "-c:v", cfg["ffmpeg_encoder"],
                "-b:v", cfg["ffmpeg_bitrate"],
                "-pix_fmt", "yuv420p",  # For broad player compatibility
                output_path
            ]
            
            logging.info(f"Encoding end-of-session timelapse video to {os.path.basename(output_path)}...")
            subprocess.run(ff_cmd, check=True, capture_output=True, text=True, timeout=600) # Generous 10-minute timeout
            logging.info(f"Successfully created timelapse video: {os.path.basename(output_path)}")

            # Also queue this final video for upload
            if self.sftp_uploader and self.running.is_set():
                logging.info(f"Queueing {os.path.basename(output_path)} for upload.")
                self.sftp_dispatch_q.put(output_path)

        except subprocess.TimeoutExpired:
            logging.error("FFmpeg process for timelapse video timed out. The video was not saved.")
            self.log_health_event("ERROR", "FFMPEG_TIMEOUT", "Timelapse video creation timed out.")
        except subprocess.CalledProcessError as e:
            logging.error("--- TIMELAPSE FFMPEG FAILED ---")
            logging.error("FFmpeg Command: %s", " ".join(e.cmd))
            logging.error("FFmpeg stderr:\n%s", e.stderr)
            if e.stderr: logging.error("FFmpeg stderr:\n%s", e.stderr.strip())
            self.log_health_event("ERROR", "FFMPEG_FAIL", "Timelapse video creation failed.")
        except Exception as e:
            logging.exception("An error occurred during timelapse video creation: %s", e)
        finally:
            # 5. Clean up the temporary file list
            if os.path.exists(filelist_path):
                os.remove(filelist_path)

    # ----------------- event stacker -----------------
    def event_stacker_loop(self):
        logging.info("Event Stacker thread started and waiting for packages.")

        out_dir = self.events_out_dir

        while True:
            try:
                
                cfg = self.cfg["timelapse"]
                method = cfg.get("stack_method", "mean")
                align = cfg.get("stack_align", True)
                aligment_features = cfg.get("min_features_for_aligment", 20)                
                
                event_frames = self.event_stack_q.get()
                if event_frames is None:
                    logging.info("Event Stacker received sentinel. Exiting.")
                    break
                if not event_frames:
                    continue

                logging.info("Event Stacker received package of %d frames.", len(event_frames))
                frames_to_stack = [f for (_, f) in event_frames]
                
                stacked_image = self.stack_frames(frames_to_stack, align=align, method=method, min_features_for_aligment=aligment_features)

                if stacked_image is not None:
                    tstamp = datetime.fromisoformat(event_frames[-1][0]).strftime("%Y%m%dT%H%M%SZ")
                    out_name = f"event_stack_{tstamp}.png"
                    full_path = os.path.join(out_dir, out_name)
                    cv2.imwrite(full_path, stacked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    self.save_event_stack_metadata(full_path, event_frames)
                    logging.info("Saved event stack image to %s", full_path)
                    
                    # Update the dashboard state with the path to the new stacked image
                    with self.status_lock:
                        self.last_event_files["image"] = full_path

            except Exception as e:
                logging.exception("Error in event_stacker_loop: %s", e)

    # ----------------- system monitor loop -----------------
    def system_monitor_loop(self):
        # Write the header if the file is new
        if not os.path.exists(self.monitor_out_file):
            with open(self.monitor_out_file, "w") as f:
                f.write("timestamp,cpu_temp,cpu_percent,mem_used_mb,mem_percent,"
                        "disk_used_mb,disk_percent,gpu_percent,load1,load5,load15,uptime_sec,Vin\n")

        while self.running.is_set():
            interval = self.cfg["monitor"].get("interval_sec", 5)
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
            
            core_voltage_v = self.get_core_voltage()            

            line = (f"{ts},{cpu_temp},{cpu_percent},{mem_used_mb:.1f},{mem.percent},"
                    f"{disk_used_mb:.1f},{disk.percent},{gpu_percent},{load1:.2f},"
                    f"{load5:.2f},{load15:.2f},{int(uptime_sec)},{core_voltage_v:.4f}\n")
            with open(self.monitor_out_file, "a") as f:
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

    # -----------------
    def save_event_metadata(self, event_path, event_frames):
        if not event_frames: return
        n_frames = len(event_frames)
        meta = self.get_metadata(n_frames=n_frames)
        meta["source_event_start_time"] = event_frames[0][0]
        meta["source_event_end_time"] = event_frames[-1][0]
        json_path = os.path.splitext(event_path)[0] + ".json"
        self.save_json_atomic(json_path, meta)
        logging.info("Saved event metadata to %s", json_path)

    # -----------------
    def save_timelapse_metadata(self, img_path):
        n_frames = self.cfg["timelapse"].get("stack_N", 1)
        meta = self.get_metadata(n_frames=n_frames)
        meta["timelapse_interval_sec"] = self.cfg["timelapse"].get("interval_sec", 0)
        meta["stack_method"] = self.cfg["timelapse"].get("stack_method", "mean")
        json_path = img_path.rsplit('.', 1)[0] + ".json" # More robust way to get json path
        self.save_json_atomic(json_path, meta)
        logging.info("Saved timelapse metadata to %s", json_path)

    # -----------------        
    def save_event_stack_metadata(self, img_path, event_frames):
        if not event_frames: return
        
        n_frames = len(event_frames)
        meta = self.get_metadata(n_frames=n_frames)
        meta["source_event_start_time"] = event_frames[0][0]
        meta["source_event_end_time"] = event_frames[-1][0]
        meta["stack_method"] = self.cfg["timelapse"].get("stack_method", "mean")
        
        json_path = os.path.splitext(img_path)[0] + ".json"
        self.save_json_atomic(json_path, meta)
        logging.info("Saved event stack metadata to %s", json_path)        

    # -----------------        
    def sky_monitor_loop(self):
        logging.info("Sky monitor started. The master control loop is now running.")
        
        while self.running.is_set():
            check_min = self.cfg["daylight"].get("check_interval_min", 30)
            
            # 1. Defer check if a meteor event is actively being recorded.
            if self.event_in_progress.is_set():
                logging.warning("Sky Monitor: An event is in progress. Deferring scheduled check for 2 minutes.")
                for _ in range(120): # Wait for 2 minutes
                    if not self.running.is_set(): break
                    time.sleep(1)
                continue # Re-start the loop to check again.

            # 2. Only perform a check if the main capture thread is currently active.
            if self.producer_thread_active():
                logging.info("Sky Monitor: Pausing capture to perform periodic quality check...")
                try:
                    logging.info("Sky Monitor: Pausing watchdog for quality check.")
                    self.watchdog_pause.set()
                    
                    self.stop_producer_thread()
                
                    # A. Check for daylight to determine the primary operating mode.
                    self.update_operating_mode()                

                    # B. If we are in Night Mode, also check for clouds.
                    if not self.daylight_mode_active.is_set():                   
                        status, _ = self.capture_daylight_snapshot()

                        if status == SkyConditionStatus.CLEAR and self.weather_hold_active.is_set():
                            logging.info("Sky Monitor: Sky has cleared. Releasing WEATHER HOLD.")
                            self.weather_hold_active.clear()
                        
                        elif status == SkyConditionStatus.CLOUDY and not self.weather_hold_active.is_set():
                            logging.warning("Sky Monitor: Clouds or poor conditions detected. Engaging WEATHER HOLD.")
                            self.weather_hold_active.set()
                        
                        elif status == SkyConditionStatus.ERROR:
                            logging.error("Sky Monitor: Failed to determine sky conditions due to an error.")
                            self.log_health_event("ERROR", "SKY_MONITOR_FAIL", "Failed to determine sky conditions due to an error.")
                finally:
                    # 3. CRITICAL: Always restart the producer after the check is complete.
                    if self.running.is_set() and not self.producer_thread_active():
                        logging.info("Sky Monitor: Check complete. Resuming main capture.")
                        self.start_producer_thread()

                    logging.info("Sky Monitor: Resuming watchdog checks.")
                    self.watchdog_pause.clear()
                        
            # 3. Wait for the next full interval before checking again.
            logging.info("Sky Monitor: Next check in %d minutes or upon request.", check_min)
            for _ in range(check_min * 60):
                if not self.running.is_set(): break # Allow for fast shutdown
                if self.request_sky_analysis_trigger.is_set():
                    logging.info("Sky Monitor: Immediate check requested. Aborting wait and running now.")
                    self.request_sky_analysis_trigger.clear()
                    break 
                time.sleep(1)

    # -----------------
    def capture_daylight_snapshot(self) -> Tuple[SkyConditionStatus, dict]:
        """
        Captures a snapshot, analyzes it for clarity, and returns a detailed status.
        """
        logging.info("Capturing sky condition snapshot...")
#        night_cfg = self.cfg["capture"]

#        with self.camera_lock:        
        try:              
            # Switch to auto-exposure for the snapshot
            self.camera.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
            time.sleep(2)

            frame = self.camera.read()
            if frame is None:
                logging.error("Failed to capture sky condition snapshot frame.")
                self.log_health_event("ERROR", "CAPTURE_DAYLIGHT_FAIL", "Failed to capture sky condition snapshot frame.")
                return SkyConditionStatus.ERROR, {"message": "Camera read returned None."}
            
            # --- Analysis and Metadata Saving ---
            today = datetime.now().strftime("%Y-%m-%d_%H-%M")
            full_path = os.path.join(self.daylight_out_dir, f"sky_conditions_{today}.jpg")
            cv2.imwrite(full_path, frame)

            results = self.analyze_sky_conditions(frame)
            with self.status_lock:
                self.last_sky_conditions = results
                
            meta = self.get_metadata(n_frames=1)
            meta.update(results)
            self.save_json_atomic(os.path.splitext(full_path)[0] + ".json", meta)
            logging.info("Saved daylight snapshot and analysis to %s", full_path)

            # --- Return Detailed Status Instead of a Simple Boolean ---
            is_clear = (results["stddev"] >= self.cfg["daylight"]["stddev_threshold"] and
                        results["stars"] >= self.cfg["daylight"]["min_stars"])

            if is_clear:
                logging.info("Sky conditions: CLEAR")
                return SkyConditionStatus.CLEAR, results
            else:
                logging.warning("Sky conditions: CLOUDY/POOR (stddev=%.2f, stars=%d)", results["stddev"], results["stars"])
                return SkyConditionStatus.CLOUDY, results
        
        except Exception as e:
            logging.exception(f"An error occurred during daylight snapshot: {e}")
            return SkyConditionStatus.ERROR, {"message": str(e)}

    # -----------------
    def update_operating_mode(self):
        """
        Checks camera's illuminance and dynamically reconfigures the camera
        for either 'Daylight Mode' (auto) or 'Night Mode' (manual) without
        stopping the capture thread.
        """
        lux_threshold = self.cfg.get("daylight", {}).get("lux_threshold", 10.0)
        night_cfg = self.cfg["capture"]

        try:
            # 1. Temporarily switch to auto-exposure to get a good light reading
            self.camera.picam2.set_controls({"AeEnable": True})
            time.sleep(2) # Allow sensor to adjust

            request = self.camera.picam2.capture_request()
            metadata = request.get_metadata()
            request.release()

            current_lux = metadata.get("Lux", -1)
            if current_lux == -1:
                logging.warning("Could not find 'Lux' value in camera metadata.")
                return
                
            with self.status_lock: self.last_illuminance = f"{current_lux:.2f}"
            logging.info("Light level check: Current illuminance is %.2f Lux.", current_lux)

            # 2. Reconfigure camera for Day or Night
            if current_lux > lux_threshold:
                if not self.daylight_mode_active.is_set():
                    logging.warning("Illuminance is HIGH. Switching camera to DAYLIGHT MODE (Auto Exposure/AWB).")
                    self.daylight_mode_active.set()
                
                # SET CAMERA TO AUTO DAY MODE
                self.camera.picam2.set_controls({"AeEnable": True, "AwbEnable": True})

            else:
                if self.daylight_mode_active.is_set():
                    logging.info("Illuminance is LOW. Switching camera to NIGHT MODE (Manual Exposure/Gains).")
                    self.daylight_mode_active.clear()
                
                # SET CAMERA TO MANUAL NIGHT MODE
                red_gain = night_cfg.get("red_gain", 2.0)
                blue_gain = night_cfg.get("blue_gain", 1.5)
                controls = {
                    "ExposureTime": night_cfg.get("exposure_us"),
                    "AnalogueGain": night_cfg.get("gain"),
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ColourGains": [red_gain, blue_gain]
                } 
                self.camera.picam2.set_controls(controls)
            

        except Exception as e:
            logging.error(f"Failed to update operating mode: {e}", exc_info=True)

    # -----------------
    def analyze_sky_conditions(self, img):
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

    # ----------------- Janitor -----------------
    def janitor_loop(self):
        """A dedicated thread to periodically manage disk space and rotate logs."""
        logging.info("Janitor thread started. Will run checks every hour.")
       
        csv_path = self.monitor_out_file
        
        while self.running.is_set():
            
            csv_max_mb = self.cfg.get("janitor", {}).get("csv_rotation_mb", 20)
            csv_backups = self.cfg.get("janitor", {}).get("csv_backup_count", 5)
        
            # 1. Perform the multi-tiered disk space cleanup.
            # This function will return False if it fails to free enough space.
            cleanup_successful = self.manage_disk_space()

            # 2. If cleanup failed, it's a critical error. Stop the entire pipeline.
            if not cleanup_successful:
                if not self.emergency_stop_active.is_set():
                    logging.critical("!!! JANITOR FAILED TO FREE DISK SPACE. ENTERING EMERGENCY STOP. !!!")
                    logging.critical("All data acquisition will be halted to prevent a system crash.")
                    logging.critical("The dashboard remains active for manual intervention.")

                    # self.stop_producer_thread()
                    self.emergency_stop_active.set()

            # 3. If cleanup was successful, perform the data file rotation.
            self.rotate_data_file(csv_path, max_size_mb=csv_max_mb, backup_count=csv_backups)
            
            logging.info("Janitor run complete. Next check in 1 hour.")
            self.initial_janitor_check_complete.set()
            for _ in range(3600):
                if not self.running.is_set():
                    break
                time.sleep(1)

    # -----------------
    def manage_disk_space(self, threshold=90.0, target=75.0):
        """
        Performs a multi-tiered cleanup. Returns True on success, False on failure.
        Tier 1: Deletes leftover temporary event frame directories.
        Tier 2: Deletes the oldest timelapse files.
        """
        try:
            # 1. Get the path to monitor from the configuration.
            monitor_path = self.cfg["janitor"].get("monitor_path", "/")
            
            # 2. Use this path for all disk usage checks.
            disk = psutil.disk_usage(monitor_path)
            if disk.percent < threshold:
                return True # Nothing to do, success.

            logging.warning(f"Disk usage on '{monitor_path}' is at {disk.percent:.1f}% (threshold is {threshold:.1f}%). Starting cleanup.")
            self.log_health_event("WARNING", "DISK_USAGE", "disk usage.")
            
            # --- TIER 1: CLEAN UP LEFTOVER EVENT BACKLOGS ---
            event_dir = self.events_out_dir
            backlog_count = 0
            if os.path.isdir(event_dir):
                with os.scandir(event_dir) as it:
                    for entry in it:
                        # Temporary directories are named 'event_..._frames'
                        if entry.is_dir() and entry.name.endswith('_frames'):
                            try:
                                logging.warning("Janitor: Found and deleting leftover event frame directory: %s", entry.name)
                                self.log_health_event("WARNING", "LEFTOVER_EVENT", "Found and deleting leftover event.")
                                shutil.rmtree(entry.path)
                                backlog_count += 1
                            except Exception:
                                logging.exception("Janitor: Error deleting backlog directory %s", entry.path)
            if backlog_count > 0:
                logging.info("Janitor: Tier 1 cleanup removed %d backlog directories.", backlog_count)
            
            # Re-check usage. If we're already good, we can stop.
            if psutil.disk_usage(monitor_path).percent < target:
                logging.info("Disk space is now sufficient. Cleanup complete.")
                return True

            # --- TIER 2: CLEAN UP OLDEST TIMELAPSE FILES ---
            deleted_count = 0
            scan_dirs = [self.timelapse_out_dir, self.events_out_dir]
            all_files_to_scan = []
            for directory in scan_dirs:
                if os.path.isdir(directory):
                    all_files_to_scan.extend(
                        [entry.path for entry in os.scandir(directory) if entry.is_file() and not entry.name.endswith('.json') and not entry.name.startswith('.')]
                    )
            
            all_files_to_scan.sort(key=os.path.getctime)

            while psutil.disk_usage(monitor_path).percent > target and all_files_to_scan:
                file_to_delete = all_files_to_scan.pop(0)
                try:
                    if self.safe_to_delete(file_to_delete):
                        logging.info("Janitor: Deleting old, uploaded file: %s", os.path.basename(file_to_delete))
                        os.remove(file_to_delete)
                        json_path = os.path.splitext(file_to_delete)[0] + ".json"
                        if os.path.exists(json_path):
                            os.remove(json_path)

                        deleted_count += 1
                    else:
                        logging.info(f"Janitor: Skipping deletion of un-uploaded file: {os.path.basename(file_to_delete)}")
                except OSError as e:
                    logging.error("Janitor: Failed to delete file %s: %s", file_to_delete, e)
                    self.log_health_event("ERROR", "JANITOR_FAIL", "Failed to delete file")
                    
            if deleted_count > 0:
                logging.info("Janitor: Tier 2 cleanup removed %d file(s).", deleted_count)

            # --- FINAL ---
            final_disk_percent = psutil.disk_usage(monitor_path).percent
            if final_disk_percent < target:
                logging.info(f"Janitor: Cleanup successful. New disk usage on '{monitor_path}' is {final_disk_percent:.1f}%.")
                return True
            else:
                logging.critical(f"!!! JANITOR FAILURE: DISK SPACE ON '{monitor_path}' COULD NOT BE FREED SUFFICIENTLY ({final_disk_percent:.1f}%) !!!")
                return False

        except Exception:
            logging.exception("An unhandled error occurred in the janitor process.")
            return True # Return true on error to avoid an unnecessary shutdown

    # -----------------
    def safe_to_delete(self, file_path):
        """Checks a file's metadata to see if it has been marked as uploaded."""
        # If SFTP is not enabled, it's always safe to delete.
        if not self.sftp_uploader:
            return True

        json_path = os.path.splitext(file_path)[0] + ".json"
        if not os.path.exists(json_path):
            # No metadata, assume it's an old file or an anomaly. Safer to keep.
            logging.warning(f"Janitor: Skipping deletion of {os.path.basename(file_path)} because it has no metadata.")
            self.log_health_event("WARNING", "JANITOR", "Skipping deletion no metadata.")
            return False

        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logging.warning(f"Janitor: Failed to read metadata for {os.path.basename(file_path)}, skipping deletion: {e}")
            self.log_health_event("WARNING", "JANITOR", "Failed to read metadata, skipping deletion.")
            return False # Fail safe: don't delete if metadata is unreadable

        # Check for the 'uploaded_at' key.
        return "uploaded_at" in meta

    # -----------------
    def rotate_data_file(self, file_path, max_size_mb=10, backup_count=5):
        """Rotates a generic data file (like a CSV) if it exceeds a max size."""
        try:
            if not os.path.exists(file_path):
                return

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            if file_size_mb < max_size_mb:
                return

            logging.warning("Data file '%s' has reached %.1fMB. Rotating.", os.path.basename(file_path), file_size_mb)
            
            # --- Backup Rotation Logic ---
            # First, shift all existing backups: .4 -> .5, .3 -> .4, etc.
            for i in range(backup_count - 1, 0, -1):
                src = f"{file_path}.{i}"
                dst = f"{file_path}.{i+1}"
                if os.path.exists(src):
                    shutil.move(src, dst)
            
            # Now, rotate the main file to .1
            if os.path.exists(file_path):
                shutil.move(file_path, f"{file_path}.1")

            # CRITICAL: Re-create the header for the new, empty CSV file
            if file_path == self.monitor_out_file:
                 with open(file_path, "w") as f:
                    f.write("timestamp,cpu_temp,cpu_percent,mem_used_mb,mem_percent,"
                            "disk_used_mb,disk_percent,gpu_percent,load1,load5,load15,uptime_sec\n")

        except Exception:
            logging.exception("An error occurred during data file rotation.")

    # ----------------- Scheduler -----------------
    def scheduler_loop(self):
        """
        A dedicated control thread that acts as the master timer. It starts and
        stops the producer thread based on the main schedule.
        """
        max_restart_attempts = self.cfg["general"].get("max_restart_failures", 3)
        logging.info("Master scheduler started. Will check every 60 seconds.")

        # --- Wait for the initial SFTP sweep to complete before doing anything. ---
        logging.info("Scheduler is waiting for initial SFTP sweep to complete...")
        self.initial_sftp_sweep_complete.wait()
        logging.info("Scheduler: SFTP sweep complete. Waiting for initial Janitor check...")
        self.initial_janitor_check_complete.wait()
        logging.info("Scheduler: All startup checks passed. Cleared to start normal operations.")
        
        while self.running.is_set():

            now_str = datetime.now().strftime("%H:%M")
            shutdown_time = self.cfg["general"].get("shutdown_time")

            # --- Scheduled System Shutdown Check ---
            if shutdown_time and now_str == shutdown_time:
                logging.warning(f"Scheduler: Reached scheduled shutdown time ({shutdown_time}).")
                # Trigger video creation BEFORE shutting down
                self.start_timelapse_video_creation()
                time.sleep(5)
                self.perform_system_action("shutdown", reason="Scheduled Daily Shutdown")
                break # Exit loop (perform_system_action should already have killed us, but just in case)

            is_active_schedule = self.within_schedule()
            is_capturing = self.producer_thread_active()

            if self.emergency_stop_active.is_set():
                if is_capturing: self.stop_producer_thread()
                logging.warning("Scheduler: System is in EMERGENCY STOP an error. All capture is inhibited.")
                self.log_health_event("CRITICAL", "EMERGENCY_STOP", "Critical state.")
                # Wait for 60 seconds before checking again
                for _ in range(60):
                    if not self.running.is_set(): break
                    time.sleep(1)
                continue # Skip the rest of the scheduler logic   

            if self.camera_fatal_error.is_set():  # the camera reach max_camera_failures
                logging.error("Scheduler detected fatal camera error. Attempting restart (%d/%d)...", self.consecutive_restart_failures + 1, max_restart_attempts)
                self.log_health_event("ERROR", "CAMERA_DETECT_FAIL", "Fatal camera error.")
                self.stop_producer_thread() # Ensure it's fully stopped
                time.sleep(0.1)
                self.camera_fatal_error.clear() # Reset for new attempt
                self.start_producer_thread() # Attempt restart
                
                # Wait a moment to potentially fail again or stabilize
                time.sleep(10) 

                if not self.producer_thread_active():
                    self.consecutive_restart_failures += 1
                    logging.critical("Producer thread failed to restart. (%d/%d)",
                                     self.consecutive_restart_failures, max_restart_attempts)
                    if self.consecutive_restart_failures >= max_restart_attempts:
                        msg="!!! FAILED TO RECOVER AFTER %d ATTEMPTS. INITIATING SYSTEM REBOOT. !!!", max_restart_attempts

                        self.perform_system_action("reboot", reason=msg)
                        break
                else:
                    logging.info("Producer thread restarted successfully after fatal error.")
                    self.consecutive_restart_failures = 0 # Reset on success
            
            # --- START/STOP LOGIC ---
            if is_active_schedule and not is_capturing:
                if self.in_maintenance_mode.is_set():
                    # If we are in schedule but not capturing BECAUSE the user paused it,
                    # DO NOTHING. Log a message and let the maintenance timeout or user handle it.
                    logging.info("Scheduler: System is in scheduled active time but is currently in user-initiated Maintenance Mode.")
                elif self.is_calibrating.is_set():
                     # Also, do nothing if a calibration is running outside the schedule time
                     logging.info("Scheduler: System is in scheduled active time but is currently performing a calibration.")
                else:
                    logging.info("Scheduler: Active schedule window has begun. Starting producer.")
                    self.consecutive_restart_failures = 0
                    # --- Record session start time ---
                    self.session_start_time = datetime.now()
                    try:
                        self.watchdog_pause.set()
                        self.start_producer_thread()
                    finally:
                        self.watchdog_pause.clear()

                    logging.info("Scheduler: Requesting an immediate sky condition check.")
                    self.request_sky_analysis_trigger.set()

            elif not is_active_schedule and is_capturing and not self.is_calibrating.is_set():
                logging.info("Scheduler: Active schedule window has ended. Stopping producer.")
                self.consecutive_restart_failures = 0
                try:
                    logging.info("Scheduler: Pausing watchdog for producer stop.")
                    self.watchdog_pause.set()
                    self.stop_producer_thread()
                finally:
                    logging.info("Scheduler: Resuming watchdog.")
                    self.watchdog_pause.clear()
                    # --- Trigger end-of-session video creation ---
                    self.start_timelapse_video_creation()
            
            elif not is_active_schedule and not is_capturing: 
                self.consecutive_restart_failures = 0
                if not self.in_maintenance_mode.is_set() and not self.is_calibrating.is_set():
                    logging.info("Scheduler: Outside the schedule time.")
                               
            # Wait for 60 seconds before the next check.
            for _ in range(60):
                if not self.running.is_set(): break
                time.sleep(1)

    # -----------------
    def producer_thread_active(self):
        """Checks if the single CaptureThread is alive."""
        return any(t.name == "CaptureThread" and t.is_alive() for t in self.worker_threads)

    # -----------------
    def start_producer_thread(self):
        """Starts ONLY the camera capture thread."""
        if any(t.name == "CaptureThread" for t in self.worker_threads if t.is_alive()):
            logging.warning("Capture thread appears to be already running.")
            return

        # Ensure the capture running flag is set before starting
        self.capture_running.set()
        capture_thread = threading.Thread(target=self.capture_loop, name="CaptureThread", daemon=True)
        capture_thread.start()
        self.worker_threads.append(capture_thread)
        logging.info("Producer thread (CaptureThread) started.")
   
    # -----------------
    def stop_producer_thread(self):
        """Stops ONLY the camera capture thread."""
        logging.info("Stopping producer thread (CaptureThread)...")
        self.capture_running.clear() # Signal the loop to exit
        
        # Post a sentinel to the acquisition queue to unblock and terminate the detection loop
        self.acq_q.put(None)

        # Wait for both threads to fully exit
        threads_to_join = ["CaptureThread", "DetectionThread"]
        for t in self.worker_threads:
            if t.name in threads_to_join and t.is_alive():
                logging.info(f"Waiting for {t.name} to terminate...")
                t.join(timeout=5)
        
        # Clean up the dead threads from our list
        self.worker_threads = [t for t in self.worker_threads if t.is_alive()]
        logging.info("Data production chain has stopped.")

    # ----------------- thread management -----------------
    def start(self):
        logging.info("Starting all pipeline services...")
        power_pin = self.cfg["general"].get("power_monitor_pin", 17)

        self.worker_threads = [
            threading.Thread(target=self.detection_loop, name="DetectionThread", daemon=True),
            threading.Thread(target=self.event_writer_loop, name="EventWriterThread", daemon=True),
            threading.Thread(target=self.timelapse_loop, name="TimelapseThread", daemon=True),
            threading.Thread(target=self.event_stacker_loop, name="EventStackerThread", daemon=True),
            threading.Thread(target=self.calibration_worker_loop, name="CalibrationThread", daemon=True)
        ]

        if self.cfg["event_log"].get("enabled", False):
            self.worker_threads.append(threading.Thread(target=self.event_logger_loop, name="EventLoggerThread", daemon=True))
        else:
            logging.info("Event log is disabled in configuration.")        
        
        for t in self.worker_threads:
            t.start()
        logging.info("All worker threads started.")

        # 1. Add mandatory control threads
        self.control_threads = [
            threading.Thread(target=self.watchdog_loop, name="WatchdogThread", daemon=True),
            threading.Thread(target=self.system_monitor_loop, name="MonitorThread", daemon=True),
            threading.Thread(target=self.janitor_loop, name="JanitorThread", daemon=True),
            threading.Thread(target=self.scheduler_loop, name="SchedulerThread", daemon=True),
            threading.Thread(target=self.health_monitor_loop, name="HealthMonitorThread", daemon=True),
            threading.Thread(target=self.config_reloader_loop, name="ConfigReloaderThread", daemon=True)
        ]

        # 2. Add optional control threads based on configuration
        
        if self.cfg["queue_monitor"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.queue_monitor_loop, name="QueueMonitorThread", daemon=True))
        else:
            logging.info("Queue monitor is disabled in configuration.")        
        
        if self.cfg["power_monitor"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.power_monitor_loop, name="PowerMonitorThread", daemon=True))
        else:
            logging.info("Power monitor is disabled in configuration.")
            self.power_status = "OK"  # bypass power control

        if self.cfg["heartbeat"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.heartbeat_loop, name="HeartbeatThread", daemon=True))
        else:
            logging.info("Hearthbeat is disabled in configuration.")

        if self.cfg["ntp"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.ntp_sync_loop, name="NTPThread", daemon=True))
        else:
            logging.info("NTP is disabled in configuration.")

        if self.cfg["daylight"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.sky_monitor_loop, name="SkyMonitorThread", daemon=True))
        else:
            logging.info("Sky monitor is disabled in configuration.")
            
        if self.cfg["dashboard"].get("enabled", False):
            self.control_threads.append(threading.Thread(target=self.dashboard_loop, name="DashboardThread", daemon=True))
            # The maintenance watchdog is tied to the dashboard's pause feature
            self.control_threads.append(threading.Thread(target=self.maintenance_watchdog_loop, name="MaintWatchdogThread", daemon=True))
        else:
            logging.info("Dashboard (and Maintenance Watchdog) is disabled in configuration.")
            
        if self.led_line:
            self.control_threads.append(threading.Thread(target=self.led_status_loop, name="LedStatusThread", daemon=True))
        else: 
            logging.info("Status LED disabled in configuration.")
                        
        for t in self.control_threads:
            t.start()
        logging.info("All control threads started.")       

        # 3. Add SFTP threads if enabled
        if self.sftp_uploader:
            sftp_thread = threading.Thread(target=self.sftp_uploader.worker_loop, name="SftpThread", daemon=True)
            sftp_thread.start()
            self.control_threads.append(sftp_thread) # Add to control threads for clean shutdown
            sftp_dispatcher_thread = threading.Thread(target=self.sftp_dispatcher_loop, name="SFTPDispatchThread", daemon=True)
            sftp_dispatcher_thread.start()
            self.control_threads.append(sftp_dispatcher_thread)
            logging.info("SFTP Uploader thread started.")

            # --- Perform the initial sweep AFTER threads are running but BEFORE capture starts ---
            try:
                logging.info("Performing mandatory startup sweep for un-uploaded files...")
                self.sweep_and_enqueue_unuploaded()
                logging.info("Startup SFTP sweep complete.")
            except Exception as e:
                logging.exception(f"An error occurred during startup SFTP sweep: {e}")
            finally:
                # CRITICAL: Always set the event, even if the sweep fails, to prevent a deadlock.
                self.initial_sftp_sweep_complete.set()
        else:
            # If SFTP is not enabled, we don't need to sweep, so we can proceed immediately.
            self.initial_sftp_sweep_complete.set()
            
    # -----------------
    def stop(self):
        """
        Performs a robust, multi-stage shutdown of all pipeline services.
        """
        if not self.running.is_set():
            logging.warning("Shutdown already in progress.")
            return

        logging.info("="*60)
        logging.info("--- INITIATING GRACEFUL SHUTDOWN ---")
        logging.info("="*60)

        # --- Phase 1: Stop all new data production ---
        self.running.clear() # Global stop signal
        self.stop_producer_thread() # This now handles both Capture and Detection

        # --- Phase 2: Signal all consumer and control threads to terminate ---
        logging.info("Posting final sentinels to all remaining worker and control queues...")
        queues_to_signal = [
            self.timelapse_q, self.event_q, self.event_stack_q,
            self.event_log_q, self.health_q
        ]
        if self.sftp_uploader:
            self.sftp_uploader.stop() # This handles the SFTP queues
        
        for q in queues_to_signal:
            try: q.put_nowait(None)
            except (queue.Full, AttributeError): pass
            
        # --- Phase 3: Wait for all remaining threads to finish cleanly ---
        logging.info("Waiting for all threads to complete their final tasks...")
        all_threads = self.worker_threads + self.control_threads
        shutdown_timeout = 10 # seconds

        for t in all_threads:
            if t.name == "DashboardThread" or not t.is_alive():
                continue
            
            t.join(timeout=shutdown_timeout)
            if t.is_alive():
                logging.error(f"Thread '{t.name}' failed to join and is still alive. Shutdown may be incomplete.")

        # --- Phase 4: Release hardware resources ---
        if self.camera:
            try:
                self.camera.release()
                self.camera = None
            except Exception: pass
            
        if self.led_line:
            try:
                logging.info("Turning off status LED.")
                self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                self.led_line.release()
            except Exception as e:
                logging.error(f"Error releasing LED GPIO pin: {e}")

        logging.info("--- PIPELINE SHUTDOWN COMPLETE ---")

    # ----------------- Heartbeat -----------------
    def heartbeat_loop(self):
        """
        Periodically sends a heartbeat ping to a configured URL.
        Crucially, this thread runs INDEPENDENTLY of the main capture schedule
        to signal that the script process is alive at all times.
        """
        hb_cfg = self.cfg.get("heartbeat", {})

        url = hb_cfg.get("url")
        active_interval_min = hb_cfg.get("interval_min", 60)
        # Use a different, potentially longer interval for when the system is idle
        idle_interval_min = self.cfg["general"].get("idle_heartbeat_interval_min", 120) # e.g., 2 hours

        if not url:
            logging.warning("Heartbeat is enabled, but no valid URL is configured.")
            return
        
        try:
            import requests
        except ImportError:
            logging.error("'requests' library is required for the heartbeat feature. Please run 'pip3 install requests'.")
            return

        logging.info(f"Heartbeat monitor started. Will ping every {active_interval_min} min (active) or {idle_interval_min} min (idle).")

        while self.running.is_set():
            # 1. Determine the current state and wait interval
            is_active = self.producer_thread_active()
            current_interval_sec = (active_interval_min if is_active else idle_interval_min) * 60
            
            # Wait for the determined interval
            for _ in range(current_interval_sec):
                if not self.running.is_set(): break
                time.sleep(1)
            if not self.running.is_set(): break

            # 2. Send the ping
            try:
                # You can append a status to the URL for services that support it
                # For Healthchecks.io, the main URL is enough.
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                status_msg = "OK (Active)" if is_active else "OK (Idle)"
                with self.status_lock:
                    self.last_heartbeat_status = f"{status_msg} ({datetime.now().strftime('%H:%M:%S')})"
                logging.info(f"Heartbeat ping sent successfully (State: {'Active' if is_active else 'Idle'}).")

            except requests.exceptions.RequestException as e:
                with self.status_lock:
                    self.last_heartbeat_status = f"FAIL ({datetime.now().strftime('%H:%M:%S')})"
                logging.warning(f"Failed to send heartbeat ping: {e}")
                self.log_health_event("WARNING", "HEARTHBEAT", "Failed to send heartbeat ping")

    # ----------------- Event Logger -----------------
    def event_logger_loop(self):
        """
        A dedicated thread that waits for event summary dictionaries and appends
        them as a new line in a structured CSV file.
        """
        log_cfg = self.cfg.get("event_log", {})
        os.makedirs(os.path.dirname(self.event_log_out_file), exist_ok=True)
        
        # Write the CSV header if the file is new or empty
        if not os.path.exists(self.event_log_out_file) or os.path.getsize(self.event_log_out_file) == 0:
            try:
                with open(self.event_log_out_file, "w") as f:
                    f.write("timestamp_utc,start_time_utc,duration_sec,num_frames_video,num_frames_stack\n")
            except IOError as e:
                logging.error("Could not write header to event log file '%s': %s", self.event_log_out_file, e)
                return

        logging.info("Event logger started. Will append summaries to %s", self.event_log_out_file)

        while True:
            try:
                # This is a blocking call, so the thread uses zero CPU while waiting
                log_data = self.event_log_q.get()

                # Check for the sentinel to exit
                if log_data is None:
                    logging.info("Event logger received sentinel. Exiting.")
                    break
                
                # Format the data into a CSV line
                line = (f"{log_data['timestamp_utc']},{log_data['start_time_utc']},"
                        f"{log_data['duration_sec']:.2f},{log_data['num_frames_video']},"
                        f"{log_data['num_frames_stack']}\n")

                with open(self.event_log_out_file, "a") as f:
                    f.write(line)

                # Increment and display the event counter
                self.event_counter += 1
                logging.info("Event #%d logged successfully. Total events captured: %d", self.event_counter, self.event_counter)

            except Exception as e:
                logging.exception("Error in event_logger_loop: %s", e)        

    # ----------------- detection -----------------
    def detection_loop(self):
        logging.info("Detection loop started")
        
        fps = self.cfg["capture"].get("fps", 2)
        pre_event_sec = self.cfg["detection"].get("pre_event_sec", 2)
        buffer_frames = int(max(1, math.ceil(pre_event_sec * max(0.5, fps))))
        recent_frames = deque(maxlen=buffer_frames)

        # --- State Management for creating two different packages ---
        in_event = False
        current_event_video_frames = []
        current_event_stack_frames = []
        event_cooldown_frames = int(fps * self.cfg["detection"].get("event_cooldown_sec")) # e.g., 2 seconds of no detection
        event_cooldown_counter = 0

        while True:
            try:
                item = self.acq_q.get()
                if item is None:
                    # If shutdown happens mid-event, finalize and dispatch what we have
                    if in_event and current_event_video_frames:
                        logging.info("Event in progress during shutdown. Finalizing packages.")
                        self.event_q.put(current_event_video_frames)
                        self.event_stack_q.put(current_event_stack_frames)
                    logging.info("Detection loop received sentinel. Draining complete, exiting.")
                    break
                ts, frame = item
            except queue.Empty:
                continue

            alpha = self.cfg["detection"].get("accumulate_alpha", 0.01)
            thr = self.cfg["detection"].get("threshold", 25)
            min_area = self.cfg["detection"].get("min_area", 50)

            if self.master_dark is not None:
                frame = cv2.subtract(frame, self.master_dark)

            # 1a. Check for calibration trigger
            if self.is_calibrating.is_set():
                try:
                    self.calibration_q.put_nowait((ts, frame.copy()))
                except queue.Full:
                    logging.warning("Calibration queue is full.")
                continue

            # 1. First, pass the frame to the unconditional timelapse.
            try:
                self.timelapse_q.put_nowait((ts, frame))
            except queue.Full:
                logging.warning("Timelapse queue is full, dropping frame.")                

            # 2. Now, check if we should even attempt the meteor hunt.
            if self.daylight_mode_active.is_set():
                continue # In Daylight Mode, skip detection.
            
            if self.weather_hold_active.is_set():
                continue # In Weather Hold, skip detection.

            # --- Detection Logic ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), 0)
            if self.background is None:
                self.background = gray.astype("float32")
                recent_frames.append((ts, frame.copy()))
                continue
            
            cv2.accumulateWeighted(gray, self.background, alpha)
            bg = cv2.convertScaleAbs(self.background)
            diff = cv2.absdiff(gray, bg)
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                       
            # Finds the contours and creates the 'cnts' variable.
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected = any(cv2.contourArea(c) >= min_area for c in cnts)
            
            # The pre-event buffer always contains the most recent frames.
            recent_frames.append((ts, frame.copy()))

            if detected:
                if not self.event_in_progress.is_set():
                    # --- EVENT START ---
                    logging.info("New event started at %s (contours=%d)", ts, len(cnts))
                    self.event_in_progress.set()
                    # The video package starts with the complete pre-event buffer.
                    current_event_video_frames = list(recent_frames)
                    # The stack package starts clean, with ONLY the first detected frame.
                    current_event_stack_frames = [(ts, frame.copy())]
                else:
                    # --- EVENT CONTINUES ---
                    # Add the current frame to both packages.
                    current_event_video_frames.append((ts, frame.copy()))
                    current_event_stack_frames.append((ts, frame.copy()))
                
                event_cooldown_counter = 0 # Reset cooldown on new detection
            
            elif self.event_in_progress.is_set():
                # --- EVENT COOLDOWN ---
                event_cooldown_counter += 1
                # Add cooldown frames to the video for context.
                current_event_video_frames.append((ts, frame.copy()))
                # DO NOT add cooldown frames to the stack, to keep it clean.
                
                # Read the cooldown from the (potentially reloaded) config every time.
                event_cooldown_sec = self.cfg["detection"].get("event_cooldown_sec", 2)
                event_cooldown_frames = int(fps * event_cooldown_sec)                
                
                if event_cooldown_counter > event_cooldown_frames:
                    # --- EVENT END ---
                    logging.info("Event finished. Dispatching packages (Video: %d frames, Stack: %d frames).", 
                                  len(current_event_video_frames), len(current_event_stack_frames))
                    try:
                        # Send the two specialized packages to their consumers.
                        self.event_q.put_nowait(current_event_video_frames)
                        self.event_stack_q.put_nowait(current_event_stack_frames)
                        
                        # Create and dispatch the summary dictionary for the event logger
                        if self.cfg.get("event_log", {}).get("enabled", True):
                            start_dt = datetime.fromisoformat(current_event_video_frames[0][0])
                            end_dt = datetime.fromisoformat(current_event_video_frames[-1][0])
                            log_summary = {
                                "timestamp_utc": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "start_time_utc": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "duration_sec": (end_dt - start_dt).total_seconds(),
                                "num_frames_video": len(current_event_video_frames),
                                "num_frames_stack": len(current_event_stack_frames)
                            }
                            self.event_log_q.put_nowait(log_summary)                        
                        
                    except queue.Full:
                        logging.warning("An event queue was full. Data for this event was dropped.")
                    
                    # Reset state for the next event.
                    self.event_in_progress.clear()
                    current_event_video_frames = []
                    current_event_stack_frames = []                    
                    event_cooldown_counter = 0

    # ----------------- event writer -----------------
    def event_writer_loop(self):
        logging.info("Event writer waiting for event packages.")
        out_dir = self.events_out_dir

        while True:
            
            fps = self.cfg["events"]["video_fps"]
            encoder = self.cfg["events"]["ffmpeg_encoder"]
            bitrate = self.cfg["events"]["ffmpeg_bitrate"]
            jpeg_quality = self.cfg["events"]["jpeg_quality"]            
            
            try:
                # Wait for a complete package of frames
                frames = self.event_q.get()
                if frames is None:
                    logging.info("Event writer received sentinel. Exiting.")
                    break
            except queue.Empty:
                continue
            
            if not frames:
                logging.info("Received an empty event package, ignoring.")
                continue

            tstamp = datetime.fromisoformat(frames[-1][0]).strftime("%Y%m%dT%H%M%SZ")
            base_name = f"event_{tstamp}"
            tmp_dir = os.path.join(out_dir, base_name + "_frames")
            self.ensure_dir(tmp_dir)

            logging.info("Encoding event video to %s.mp4 (%d frames)", base_name, len(frames))

            try:
                for i, (ts, frame) in enumerate(frames):
                    fname = os.path.join(tmp_dir, f"f_{i:05d}.jpg")
                    cv2.imwrite(fname, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

                video_path = os.path.join(out_dir, base_name + ".mp4")
                ff_cmd = [
                    "ffmpeg", "-y", "-framerate", str(fps),
                    "-i", os.path.join(tmp_dir, "f_%05d.jpg"),
                    "-c:v", encoder, "-b:v", bitrate, "-pix_fmt", "yuv420p", video_path
                ]
                
                # timeout 300 seconds
                subprocess.run(ff_cmd, check=True, capture_output=True, text=True, timeout=300)
                self.save_event_metadata(video_path, frames)
                logging.info("Event video and metadata saved: %s", video_path)
                
                # Update the dashboard state with the path to the new video
                with self.status_lock:
                    self.last_event_files["video"] = video_path
                
                if self.sftp_uploader and self.running.is_set():
                    logging.info(f"Dispatching {os.path.basename(video_path)} and metadata for upload.")
                    self.sftp_dispatch_q.put(video_path)
                    self.sftp_dispatch_q.put(os.path.splitext(video_path)[0] + ".json")

            except subprocess.TimeoutExpired:
                logging.error("FFmpeg process timed out for %s. The process was killed. The event video was not saved.", base_name)
                self.log_health_event("ERROR", "FFMPEG_FAIL", "Process timed out")
            except subprocess.CalledProcessError as e:
                logging.error("--- FFMPEG FAILED ---")
                logging.error("FFmpeg Command: %s", " ".join(e.cmd))
                logging.error("FFmpeg stderr:\n%s", e.stderr)
                if e.stderr: logging.error("FFmpeg stderr:\n%s", e.stderr.strip())
                self.log_health_event("ERROR", "FFMPEG_FAIL", f"Event video {base_name} failed.")
            except Exception as e:
                logging.exception("An error occurred in event_writer_loop: %s", e)
            finally:
                if os.path.isdir(tmp_dir):
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass

    # ----------------- Config Reloader -----------------
    def config_reloader_loop(self):
        """
        A dedicated thread that waits for a signal to reload the config file.
        It intelligently compares the new config to the old one, pauses the watchdog,
        and restarts critical threads if necessary.
        """
        logging.info("Smart configuration reloader thread started.")

        CRITICAL_CAPTURE_PARAMS = ["width", "height", "fps", "exposure_us", "gain", "auto_exposure", "red_gain", "blue_gain"]
        CRITICAL_POWER_PARAMS = ["pin"] # Pin for the power monitor

        last_cfg = json.loads(json.dumps(self.cfg)) 

        while self.running.is_set():
            if self.reload_config_signal.wait(timeout=10.0):
                try:
                    logging.warning("Reload signal received. Performing smart configuration reload.")
                    new_cfg = self.load_and_validate_config(self.config_path)

                    # --- Comparison Phase ---
                    requires_capture_restart = False
                    for param in CRITICAL_CAPTURE_PARAMS:
                        old_val = last_cfg.get("capture", {}).get(param)
                        new_val = new_cfg.get("capture", {}).get(param)
                        if old_val != new_val:
                            logging.critical(f"CRITICAL CHANGE DETECTED: 'capture.{param}' changed from '{old_val}' to '{new_val}'.")
                            requires_capture_restart = True
                            break
                    
                    power_pin_changed = False
                    for param in CRITICAL_POWER_PARAMS:
                        old_val = last_cfg.get("power_monitor", {}).get(param)
                        new_val = new_cfg.get("power_monitor", {}).get(param)
                        if old_val != new_val:
                            logging.critical(f"CRITICAL CHANGE DETECTED: 'power_monitor.{param}' changed from '{old_val}' to '{new_val}'.")
                            power_pin_changed = True
                            break

                    # --- Action Phase ---
                    if requires_capture_restart or power_pin_changed:
                        try:
                            if power_pin_changed:
                                msg="!!! Power monitor pin changed. !!!"
#                                logging.critical("="*60)
#                                logging.critical("!!! A CRITICAL PARAMETER HAS CHANGED. !!!")
#                                logging.critical("!!! A full service restart is required to apply this change safely. !!!")
#                                logging.critical("!!! The program restart automatically !!!")
#                                logging.critical("="*60)
                                self.perform_system_action("exit", reason=msg)
                                # The code will not proceed past this point.                           
                                                       
                            # --- PAUSE WATCHDOG ---
                            logging.warning("Pausing watchdog for thread restart procedure.")
                            self.watchdog_pause.set()
                            time.sleep(1) # Grace period

                            if requires_capture_restart:
                                logging.warning("Initiating graceful restart of the capture thread...")
                                self.stop_producer_thread()
                                time.sleep(2)
                                
                                # Apply new config BEFORE re-initializing camera
                                with self.status_lock:
                                    self.cfg = new_cfg

                                try:
                                    self.camera.release()
                                    self.camera = CameraCapture(self.cfg["capture"])
                                    logging.info("Camera re-initialized successfully with new settings.")
                                except Exception:
                                    logging.critical("Failed to re-initialize camera! This is a fatal error.", exc_info=True)
                                    self.perform_system_action("exit", reason="Fatal camera re-initialization failure")
                                
                                self.start_producer_thread()

                        finally:
                            # --- RESUME WATCHDOG (CRITICAL) ---
                            logging.info("Resuming watchdog checks.")
                            self.watchdog_pause.clear()
                    
                    # If no restarts were needed, just apply the config live.
                    if not requires_capture_restart:
                        logging.info("No critical changes detected that require restarts. Applying new settings live.")
                        with self.status_lock:
                            self.cfg = new_cfg
                    
                    # Update last known config for the next comparison
                    last_cfg = json.loads(json.dumps(new_cfg)) 
                    logging.info("Smart configuration reload complete.")
                    self.config_reloaded_ack.set()

                except Exception as e:
                    logging.exception("Failed to reload configuration: %s", e)
                finally:
                    self.reload_config_signal.clear()

    # ----------------- Web Dashboard -----------------
    def dashboard_loop(self):
        """A dedicated thread to run a Flask web server for status and control."""
        dash_cfg = self.cfg.get("dashboard", {})

        try:
            from flask import Flask, send_from_directory, redirect, url_for, request, render_template_string, Response, abort, jsonify
            from functools import wraps
            from datetime import timedelta
        except ImportError:
            logging.error("Flask library not found. 'pip3 install Flask'. Dashboard disabled.")
            return

        # 1. Read the token from the environment.
        DASH_TOKEN = os.environ.get("METEORA_DASH_TOKEN")

        # 2. If the token is not set or is empty, refuse to start.
        if not DASH_TOKEN:
            logging.critical("="*60)
            logging.critical("!!! SECURITY ALERT: METEORA_DASH_TOKEN is NOT SET. !!!")
            logging.critical("!!! All access to the dashboard will be DENIED as a security precaution. !!!")
            logging.critical("!!! To enable the dashboard, set this environment variable to a long, random string. !!!")
            logging.critical("="*60)
        else:
            logging.info("Dashboard authentication is ENABLED. Token has been loaded.")
          
        def require_token(f):
            @wraps(f)
            def inner(*args, **kwargs):
                # Rule 1: If the DASH_TOKEN was never configured on the server, deny all access.
                if not DASH_TOKEN:
                    abort(401) # Access Denied

                # Rule 2: If the token IS configured, the user must provide the correct one.
                token_header = request.headers.get("X-Auth-Token")
                token_query = request.args.get("token")

                if token_query == DASH_TOKEN or token_header == DASH_TOKEN:
                    return f(*args, **kwargs) # Access Granted
                
                # If we reach here, the token was wrong or missing.
                logging.warning(f"Dashboard: Unauthorized access attempt from {request.remote_addr}.")
                abort(401) # Access Denied
            return inner

        app = Flask(__name__, static_folder=os.path.abspath("output"))
        
        # --- PARAMETER ---
        EDITABLE_PARAMS = {
            "capture": ["fps", "exposure_us", "gain", "red_gain", "blue_gain"],
            "detection": ["min_area", "threshold", "pre_event_sec", "event_cooldown_sec"],
            "timelapse": ["stack_N", "interval_sec"],
            "daylight": ["stddev_threshold", "min_stars", "lux_threshold"],
        }
        TIMELAPSE_DIR = os.path.abspath(self.timelapse_out_dir)
        EVENTS_DIR = os.path.abspath(self.events_out_dir)
        ALLOWED_DIRS = [TIMELAPSE_DIR, EVENTS_DIR]

        # --- API Endpoints ---
        
        # --- START: SYSTEM TIME PAGE ---
        @app.route('/system_time', methods=['GET', 'POST'])
        @require_token
        def system_time():
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
            
            if request.method == 'POST':
                try:
                    new_date = request.form.get('new_date')
                    new_time = request.form.get('new_time')
                    if not new_date or not new_time:
                        raise ValueError("Date and time fields cannot be empty.")

                    # Combine and format for the `date` command
                    datetime_str = f"{new_date} {new_time}"
                    logging.warning(f"Dashboard user is attempting to set system time to: {datetime_str}")
                    
                    # Securely execute the command
                    cmd = ["sudo", "date", "-s", datetime_str]
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
                    
                    logging.info(f"System time successfully set. Command output: {result.stdout.strip()}")
                    return redirect(url_for('system_time', token=DASH_TOKEN, success='true'))
                
                except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    error_message = str(e)
                    if hasattr(e, 'stderr'):
                        error_message = e.stderr.strip()
                    logging.error(f"Failed to set system time: {error_message}")
                    # URL-encode the error message for safe transport
                    from urllib.parse import quote
                    return redirect(url_for('system_time', token=DASH_TOKEN, error=quote(error_message)))

            # Handle GET request (display the page)
            now = datetime.now()
            current_date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            
            feedback_html = ""
            if request.args.get('success'):
                feedback_html = '<div class="banner-success">System time updated successfully!</div>'
            elif request.args.get('error'):
                error_msg = html.escape(request.args.get('error'))
                feedback_html = f'<div class="banner-error"><strong>Error:</strong> {error_msg}<br><small>Ensure /bin/date is in your sudoers file.</small></div>'
                
            html_page = f"""
            <!DOCTYPE html><html lang="en">
            <head>
              <meta charset="UTF-8"><title>Set System Time - Meteora</title>
              <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }}
                header {{ background: #2c2c2c; padding: 1em 2em; border-bottom: 1px solid #444; }}
                h1 {{ margin: 0; color: #61afef; }}
                .main {{ padding: 2em; max-width: 900px; margin: auto; }}
                .card {{ background: #2b2b2b; border-radius: 12px; padding: 1.5em; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }}
                label {{ font-weight: bold; display: block; margin-bottom: 5px; }}
                input[type=date], input[type=time] {{ background: #1a1a1a; border: 1px solid #444; color: #e0e0e0; padding: 10px; border-radius: 4px; font-size: 1.1em; margin-bottom: 1em; }}
                button {{ background: #e5c07b; color: #1e1e1e; font-weight: bold; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-size: 1em; margin-top: 1em; }}
                button:hover {{ background: #ffd791; }}
                a {{ color: #c678dd; }}
                .banner-success {{ background-color: #2c5f2d; color: #fff; padding: 1em; border-radius: 8px; margin-bottom: 1em; text-align: center; }}
                .banner-error {{ background-color: #8b0000; color: #fff; padding: 1em; border-radius: 8px; margin-bottom: 1em; }}
              </style>
            </head>
            <body>
              <header><h1>Set System Time</h1></header>
              <div class="main">
                <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                {feedback_html}
                <div class="card">
                  <h2>Current System Time: {current_date} {current_time}</h2>
                  <p style="color: #e5c07b;">Use this page to manually correct the system clock if the device has booted without an internet connection for NTP sync.</p>
                  <form method="post" action="/system_time{token_param}" onsubmit="return confirm('Are you sure you want to change the system time? This can affect logging and file timestamps.');">
                    <div>
                        <label for="new_date">New Date:</label>
                        <input type="date" id="new_date" name="new_date" value="{current_date}">
                    </div>
                    <div>
                        <label for="new_time">New Time (UTC):</label>
                        <input type="time" id="new_time" name="new_time" value="{current_time}" step="1">
                    </div>
                    <button type="submit">Set New Time</button>
                  </form>
                </div>
              </div>
            </body></html>
            """
            return render_template_string(html_page)

        # --- END: SYSTEM TIME PAGE ---
        
        # --- Pause the aquisition.
        @app.route('/api/pause')
        @require_token
        def pause_capture():
            logging.warning("Pause command received from dashboard. Stopping worker threads.")
            self.stop_producer_thread()
            self.in_maintenance_mode.set()
            with self.maintenance_timeout_lock:
                self.maintenance_timeout_until = time.time() + self.maintenance_timeout_duration_sec            
            return redirect(url_for('dashboard', token=DASH_TOKEN))
        
        # --- Resunme the aquisition.
        @app.route('/api/resume')
        @require_token
        def resume_capture():
            logging.warning("Resume command received from dashboard. Restarting worker threads.")
            self.start_producer_thread()
            self.in_maintenance_mode.clear()
            with self.maintenance_timeout_lock:
                self.maintenance_timeout_until = 0            
            return redirect(url_for('dashboard', token=DASH_TOKEN))
        
        # --- Show EDITABLE_PARAMS, save and test for image test 
        @app.route('/api/save_and_test', methods=['POST'])
        @require_token
        def save_and_test():
            logging.info("Dashboard: Received /api/save_and_test request.")
            if self.producer_thread_active():
                logging.warning("Dashboard: Capture threads ARE active, skipping test shot trigger.")
                return redirect(url_for('dashboard', token=DASH_TOKEN))

            try:
                # This ensures the dashboard doesn't show stale error messages or images.
                with self.status_lock:
                    self.last_calibration_error = None
                    self.last_calibration_image_path = None
                
                # refresh maintenance timeout, if saved new parameter, while wait the frame for the test image, reset the timeout
                with self.maintenance_timeout_lock:
                    self.maintenance_timeout_until = time.time() + self.maintenance_timeout_duration_sec
                
                logging.info("Dashboard: Saving new configuration from dashboard.")
                
                # --- START: NEW, ROBUST SAVE LOGIC ---
                
                # 1. Create a dictionary of the changes from THIS form only.
                changes_from_form = {}
                for section, params in EDITABLE_PARAMS.items():
                    for param in params:
                        form_key = f"{section}_{param}"
                        if form_key in request.form:
                            if section not in changes_from_form:
                                changes_from_form[section] = {}
                            changes_from_form[section][param] = request.form[form_key]

                # 2. Validate these changes using Pydantic. This will also coerce types.
#                if IS_PYDANTIC_AVAILABLE:
                try:
                    # Pydantic validates the partial dict against the full model
                    validated_changes = MainConfig.model_validate(changes_from_form).model_dump(exclude_unset=True)
                except Exception as e:
                    logging.error(f"Dashboard: New configuration failed validation: {e}")
                    # You could add an error message to the dashboard here
                    return redirect(url_for('dashboard', token=DASH_TOKEN))
#                else:
#                    # Fallback if Pydantic is not installed (less safe type handling)
#                    validated_changes = changes_from_form
                
                # 3. Load the existing MINIMAL config from disk.
                config_on_disk = self.legacy_load_config(self.config_path, merge=False)
                if not isinstance(config_on_disk, dict): config_on_disk = {}

                # 4. Recursively update the on-disk config with the validated changes.
                def recursive_update(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = recursive_update(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d
                
                config_to_save = recursive_update(config_on_disk, validated_changes)
                
                # 5. Save the updated minimal config and reload.
                self.save_json_atomic(self.config_path, config_to_save)
                self.reload_config_signal.set()
                if not self.config_reloaded_ack.wait(timeout=5.0):
                    logging.error("Dashboard: Timed out waiting for config reloader ack.")
                self.config_reloaded_ack.clear()
                
                # --- END: NEW SAVE LOGIC ---
                
                # 6. Start the calibration process.
                if self.is_calibrating.is_set():
                    logging.warning("Dashboard: Calibration is already in progress.")
                else:
                    logging.info("Dashboard: Setting calibration mode and starting producer thread...")
                    self.is_calibrating.set()
                    self.start_producer_thread()

            except Exception as e:
                logging.exception("A critical error occurred while saving the configuration: %s", e)

            return redirect(url_for('dashboard', token=DASH_TOKEN))

        @app.route('/calibration_image')
        @require_token
        # --- The image coming from the calibration settings
        def serve_calibration_image():
            with self.status_lock:
                path = self.last_calibration_image_path
            if path and os.path.exists(path):
                return send_from_directory(os.path.dirname(path), os.path.basename(path))
            return "No image available.", 404

        @app.route('/files/action', methods=['POST'])
        @require_token
        # --- Download and delete file from files page
        def file_action():
            selected_files = request.form.getlist('selected_files')
            action = request.form.get('action')

            if not selected_files:
                logging.warning("File action requested but no files were selected.")
                return redirect(url_for('file_manager', token=DASH_TOKEN))

            # --- SECURITY VALIDATION ---
            for f_path in selected_files:
                if not self.is_safe_path(ALLOWED_DIRS, f_path):
                    logging.error("SECURITY ALERT: Attempted action on unsafe path: %s", f_path)
                    return "Forbidden: Path is outside of allowed directories.", 403

            if action == 'delete':
                logging.warning("User initiated deletion of %d file(s) from dashboard.", len(selected_files))
                deleted_count = 0
                for f_path in selected_files:
                    try:
                        os.remove(f_path)
                        # Also try to remove associated .json file
                        json_path = os.path.splitext(f_path)[0] + ".json"
                        if os.path.exists(json_path):
                            os.remove(json_path)
                        deleted_count += 1
                    except OSError as e:
                        logging.error("Failed to delete file %s: %s", f_path, e)
                logging.info("Successfully deleted %d file(s).", deleted_count)
                return redirect(url_for('file_manager', token=DASH_TOKEN))

            elif action == 'download':
                if len(selected_files) == 1:
                    # Download a single file directly
                    f_path = selected_files[0]
                    try:
                        directory = os.path.dirname(f_path)
                        filename = os.path.basename(f_path)
                        return send_from_directory(directory, filename, as_attachment=True)
                    except FileNotFoundError:
                        logging.error("Attempted to download a file that does not exist: %s", f_path)
                        return "File not found.", 404                        
                else:
                    # Zip multiple files and download
                    memory_file = io.BytesIO()
                    with ZipFile(memory_file, 'w') as zipf:
                        for f_path in selected_files:
                            try:
                                # This operation is now protected.
                                if os.path.exists(f_path):
                                    zipf.write(f_path, arcname=os.path.basename(f_path))
                                else:
                                    logging.warning("Skipping missing file during zip creation: %s", f_path)
                            except Exception as e:
                                # Catch any other file-related errors (e.g., permissions)
                                logging.error("Error adding file to zip archive %s: %s", f_path, e)
                    memory_file.seek(0)
                    
                    # If we added at least one file, send the zip. Otherwise, redirect.
                    if len(zipf.infolist()) > 0:
                        return Response(
                            memory_file,
                            mimetype='application/zip',
                            headers={'Content-Disposition': 'attachment;filename=meteora_download.zip'}
                        )
                    else:
                        logging.warning("Multi-file download requested, but no valid files could be zipped.")
                        return redirect(url_for('file_manager', token=DASH_TOKEN))
            
            return "Invalid action.", 400

        # --- Settings page ---
        @app.route('/api/save_settings', methods=['POST'])
        @require_token
        def save_settings():
            logging.warning("Received request to save full configuration from dashboard.")
            try:
                # --- START: NEW, CORRECTED LOGIC ---

                # 1. Build a dictionary from the multi-part form data.
                # This correctly handles checkboxes by taking the last value ('true' if checked).
                form_dict = {}
                for key in request.form.keys():
                    parts = key.split('.')
                    d = form_dict
                    for part in parts[:-1]:
                        d = d.setdefault(part, {})
                    # For checkboxes, request.form.getlist(key) will be ['false', 'true'].
                    # We take the last one, which is the correct state.
                    d[parts[-1]] = request.form.getlist(key)[-1]

                # 2. Validate the data from the form. Pydantic will handle type coercion.
                try:
                    validated_model = MainConfig.model_validate(form_dict)
                    final_cfg = validated_model.model_dump()
                except Exception as e:
                    logging.error(f"New configuration failed validation: {e}")
                    return f"Configuration validation failed. Check logs. <a href='/settings?token={DASH_TOKEN}'>Go back</a>"

                # 3. Load the user's original config file to use as a structural template.
                config_to_save = self.legacy_load_config(self.config_path, merge=False)
                if not isinstance(config_to_save, dict): config_to_save = {}
                
                # 4. Recursively update the template with new values from the validated form data.
                #    This preserves the exact structure of the user's original config file.
                def update_existing_keys(original_dict, new_full_dict):
                    for key, value in original_dict.items():
                        if key in new_full_dict:
                            if isinstance(value, dict) and isinstance(new_full_dict.get(key), dict):
                                # Recurse into nested dictionaries
                                update_existing_keys(value, new_full_dict[key])
                            else:
                                # Update the value in the original dictionary
                                original_dict[key] = new_full_dict[key]

                update_existing_keys(config_to_save, final_cfg)
                
                # 5. Save the updated configuration and trigger a reload.
                self.save_json_atomic(self.config_path, config_to_save)
                self.reload_config_signal.set()
                self.config_reloaded_ack.wait(timeout=3.0)
                self.config_reloaded_ack.clear()
                logging.info("Successfully saved updated configuration file and reloaded.")
                return redirect(url_for('settings', token=DASH_TOKEN, save_success='true'))

            except Exception as e:
                logging.exception("A critical error occurred in /api/save_settings: %s", e)
                return f"An internal error occurred. <a href='/settings?token={DASH_TOKEN}'>Go back</a>"
                
        def generate_form_fields(data, prefix=''):
            html_out = ""
            # Iterate in dictionary order, to match DEFAULTS structure
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    html_out += f'<tr><th colspan="2" style="padding-top: 1em;">{key.replace("_", " ").upper()}</th></tr>'
                    html_out += generate_form_fields(value, prefix=current_prefix)
                else:
                    if isinstance(value, bool):
                        # For booleans (read-only), generate a label WITHOUT the 'for' attribute
                        # This severs the link that allows the label to toggle the checkbox
                        # for future use, actually yhe checkboxes are disabled
                        label_cell = f'<td><label>{key}</label></td>'
                        input_cell = f'''
                        <td class="readonly-cell">
                            <input type="hidden" name="{current_prefix}" value="false">
                            <input type="checkbox" id="{current_prefix}" name="{current_prefix}" value="true" {"checked" if value else ""}>
                        </td>
                        '''
                    else:
                        # For all other editable types, keep the 'for' attribute for better UX
                        label_cell = f'<td><label for="{current_prefix}">{key}</label></td>'
                        val_str = html.escape(str(value if value is not None else ""))
                        input_cell = f'<td><input type="text" id="{current_prefix}" name="{current_prefix}" value="{val_str}" size="40"></td>'
                    
                    html_out += f'<tr>{label_cell}{input_cell}</tr>'
            return html_out

        @app.route('/settings')
        @require_token
        def settings():
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
            
            current_merged_config = self.load_and_validate_config(self.config_path, merge=False)
            form_fields = generate_form_fields(current_merged_config)

            save_success_banner = ""
            if request.args.get('save_success') == 'true':
                save_success_banner = '<div class="banner-success">Configuration saved and reloaded successfully!</div>'

            html_page = f"""
            <!DOCTYPE html><html lang="en">
            <head>
              <meta charset="UTF-8"><title>Configuration Editor</title>
              <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }}
                header {{ background: #2c2c2c; padding: 1em 2em; border-bottom: 1px solid #444; }}
                h1 {{ margin: 0; color: #61afef; }}
                .main {{ padding: 2em; max-width: 900px; margin: auto; }}
                .card {{ background: #2b2b2b; border-radius: 12px; padding: 1.5em; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                td, th {{ padding: 8px 10px; border-bottom: 1px solid #444; text-align: left; }}
                th {{ color: #98c379; background: #2c2c2c; }}
                label {{ font-weight: bold; }}
                input[type=text] {{ background: #1a1a1a; border: 1px solid #444; color: #e0e0e0; padding: 6px; border-radius: 4px; width: 90%; }}
                input[type=checkbox] {{ transform: scale(1.3); }}
                button {{ background: #61afef; color: #fff; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-size: 1em; margin-top: 1em; }}
                button:hover {{ background: #4fa3d8; }}
                a {{ color: #c678dd; }}
                .banner-success {{ background-color: #2c5f2d; color: #97f097; padding: 1em; border-radius: 8px; margin-bottom: 1em; text-align: center; }}
                .readonly-cell {{ pointer-events: none; opacity: 0.6; }}
              </style>
            </head>
            <body>
              <header><h1>Configuration Editor</h1></header>
              <div class="main">
                <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                {save_success_banner}
                <div class="card">
                  <p style="color: #ccc; font-size: 0.9em;">This form shows the complete, active configuration.</p>
                  <p style="color: #e5c07b; font-size: 0.9em;"><b>Note:</b> checkbox are read-only.</p>
                  <form action="/api/save_settings{token_param}" method="post">
                    <table>{form_fields}</table>
                    <button type="submit">Save and Reload Configuration</button>
                    <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                  </form>
                </div>
              </div>
            </body></html>
            """
            return render_template_string(html_page)

        @app.route('/files')
        @require_token
        # --- Files page
        def file_manager():
            event_files = self.get_file_list(EVENTS_DIR)
            timelapse_files = self.get_file_list(TIMELAPSE_DIR)
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
            
            def generate_table_rows(files):
                rows = ""
                for f in files:
                    rows += f"""
                    <tr>
                        <td><input type="checkbox" name="selected_files" value="{f['path']}"></td>
                        <td>{f['name']}</td>
                        <td>{f['size_mb']}</td>
                        <td>{f['mtime']}</td>
                    </tr>
                    """
                return rows

            html_page = f"""
            <!DOCTYPE html><html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>File Manager - Meteora Pipeline</title>
              <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }}
                header {{ background: #2c2c2c; padding: 1em 2em; border-bottom: 1px solid #444; }}
                h1 {{ margin: 0; color: #61afef; }}
                .main {{ padding: 2em; display: grid; grid-template-columns: 1fr; gap: 20px; }}
                .card {{ background: #2b2b2b; border-radius: 12px; padding: 1.5em; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }}
                .card h2 {{ margin-top: 0; font-size: 1.2em; color: #98c379; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 1em; }}
                th, td {{ padding: 8px 10px; border-bottom: 1px solid #444; }}
                th {{ text-align: left; color: #61afef; background: #2c2c2c; }}
                .ok {{ color: #98c379; font-weight: bold; }}
                .warn {{ color: #e5c07b; font-weight: bold; }}
                .err {{ color: #e06c75; font-weight: bold; }}                
                input[type=checkbox] {{ transform: scale(1.2); }}
                button {{ background: #61afef; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 0.9em; margin: 10px 5px 0 0; transition: 0.2s; }}
                button:hover {{ background: #4fa3d8; }}
                .btn-danger {{ background: #e06c75; }}
                .btn-danger:hover {{ background: #c65c66; }}
                a {{ color: #c678dd; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
              </style>
            </head>
            <body>
              <header>
                <h1>File Manager</h1>
              </header>
              <div class="main">
                <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                <form action="/files/action{token_param}" method="post">
                  <div class="card">
                    <h2>Events ({len(event_files)} files)</h2>
                    <table>
                      <thead>
                        <tr><th>Select</th><th>Filename</th><th>Size (MB)</th><th>Date Modified</th></tr>
                      </thead>
                      <tbody>{generate_table_rows(event_files)}</tbody>
                    </table>
                  </div>
                  <div class="card">
                    <h2>Timelapse ({len(timelapse_files)} files)</h2>
                    <table>
                      <thead>
                        <tr><th>Select</th><th>Filename</th><th>Size (MB)</th><th>Date Modified</th></tr>
                      </thead>
                      <tbody>{generate_table_rows(timelapse_files)}</tbody>
                    </table>
                  </div>
                  <div>
                    <button type="submit" name="action" value="download">Download Selected</button>
                    <button type="submit" name="action" value="delete" class="btn-danger" onclick="return confirm('Are you sure you want to permanently delete the selected files?');">Delete Selected</button>
                    <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                  </div>
                </form>
              </div>
            </body>
            </html>
            """

            return render_template_string(html_page)

        @app.route('/latest_timelapse_image')
        @require_token
        # --- The last timelapse image ---
        def serve_latest_timelapse_image():
            # This logic is safe because it only ever serves from one specific directory
            latest_image_path = self.get_latest_file(TIMELAPSE_DIR)
            if latest_image_path and os.path.exists(latest_image_path):
                return send_from_directory(os.path.dirname(latest_image_path), os.path.basename(latest_image_path))
            # Return a 404 if no image is found
            return "No timelapse image available.", 404

    # ------ Event Statistics
        @app.route('/api/event_stats')
        @require_token
        def event_stats():
            """API endpoint to read and process events.csv for charting."""
            stats = {
                "by_hour": {str(h): 0 for h in range(24)},
                "by_night": {}
            }
            if not os.path.exists(self.event_log_out_file):
                return jsonify(stats) # Return empty stats if file doesn't exist

            try:
                with open(self.event_log_out_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            # Parse the UTC timestamp
                            ts_str = row['timestamp_utc'].replace('Z', '+00:00')
                            dt_utc = datetime.fromisoformat(ts_str)
                            
                            # Aggregate by hour
                            hour_key = str(dt_utc.hour)
                            stats["by_hour"][hour_key] += 1
                            
                            # Aggregate by "night" (noon to noon)
                            # If it's before noon, it belongs to the previous day's "night".
                            if dt_utc.hour < 12:
                                night_date = (dt_utc - timedelta(days=1)).strftime('%Y-%m-%d')
                            else:
                                night_date = dt_utc.strftime('%Y-%m-%d')
                            
                            stats["by_night"][night_date] = stats["by_night"].get(night_date, 0) + 1
                            
                        except (ValueError, KeyError):
                            continue # Skip malformed rows
                
                return jsonify(stats)
            except Exception as e:
                logging.error(f"Failed to process event stats: {e}")
                return jsonify({"error": "Failed to process event log"}), 500

        @app.route('/stats')
        @require_token
        def stats_page():
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else ""
            html_page = f"""
            <!DOCTYPE html><html lang="en">
            <head>
              <meta charset="UTF-8"><title>Event Statistics - Meteora</title>
              <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
              <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }}
                header {{ background: #2c2c2c; padding: 1em 2em; border-bottom: 1px solid #444; }}
                h1, h2 {{ margin: 0; color: #61afef; }}
                .main {{ padding: 2em; max-width: 1200px; margin: auto; }}
                .chart-container {{ background: #2b2b2b; border-radius: 12px; padding: 1.5em; box-shadow: 0 2px 6px rgba(0,0,0,0.5); margin-top: 2em; }}
                a {{ color: #c678dd; }}
              </style>
            </head>
            <body>
              <header><h1>Event Statistics</h1></header>
              <div class="main">
                <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
                <div class="chart-container">
                  <h2>Meteors Detected per Hour (UTC)</h2>
                  <canvas id="hourlyChart"></canvas>
                </div>
                <div class="chart-container">
                  <h2>Meteors Detected per Night</h2>
                  <canvas id="nightlyChart"></canvas>
                </div>
                <p><a href="/{token_param}">← Back to Main Dashboard</a></p>
              </div>
              <script>
                // Use a self-executing async function to fetch data and build charts
                (async () => {{
                    try {{
                        const response = await fetch('/api/event_stats{token_param}');
                        if (!response.ok) {{ throw new Error('Failed to fetch stats'); }}
                        const stats = await response.json();

                        // --- Build Hourly Chart ---
                        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
                        const hourlyLabels = Array.from({{length: 24}}, (_, i) => i.toString().padStart(2, '0') + ':00');
                        const hourlyData = hourlyLabels.map((_, i) => stats.by_hour[i.toString()] || 0);

                        new Chart(hourlyCtx, {{
                            type: 'bar',
                            data: {{
                                labels: hourlyLabels,
                                datasets: [{{
                                    label: 'Meteors per Hour',
                                    data: hourlyData,
                                    backgroundColor: 'rgba(97, 175, 239, 0.6)',
                                    borderColor: 'rgba(97, 175, 239, 1)',
                                    borderWidth: 1
                                }}]
                            }},
                            options: {{
                                scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#e0e0e0' }} }}, x: {{ ticks: {{ color: '#e0e0e0' }} }} }},
                                plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
                            }}
                        }});

                        // --- Build Nightly Chart ---
                        const nightlyCtx = document.getElementById('nightlyChart').getContext('2d');
                        const sortedNights = Object.keys(stats.by_night).sort();
                        const nightlyLabels = sortedNights;
                        const nightlyData = sortedNights.map(night => stats.by_night[night]);

                        if (nightlyLabels.length > 0) {{
                            new Chart(nightlyCtx, {{
                                type: 'bar',
                                data: {{
                                    labels: nightlyLabels,
                                    datasets: [{{
                                        label: 'Meteors per Night',
                                        data: nightlyData,
                                        backgroundColor: 'rgba(152, 195, 121, 0.6)',
                                        borderColor: 'rgba(152, 195, 121, 1)',
                                        borderWidth: 1
                                    }}]
                                }},
                                options: {{
                                    scales: {{ y: {{ beginAtZero: true, ticks: {{ color: '#e0e0e0' }} }}, x: {{ ticks: {{ color: '#e0e0e0' }} }} }},
                                    plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }}
                                }}
                            }});
                        }} else {{
                            nightlyCtx.font = '16px Segoe UI';
                            nightlyCtx.fillStyle = '#999';
                            nightlyCtx.textAlign = 'center';
                            nightlyCtx.fillText('No nightly data available yet.', nightlyCtx.canvas.width / 2, 50);
                        }}

                    }} catch (error) {{
                        console.error('Error loading chart data:', error);
                    }}
                }})();
              </script>
            </body></html>
            """
            return render_template_string(html_page)

        @app.route('/events/<path:filename>')
        @require_token
        def serve_event_file(filename):
            """Serves the captured event images and videos."""
            return send_from_directory(self.events_out_dir, filename)

        # --- Main Dashboard Page ---
        @app.route('/')
        @require_token
        def dashboard():
#            import html
            token_param = f"?token={DASH_TOKEN}" if DASH_TOKEN else "" 																				  
            # --- Gather Live Metrics ---
            cpu_temp_obj = psutil.sensors_temperatures().get("cpu_thermal", [None])[0]
            cpu_temp_raw = cpu_temp_obj.current if cpu_temp_obj else 0.0
            cpu_temp = f"{cpu_temp_raw:.1f}°C" if cpu_temp_obj else "N/A"
            mem = psutil.virtual_memory()
            mem_used = f"{mem.used / (1024**3):.2f} GB"
            disk = psutil.disk_usage('/')
            disk_used = f"{disk.used / (1024**3):.2f} GB"
            load1, _, _ = os.getloadavg()
            uptime_sec = time.time() - psutil.boot_time()
            uptime_days = int(uptime_sec // 86400)
            uptime_hours = int((uptime_sec % 86400) // 3600)
            uptime_str = f"{uptime_days}d {uptime_hours}h"             
            now = datetime.now()
            system_date = now.strftime('%Y-%m-%d')
            system_time = now.strftime('%H:%M:%S')

            # Calculate the estimated real timelapse interval for the info box
            try:
                # Use the CURRENT config, which may have been reloaded
                current_cfg = self.cfg
                capture_fps = current_cfg["capture"]["fps"]
                stack_n = current_cfg["timelapse"]["stack_N"]
                interval_sec = current_cfg["timelapse"]["interval_sec"]
                
                # Phase 1: Frame Collection Time
                collection_time = stack_n / capture_fps
                # Phase 2: Processing Time (estimate)
                processing_time_estimate = 5 # A reasonable estimate in seconds
                
                real_interval_estimate = collection_time + processing_time_estimate + interval_sec
                
                interval_note = (f"Note: Real interval is ~{int(real_interval_estimate)}s "
                                 f"({int(collection_time)}s collection + "
                                 f"~{processing_time_estimate}s processing + "
                                 f"{interval_sec}s wait)")
            except (KeyError, ZeroDivisionError):
                interval_note = "Note: Could not calculate real interval due to config values."

            with self.status_lock:
                status = { 
                    "version": self.cfg["general"]["version"], 
                    "capture_active": self.producer_thread_active(), 
                    "daylight_mode": self.daylight_mode_active.is_set(),
                    "weather_hold": self.weather_hold_active.is_set(),
                    "power_status": self.power_status, 
                    "event_count": self.event_counter, 
                    "last_illuminance": self.last_illuminance, 
                    "last_sky_stddev": self.last_sky_conditions.get("stddev", "N/A"), 
                    "last_sky_stars": self.last_sky_conditions.get("stars", "N/A"), 
                    "last_heartbeat": self.last_heartbeat_status, 
                    "live_threads": {t.name for t in (self.control_threads + self.worker_threads) if t.is_alive()},
                    "calib_error": self.last_calibration_error,
                    "last_calib_image": self.last_calibration_image_path,
                    "last_event_files": self.last_event_files.copy(),
                    "is_calibrating": self.is_calibrating.is_set(),
                    "emergency_stop": self.emergency_stop_active.is_set(),
                    "maintenance_mode": self.in_maintenance_mode.is_set(),
                    "is_in_schedule": self.within_schedule(),
                }
#                calib_error = self.last_calibration_error
            with self.maintenance_timeout_lock:
                status["maintenance_timeout_until"] = self.maintenance_timeout_until

            system_status_str = "Active Schedule" if status["is_in_schedule"] else "Idle Schedule"
            system_status_class = "ok" if status["is_in_schedule"] else "warn"

            try:
                current_cfg = self.cfg
                collection_time = current_cfg["timelapse"]["stack_N"] / current_cfg["capture"]["fps"]
                interval_note = (f"Note: Real interval is ~{int(collection_time + 5 + current_cfg['timelapse']['interval_sec'])}s")
            except (KeyError, ZeroDivisionError):
                interval_note = "Note: Could not calculate real interval."

            # 1. Get thresholds from the configuration
            # lux_threshold = self.cfg.get("daylight", {}).get("lux_threshold", 20.0)
            # min_stars_threshold = self.cfg.get("daylight", {}).get("min_stars", 20)

            # 2. Determine the CSS class for illuminance
            illuminance_class = "ok" # Default to green
            try:
                # Convert last_illuminance to a float for comparison
                last_lux_val = float(status['last_illuminance'])
                if last_lux_val > self.cfg.get("daylight", {}).get("lux_threshold", 20.0):
                    illuminance_class = "err" # It's daytime, show red
            except (ValueError, TypeError):
                illuminance_class = "warn" # Not a valid number, show yellow

            # 3. Determine the CSS class for star count
            stars_class = "ok" # Default to green
            try:
                # Convert last_sky_stars to an int for comparison
                last_stars_val = int(status['last_sky_stars'])
                if last_stars_val < self.cfg.get("daylight", {}).get("min_stars", 20):
                    stars_class = "err" # Not enough stars, show red
            except (ValueError, TypeError):
                stars_class = "warn" # Not a valid number, show yellow

            threshold = self.cfg["janitor"].get("threshold", 90.0)
            # Get the configured path and use it for the dashboard display.
            monitor_path = self.cfg["janitor"].get("monitor_path", "/")
            disk = psutil.disk_usage(monitor_path)
            disk_used = f"{disk.used / (1024**3):.2f} GB"

            sftp_status_html = ""
            if self.sftp_uploader:
                sftp_class, sftp_message = self.sftp_uploader.get_status()
                sftp_status_html = f'<tr><td>SFTP Status</td><td class="{sftp_class}">{sftp_message}</td></tr>'

            emergency_message_html = ""
            buttons_html = ""
            timeout_message_html = ""            
            config_and_calibration_html = ""
  
            # 4. Read the last 30 lines of the main pipeline log file.
            log_lines = self.read_last_lines(os.path.join(self.general_log_dir, "pipeline.log"), num_lines=30)
            
            # 5. Escape HTML characters for safe rendering and join into a single string.
#            import html
            log_html = "<br>".join(html.escape(line) for line in log_lines)

            # 6. Create the HTML block for the log viewer.
            log_viewer_html = f"""
            <div class="card">
                <h2>Live Log (Last {len(log_lines)} Lines)</h2>
                <div class="log-box">
                    <pre><code>{log_html}</code></pre>
                </div>
            </div>
            """

            # --- Gather Health Statistics ---
            health_stats = self.get_health_statistics(self.health_log_out_file)
            health_stats_html_rows = ""
            if not health_stats:
                health_stats_html_rows = "<tr><td colspan='4'>No health events recorded yet.</td></tr>"
            else:
                for event_data in health_stats:
                    # Use .get() for all keys to prevent KeyErrors if a row is malformed
                    level_class = event_data.get('level', 'INFO').lower()
                    event_type = event_data.get('event_type', 'N/A')
                    count = event_data.get('count', 0)
                    last_message = event_data.get('last_message', '')
                    health_stats_html_rows += f"""
                    <tr>
                        <td class="event-{level_class}">{event_type}</td>
                        <td>{count}</td>
                        <td>{html.escape(last_message)}</td>
                    </tr>
                    """

            health_card_html = f"""
            <div class="card" style="grid-column: 1 / -1;">
                <h2>Health & Event Statistics</h2>
                <div class="log-box" style="height: 200px;">
                    <table style="font-size: 0.8em;">
                        <thead>
                            <tr>
                                <th>Event Type</th>
                                <th>Count</th>
                                <th>Last Message</th>
                            </tr>
                        </thead>
                        <tbody>{health_stats_html_rows}</tbody>
                    </table>
                </div>
            </div>
            """  

            if status["emergency_stop"]:
                emergency_message_html = f"""
                <div style="border: 3px solid #e06c75; background: #3c3c3c; padding: 1.5em; margin: 1em 2em; border-radius: 8px; text-align: center;">
                    <h2 style="color: #e06c75; margin: 0 0 0.5em 0;">EMERGENCY STOP ACTIVATED</h2>
                    <p style="margin: 0; font-size: 1.1em;">
                        A critical error happen, the system is in an unknown state.
                        <br>All new data acquisition has been **HALTED** to prevent system failure.
                    </p>
                    <p style="margin-top: 1em;">
                        <strong>Action Required:</strong> Use the <a href="/files{token_param}">File Manager</a> to manually delete files.
                        <br>A system restart will be required after space has been cleared.
                    </p>
                </div>
                """
                # When in emergency, we explicitly DISABLE all control buttons.
            else:
                if status["maintenance_mode"]:
                    buttons_html = f'<a href="/api/resume{token_param}"><button style="background: #98c379;">Resume Normal Operation</button></a>'

                    # Check if a timeout is currently active
                    with self.maintenance_timeout_lock:
                        if status["maintenance_timeout_until"] > 0:
                            remaining_sec = self.maintenance_timeout_until - time.time()
                            if remaining_sec > 0:
                                remaining_min = int(remaining_sec / 60)
                                timeout_message_html = f"""
                                <div style="border: 2px solid #61afef; padding: 1em; margin-bottom: 1em; border-radius: 8px;">
                                    <p style="color: #61afef; font-weight: bold; margin: 0;">
                                        System is in Maintenance Mode. It will automatically resume in approximately {remaining_min} minute(s).
                                    </p>
                                </div>
                                """
                    # 2. Always generate the settings form HTML.
                    settings_html_rows = ""
                    for section, params in EDITABLE_PARAMS.items():
                        settings_html_rows += f'<tr><th colspan="2">{section.replace("_", " ").title()}</th></tr>'
                        for param in params:
                            value = self.cfg.get(section, {}).get(param, 'N/A')
                            note = f'<small style="color:#e5c07b;">{interval_note}</small>' if param == "interval_sec" else "" # Simplified for brevity
                            settings_html_rows += f'<tr><td>{param}</td><td><input type="text" name="{section}_{param}" value="{value}" size="10"> {note}</td></tr>'

                    error_box_html = ""
                    if status["calib_error"]:
                        error_box_html = f"""
                        <div class="calibration-box" style="border: 2px solid #e06c75; padding: 1em; background: #3c3c3c;">
                            <h4 style="color: #e06c75; margin-top: 0;">Test Shot Failed</h4>
                            <p style="font-family: monospace; color: #e5c07b;">{calib_error}</p>
                            <p style="font-size: smaller;">Check `pipeline.log` for more details. You can adjust settings and try again.</p>
                        </div>
                        """
                    if status["is_calibrating"]:
                        config_and_calibration_html = """
                        <div class="config-container">
                            <h2>Maintenance & Calibration</h2>
                            <p style="color: #61afef; font-weight: bold;">TEST SHOT IN PROGRESS... The page will refresh.</p>
                        </div>
                        """
                    else:
                        image_box = ""
                        if self.last_calibration_image_path:
                            image_url = f"/calibration_image{token_param}&t={time.time()}"
                            image_box = f'<h4>Last Test Shot:</h4><a href="{image_url}" target="_blank"><img src="{image_url}" alt="Last Test Shot" width="400"></a>'

                        config_and_calibration_html = f"""
                        <div class="config-container">
                            <form action="/api/save_and_test{token_param}" method="post" id="config-form">
                                <h2>Maintenance & Calibration</h2>
                                <table>{settings_html_rows}</table>
                                <button type="submit">Save Config & Take Test Shot</button>
                            </form>
                            <div class="calibration-box">
                                {error_box_html} 
                                {image_box}
                            </div>
                        </div>
                        """
                else:
                    buttons_html = f'<a href="/api/pause{token_param}"><button style="background: #e5c07b;">Pause Capture & Enter Maintenance Mode</button></a>'
                    timeout_message_html = "" # No timeout message
                    config_and_calibration_html = "" # No config editor

            # --- "LAST EVENT" CARD LOGIC ---
            last_event_html = ""
            last_event = status["last_event_files"]
            event_image_path = last_event.get("image")
            event_video_path = last_event.get("video")

            if event_image_path and os.path.exists(event_image_path):
                # We need a relative path for the URL, not the full system path
                # This assumes 'output' is the static folder for Flask
                relative_image_url = os.path.join('events', os.path.basename(event_image_path))
                video_download_html = ""
                if event_video_path and os.path.exists(event_video_path):
                    relative_video_url = os.path.join('events', os.path.basename(event_video_path))
                    video_download_html = f'<p><a href="/{relative_video_url}{token_param}" download><b>Download Event Video (.mp4)</b></a></p>'

                last_event_html = f"""
                <div class="card" style="grid-column: 1 / -1;">
                    <h2>Last Captured Event</h2>
                    <div style="text-align: center;">
                        <a href="/{relative_image_url}{token_param}" target="_blank">
                            <img src="/{relative_image_url}{token_param}&t={time.time()}" alt="Last Event Image" style="max-width: 50%; height: auto; border: 1px solid #555; border-radius: 8px;">
                        </a>
                        <p style="font-size: smaller; color: #999;">{os.path.basename(event_image_path)}</p>
                        {video_download_html}
                    </div>
                </div>
                """
            else:
                last_event_html = f"""
                <div class="card" style="grid-column: 1 / -1;">
                    <h2>Last Captured Event</h2>
                    <p style="text-align: center; color: #999;">Waiting for first event to be detected and processed...</p>
                </div>
                """
            # --- END: "LAST EVENT" CARD LOGIC ---

            # --- START: NEW PIPELINE MODE LOGIC ---
            pipeline_mode_str = ""
            pipeline_mode_class = ""

            if not status['capture_active']:
                pipeline_mode_str = "Idle (Stopped)"
                pipeline_mode_class = "warn" # Yellow for idle/paused
            elif status['daylight_mode'] or status['weather_hold']:
                reason = "Daylight" if status['daylight_mode'] else "Weather Hold"
                pipeline_mode_str = f"Timelapse Only ({reason})"
                pipeline_mode_class = "warn" # Yellow for partial operation
            else:
                pipeline_mode_str = "Timelapse + Events"
                pipeline_mode_class = "ok" # Green for full operation
            # --- END: NEW PIPELINE MODE LOGIC ---

            # --- GENERATE ALL HTML COMPONENTS ---
            pipeline_status_rows = f"""
                <tr><td>System Status</td><td class="{system_status_class}">{system_status_str}</td></tr>
                <tr><td>Capture Active</td><td class="{'ok' if status['capture_active'] else 'warn'}">{status['capture_active']}</td></tr>
                <tr><td>Pipeline Mode</td><td class="{pipeline_mode_class}">{pipeline_mode_str}</td></tr>
                <tr><td>Power Status</td><td class="{'ok' if status['power_status'] == 'OK' else 'warn'}">{status['power_status']}</td></tr>
                <tr><td>Events Captured</td><td>{status['event_count']}</td></tr>
                <tr><td>Last Illuminance (Lux)</td><td class="{illuminance_class}">{status['last_illuminance']}</td></tr>
                <tr><td>Last Sky StdDev / Stars</td><td class="{stars_class}">{status['last_sky_stddev']} / {status['last_sky_stars']}</td></tr>
                <tr><td>Last Heartbeat</td><td>{status['last_heartbeat']}</td></tr>
                {sftp_status_html}
            """
            
            # Component: Threads Table
            all_thread_names = sorted(list(set([t.name for t in self.control_threads] + [t.name for t in self.worker_threads])))
            threads_html_rows = ""
            for name in all_thread_names:
                 status_html = '<td class="ok">OK</td>' if name in status["live_threads"] else '<td class="err">KO</td>'
                 threads_html_rows += f"<tr><td>{name}</td>{status_html}</tr>"
            
            # Component: System Vitals Table
            system_vitals_rows = f"""
                <tr><td>System Date</td><td>{system_date}</td></tr>
                <tr><td>System Time (Local)</td><td>{system_time}</td></tr>            
                <tr><td>CPU Temperature</td><td class="{'warn' if cpu_temp_raw > 75 else 'ok'}">{cpu_temp}</td></tr>
                <tr><td>Memory Usage</td><td class="{'warn' if mem.percent > 80 else 'ok'}">{mem_used} ({mem.percent}%)</td></tr>
                <tr><td>Disk Usage ({monitor_path})</td><td class="{'err' if disk.percent > 90 else 'warn' if disk.percent > threshold else 'ok'}">{disk_used} ({disk.percent}%)</td></tr>
                <tr><td>Load Average (1m)</td><td>{load1:.2f}</td></tr>
                <tr><td>Uptime</td><td>{uptime_str}</td></tr>
            """            
           
            picture_box_html = ""
            # is_timelapse_enabled = self.cfg["timelapse"].get("stack_N", 0) > 0
            
            # Show the picture box only if timelapse is enabled AND the pipeline is active
            if self.cfg["timelapse"].get("stack_N", 0) > 0 and status["capture_active"] and not status["maintenance_mode"]:
                # The 't={time.time()}' part is a cache-buster to ensure the browser always fetches the latest image
                image_url = f"/latest_timelapse_image{token_param}&t={time.time()}"
                picture_box_html = f"""
                <div class="picture-box-container">
                    <h2>Latest Timelapse</h2>
                    <a href="{image_url}" target="_blank">
                        <img src="{image_url}" alt="Latest Timelapse Image" style="max-width: 50%; height: auto; border: 1px solid #555;">
                    </a>
                    <p style="font-size: smaller; color: #999;">This image updates automatically. Click to view full size.</p>
                </div>
                """

            # --- FOOTER TABLE ---
            footer_table_html = f"""
            
            <div class="section" style="padding: 0 2em;">
            <div class="card">
            <h2>Footer</h2>
            <div class="log-box">
                <table class="footer-nav">
                    <thead>
                        <tr>
                            <th>Logs</th>
                            <th>File Management</th>
                            <th>Configuration</th>
                            <th>Data Analysis</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><a href="/logs/pipeline.log{token_param}">pipeline.log</a></td>
                            <td><a href="/files{token_param}">Browse Event & Timelapse Files</a></td>
                            <td><a href="/settings{token_param}">Edit Full Configuration</a></td>
                            <td><a href="/stats{token_param}">View Event Statistics</a></td>
                        </tr>
                        <tr>
                            <td><a href="/logs/system_monitor.csv{token_param}">system_monitor.csv</a></td>
                            <td></td>
                            <td><a href="/system_time{token_param}">Set System Time</a></td>
                            <td></td>
                        </tr>
                        <tr>
                            <td><a href="/logs/events.csv{token_param}">events.csv</a></td>
                            <td></td>
                            <td></td>
                            <td></td>
                        </tr>
                        <tr>
                            <td><a href="/logs/health_stats.csv{token_param}">health_stats.csv</a></td>
                            <td></td>
                            <td></td>
                            <td></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            </div>
            </div>
            """

            refresh_url = f"/{token_param}"
            html_page = f"""
            <!DOCTYPE html><html lang="en">
            <head>
              <meta charset="UTF-8">
              <title>Meteora Pipeline Dashboard</title>
              <meta http-equiv="refresh" content="10; url={refresh_url}">
              <style>
                body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1e1e1e; color: #e0e0e0; margin: 0; padding: 0; }}
                header {{ background: #2c2c2c; padding: 1em 2em; border-bottom: 1px solid #444; }}
                h1 {{ margin: 0; color: #61afef; }}
                .main {{ padding: 2em; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .card {{ background: #2b2b2b; border-radius: 12px; padding: 1em; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }}
                .card h2 {{ margin-top: 0; font-size: 1.2em; color: #98c379; }}
                table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
                th, td {{ padding: 6px 8px; border-bottom: 1px solid #444; }}
                th {{ text-align: left; color: #61afef; }}
                .ok {{ color: #98c379; font-weight: bold; }}
                .warn {{ color: #e5c07b; font-weight: bold; }}
                .err {{ color: #e06c75; font-weight: bold; }}
                button {{ background: #61afef; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 0.9em; margin: 5px 0; transition: 0.2s; }}
                button:hover {{ background: #4fa3d8; }}
                .btn-danger {{ background: #e06c75; }}
                .btn-danger:hover {{ background: #c65c66; }}
                img {{ max-width: 100%; border-radius: 6px; margin-top: 10px; }}
                .section {{ margin-top: 2em; }}
                .log-box {{ background-color: #1a1a1a; border: 1px solid #444; border-radius: 6px; padding: 10px; height: 300px; overflow-y: scroll; font-size: 0.8em; }}
                .log-box pre, .log-box code {{ margin: 0; padding: 0; white-space: pre-wrap; word-wrap: break-word; }}    
                .event-warning {{ color: #e5c07b; font-weight: bold; }}
                .event-error {{ color: #e06c75; font-weight: bold; }}
                .event-critical {{ color: #e06c75; font-weight: bold; text-transform: uppercase; }}  
                .footer-nav th {{ color: #98c379; font-size: 1.3em; border-bottom: 2px solid #444; padding-bottom: 10px; }}
                .footer-nav td {{ padding-top: 10px; padding-bottom: 10px; border-bottom: none; }}
                .footer-nav a {{ font-size: 1.2em; }}
              </style>
            </head>
            <body>
              <header>
                <h1>Meteora Pipeline Status (v{status['version']})</h1>
              </header>
              {emergency_message_html}
              <div class="main">
                <div class="card">
                  <h2>Pipeline Status</h2>
                  <table>{pipeline_status_rows}</table>
                </div>
                <div class="card">
                  <h2>Thread Status</h2>
                  <table>{threads_html_rows}</table>
                </div>
                <div class="card">
                  <h2>System Vitals</h2>
                  <table>{system_vitals_rows}</table>
                </div>
              </div>
              <div class="section" style="padding: 0 2em;">
                {last_event_html}
              </div>              
              <div class="section" style="padding: 0 2em;">
                {health_card_html}
              </div>
              <div class="section" style="padding: 0 2em;">
                {log_viewer_html}
              </div>              
              <div class="section" style="padding: 0 2em;">
                {buttons_html}
                {timeout_message_html}
                {picture_box_html}
                {config_and_calibration_html}
              </div>
              <div class="section" style="padding: 0 2em;>
                {footer_table_html}
              </div>
            </body>
            </html>
            """
            
            # -------- Save the generated HTML to a file for troubleshooting
            # try:
                # debug_path = os.path.join(self.general_log_dir, "dashboard_debug.html")
                # with open(debug_path, "w") as f:
                    # f.write(html_file)
            # except Exception as e:
                # # Log this error but don't crash the dashboard
                # logging.warning("Could not save dashboard debug HTML file: %s", e)            
            # -------- 
            
            return render_template_string(html_page)        

        # --- Log Download Endpoint (unchanged) ---
        @app.route('/logs/<path:filename>')
        @require_token
        def download_log(filename):
            log_dir = os.path.abspath(self.general_log_dir)
            allowed_files = ["pipeline.log", "system_monitor.csv", "events.csv", "health_stats.csv"]
            if filename in allowed_files:
                return send_from_directory(log_dir, filename, as_attachment=True)
            else:
                return "File not found.", 404

        host = dash_cfg.get("host", "0.0.0.0")
        port = dash_cfg.get("port", 5000)
        logging.info("Starting status dashboard. Access at http://%s:%d or http://<IP_ADDRESS>:%d", socket.gethostname(), port, port)
        app.logger.disabled = True
        logging.getLogger('werkzeug').disabled = True
        
        try:
            app.run(host=host, port=port)
        except OSError as e:
            logging.error("Could not start dashboard on port %d: %s. Is another process using it?", port, e)

    # ----------------- Watchdog -----------------
    def watchdog_loop(self, startup_delay_sec=10, check_interval_sec=5):
        """
        The master supervisor thread. It performs two critical functions:
        1. Periodically checks if all other critical threads are alive.
        2. Periodically pings the systemd watchdog to signal script health.

        If any thread has crashed, this loop will exit with an error code,
        triggering a service restart by systemd. If this loop itself freezes,
        systemd will stop receiving pings and restart the service.
        """
        try:
            import sdnotify
            n = sdnotify.SystemdNotifier()
            systemd_integration = True
        except ImportError:
            logging.warning("sdnotify library not found. Running without systemd integration.")
            systemd_integration = False

        # Give all other threads a moment to start up before signaling readiness.
        logging.info("Watchdog thread started. Waiting %d seconds before signaling READY.", startup_delay_sec)
        for _ in range(startup_delay_sec):
            if not self.running.is_set(): return
            time.sleep(1)

        # Signal systemd that the service is fully initialized and ready.
        if systemd_integration:
            n.notify("READY=1")
            logging.info("Watchdog Supervisor signaled READY=1 to systemd.")

        while self.running.is_set():
            if self.watchdog_pause.is_set():
                logging.warning("Watchdog checks are temporarily suspended for configuration reload.")
                # We still need to ping the systemd watchdog to prevent it from timing out
                if systemd_integration:
                    n.notify("WATCHDOG=1")
                # Wait for the interval and then restart the loop, skipping the thread checks
                time.sleep(check_interval_sec)
                continue
            
            # 1. Ping systemd to let it know the script is responsive.
            if systemd_integration:
                n.notify("WATCHDOG=1")

            # 2. Check the health of all other threads.
            all_monitored_threads = self.worker_threads + self.control_threads
            for thread in all_monitored_threads:
                if not thread.is_alive():
                    # --- EMERGENCY: A THREAD HAS CRASHED ---
                    msg = f"WATCHDOG: DETECTED CRASHED THREAD: {thread.name}"

                    self.log_health_event("CRITICAL", "THREAD_CRASH", msg)
                    # Stop the data producer.
                    self.stop_producer_thread()
                    # Set emergency mode
                    self.emergency_stop_active.set()
                    
#                    logging.critical("="*60)
#                    logging.critical(f"!!! {msg} !!!")
#                    logging.critical("A critical component has failed. Initiating a graceful shutdown.")
#                    logging.critical("systemd will restart the service automatically.")
#                    logging.critical("="*60)
                    
                    self.perform_system_action("exit", reason=msg)
            # If all is well, wait for the next check interval.
            for _ in range(check_interval_sec):
                if not self.running.is_set(): break
                time.sleep(1)

    # -----------------
    def maintenance_watchdog_loop(self):
        """
        A dedicated thread that monitors the maintenance mode timeout.
        If the timeout is reached, it automatically resumes the pipeline.
        """
        logging.info("Maintenance Watchdog started.")
        while self.running.is_set():
            timeout_target = 0
            with self.maintenance_timeout_lock:
                if self.maintenance_timeout_until > 0:
                    timeout_target = self.maintenance_timeout_until

            if timeout_target > 0:
                # A timeout is active, check if it has expired
                if time.time() >= timeout_target:
                    logging.warning("Maintenance mode timeout reached. Automatically resuming pipeline...")
                    self.in_maintenance_mode.clear()
                    # Reset the timer
                    with self.maintenance_timeout_lock:
                        self.maintenance_timeout_until = 0
                    
                    # Only resume if we are within the active schedule
                    if self.within_schedule():
                        self.start_producer_thread()
                    else:
                        logging.info("Maintenance timeout expired, but system is outside of active schedule. Staying idle.")
                
            # Sleep for a short interval before checking again
            for _ in range(5):
                if not self.running.is_set(): break
                time.sleep(1)

    # ----------------- Calibration Worker -----------------
    def calibration_worker_loop(self):
        """
        A dedicated worker that waits for frames on the calibration_q,
        collects N of them, stacks them, and saves the result.
        """
        logging.info("Calibration worker started, waiting for triggered frames.")
        calib_cfg = self.cfg.get("calibration", {})

        while self.running.is_set():
            try:
                # 1. Wait for the very first frame to arrive.
                first_frame_item = self.calibration_q.get(timeout=10)
                
                stack_n = self.cfg["timelapse"].get("stack_N", 10)
                logging.warning("Calibration worker received first frame. Capturing %d total for test shot...", stack_n)
                
                with self.status_lock:
                    self.last_calibration_error = None

                frames_for_stack = [first_frame_item]
                capture_ok = True

                # 2. Collect the REMAINING frames.
                for i in range(stack_n - 1):
                    if not self.running.is_set():
                        capture_ok = False
                        break
                    try:
                        # Use a timeout in case the frame stream stops for any reason
                        item = self.calibration_q.get(timeout=5.0)
                        frames_for_stack.append(item)
                        logging.info(f"Captured live test frame {i+2}/{stack_n}...")
                    except queue.Empty:
                        error_msg = f"Timed out waiting for frame {i+2}/{stack_n}. Aborting test shot."
                        logging.error(f"Calibration worker: {error_msg}")
                        with self.status_lock:
                            self.last_calibration_error = error_msg
                        capture_ok = False
                        break
                
                # 3. Process the collected frames if everything went well.
                if capture_ok:
                    logging.info("Stacking %d frames for calibration image...", len(frames_for_stack))
                    stacked_image = self.stack_frames(
                        [f for _, f in frames_for_stack],
                        align=self.cfg["timelapse"]["stack_align"],
                        method="mean",
                        min_features_for_aligment=self.cfg["timelapse"]["min_features_for_aligment"]
                    )
                    if stacked_image is not None:
                        tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"calibration_shot_{tstamp}.png"
                        full_path = os.path.join(self.calibration_out_dir, filename)
                        cv2.imwrite(full_path, stacked_image)
                        
                        with self.status_lock:
                            self.last_calibration_image_path = full_path
                        logging.info("Calibration image saved to: %s", full_path)
                    else:
                        raise RuntimeError("Frame stacking returned None.")
                
            except queue.Empty:
                # This is normal, it just means no trigger was received in the last 10s.
                continue
            except Exception as e:
                error_msg = f"An unexpected error occurred during test shot: {e}"
                logging.exception(error_msg)
                with self.status_lock:
                    self.last_calibration_error = error_msg
            finally:
                # 4. CRITICAL: Always reset the state for the next run.
                # Clear the trigger so the detection_loop stops sending frames.
                # Drain any extra frames that might have queued up during stacking.
                if self.is_calibrating.is_set():
                    logging.warning("Calibration worker is terminating the temporary capture session.")
                    self.stop_producer_thread()
                    self.is_calibrating.clear() # Take system out of calibration mode

                while not self.calibration_q.empty():
                    try:
                        self.calibration_q.get_nowait()
                    except queue.Empty:
                        break

    # ----------------- SFTP Dispatcher   
    def sftp_dispatcher_loop(self):
        """
        A smart dispatcher that acts as a persistent, disk-backed queue for the SFTP uploader.
        Implements the logic to use a disk file as a backlog when the memory queue is full.
        """
        if not self.sftp_uploader:
            logging.info("SFTP Dispatcher: Uploader is disabled. Thread will exit.")
            return

        logging.info(f"SFTP Dispatcher started. Using backlog file: {self.backlog_file_path}")

        while self.running.is_set():
            try:
                # 1. Check if we can flush from the disk backlog first.
                upload_q = self.sftp_uploader.upload_q
                
                if upload_q.qsize() < upload_q.maxsize and os.path.exists(self.backlog_file_path):
                    logging.info("SFTP Dispatcher: Upload queue has space. Flushing from disk backlog.")
                    temp_backlog = []
                    with open(self.backlog_file_path, 'r') as f:
                        temp_backlog = [line.strip() for line in f if line.strip()]
                    
                    files_flushed = 0
                    while temp_backlog and not upload_q.full():
                        file_path = temp_backlog.pop(0)
                        self.sftp_uploader.upload_q.put(file_path)
                        files_flushed += 1
                    
                    # Rewrite the backlog file with the remaining items
                    with open(self.backlog_file_path, 'w') as f:
                        for item in temp_backlog:
                            f.write(item + '\n')
                    
                    if not temp_backlog:
                        os.remove(self.backlog_file_path) # Clean up the file if it's empty
                    
                    logging.info(f"SFTP Dispatcher: Flushed {files_flushed} items from disk to memory queue.")

                # 2. Process new items from the main dispatch queue.
                try:
                    # Use a timeout to allow the loop to re-check the backlog file periodically
                    new_item = self.sftp_dispatch_q.get(timeout=5.0)
                    
                    # 3. Decide where to put the new item
                    if not self.sftp_uploader.upload_q.full():
                        # Rule 1: There's space, enqueue directly.
                        self.sftp_uploader.upload_q.put(new_item)
                    else:
                        # Rule 2: Queue is full, write to disk backlog.
                        logging.warning(f"SFTP Dispatcher: Upload queue is full. Writing {os.path.basename(new_item)} to disk backlog.")
                        with open(self.backlog_file_path, 'a') as f:
                            f.write(new_item + '\n')

                except queue.Empty:
                    # This is normal, just means no new files were created in the last 5 seconds.
                    # The loop will now restart and check the backlog file again.
                    continue

            except Exception as e:
                logging.exception(f"An error occurred in the SFTP Dispatcher loop: {e}")
                time.sleep(30) # Wait a bit before retrying on a major error

    # --- Sweep and Enqueue ---
    def sweep_and_enqueue_unuploaded(self):
        """Scans output directories on startup for files that were not yet uploaded."""
        logging.info("Performing startup sweep for un-uploaded files...")
        count = 0
        scan_dirs = [self.events_out_dir, self.timelapse_out_dir]
        for directory in scan_dirs:
            for entry in os.scandir(directory):
                if entry.is_file() and not entry.name.endswith('.json') and not entry.name.startswith('.'):
                    json_path = os.path.splitext(entry.path)[0] + ".json"
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                meta = json.load(f)
                            if not meta.get("uploaded_at"):
                                self.sftp_dispatch_q.put(entry.path)
                                self.sftp_dispatch_q.put(json_path)
                                count += 2
                        except (json.JSONDecodeError, IOError):
                            logging.warning(f"Could not read metadata for {entry.name}, skipping upload check.")
                            self.log_health_event("WARNING", "METADATA_READ_ERROR", "Could not read metadata, skipping upload check.")
        if count > 0:
            logging.info(f"Found and enqueued {count} previously un-uploaded files.")

    # ----------------- NTP Sync
    def ntp_sync_loop(self):
        """
        Periodically checks the system clock against an NTP server. If significant
        drift is detected, it triggers the OS's own time service to re-synchronize.
        """
        ntp_cfg = self.cfg.get("ntp", {})

        try:
            import ntplib
        except ImportError:
            logging.error("The standard 'ntplib' library could not be imported. NTP check is disabled.")
            return

        server = ntp_cfg.get("server", "pool.ntp.org")
        interval_sec = ntp_cfg.get("sync_interval_hours", 6) * 3600
        max_offset = ntp_cfg.get("max_offset_sec", 2.0)
        
        logging.info(f"NTP clock monitor started. Will check against '{server}' every {interval_sec / 3600} hours.")
        time.sleep(60) # Initial delay for network

        while self.running.is_set():
            try:
                
                server = ntp_cfg.get("server", "pool.ntp.org")
                interval_sec = ntp_cfg.get("sync_interval_hours", 6) * 3600
                max_offset = ntp_cfg.get("max_offset_sec", 2.0)                
                
                logging.info("Performing periodic NTP clock check...")
                client = ntplib.NTPClient()
                response = client.request(server, version=3, timeout=15)
                offset = response.offset
                logging.info(f"NTP check complete. System clock offset: {offset * 1000:.2f} ms.")

                if abs(offset) > max_offset:
                    logging.critical("="*60)
                    logging.critical("!!! CRITICAL CLOCK DRIFT DETECTED !!!")
                    logging.critical(f"System clock is off by {offset:.3f} seconds, which exceeds the threshold of {max_offset:.3f}s.")
                    logging.critical("Attempting to force a re-synchronization via the OS time service.")
                    logging.critical("="*60)
                    
                    try:
                        # This command tells systemd-timesyncd to re-enable and force a sync.
                        cmd = ["sudo", "/usr/bin/timedatectl", "set-ntp", "true"]
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        logging.info("Successfully triggered OS time synchronization service.")
                        # After triggering, wait a bit for the sync to happen before the next check.
                        time.sleep(60)
                    except subprocess.CalledProcessError as e:
                        logging.error("Failed to trigger OS time synchronization.")
                        self.log_health_event("ERROR", "NTP_FAIL", "Failed to trigger OS time synchronization.")
                        logging.error("Command failed: %s", " ".join(e.cmd))
                        logging.error("Stderr: %s", e.stderr)
                    except FileNotFoundError:
                        logging.error("Could not find '/usr/bin/timedatectl'. Cannot trigger time sync.")
                else:
                    logging.info("System clock is within acceptable tolerance.")

            except Exception as e:
                logging.warning(f"Could not complete NTP query against '{server}': {e}")
                self.log_health_event("WARNING", "NTP_ERROR", "Could not complete NTP query.")

            # Wait for the next interval
            logging.info(f"NTPThread is sleeping for {interval_sec / 3600:.1f} hours.")
            for _ in range(interval_sec):
                if not self.running.is_set(): break
                time.sleep(1)

    # ----------------- Queue Monitor Loop -----------------
    def queue_monitor_loop(self):
        """A dedicated thread to periodically report the size of ALL critical queues."""
        monitor_cfg = self.cfg.get("queue_monitor", {})
        
        interval = monitor_cfg.get("interval_sec", 30)
        logging.info(f"Queue monitor started. Will report on ALL queues every {interval} seconds.")

        while True:
            try:
                # Build a dictionary of all queues to monitor for this iteration
                queues_to_monitor = {
                    "Acq": self.acq_q,
                    "Timelapse": self.timelapse_q,
                    "EventWriter": self.event_q,
                    "EventStack": self.event_stack_q,
                    "EventLog": self.event_log_q,
                    "Calibration": self.calibration_q
                }
                
                # Safely add SFTP queues to the report ONLY if the module is enabled
                sftp_uploader_instance = self.sftp_uploader
                if sftp_uploader_instance:
                    queues_to_monitor["SFTP-Dispatch"] = self.sftp_dispatch_q
                    queues_to_monitor["SFTP-Upload"] = sftp_uploader_instance.upload_q

                # Build the status string
                status_parts = []
                is_congested = False
                for name, q in queues_to_monitor.items():
                    # Check if the queue object itself is valid before using it
                    if q is None: continue
                    
                    size = q.qsize()

                    # maxsize is 0 for unbounded queues (like event_q)
                    maxsize = q.maxsize if hasattr(q, 'maxsize') and q.maxsize > 0 else "∞"
                    
                    percent_full = 0
                    if isinstance(maxsize, int) and maxsize > 0:
                        percent_full = (size / maxsize) * 100
                        status_parts.append(f"{name}: {size}/{maxsize} ({percent_full:.0f}%)")
                        if percent_full > 95:
                            is_congested = True
                    else:
                        # For unbounded queues, just show the size
                        status_parts.append(f"{name}: {size}")

                # Choose the log level based on whether any queue is nearing capacity
                log_level = logging.WARNING if is_congested else logging.INFO
                logging.log(log_level, f"Queue Status | {' | '.join(status_parts)}")

            except Exception as e:
                logging.exception("An error occurred in the queue_monitor_loop: %s", e)

            for _ in range(interval):
                if not self.running.is_set():
                    break
                time.sleep(1)            
              
    # -----------------
    def perform_system_action(self, action="exit", reason="Unknown reason"):
        """
        Unified handler for system-level state changes.
        Actions:
        - "exit": Exits script with status 1 (triggers systemd service restart).
        - "reboot": Gracefully stops pipeline, then reboots operating system.
        - "shutdown": Gracefully stops pipeline, then halts operating system.
        """
        logging.critical("="*60)
        logging.critical(f"!!! SYSTEM ACTION TRIGGERED: {action.upper()} !!!")
        logging.critical(f"!!! Reason: {reason} !!!")
        logging.critical("="*60)

        if action == "exit":
            logging.info("Exiting script with status 1 for service manager restart.")
            sys.exit(1)

        # For reboot/shutdown, we MUST try to stop gracefully first to save data.
        try:
            self.stop()
        except Exception as e:
            logging.error(f"Error during graceful stop before {action}: {e}")

        # Execute OS-level commands
        if action == "reboot":
            logging.info("Executing OS reboot...")
            subprocess.run(["sudo", "reboot"], check=False)
        elif action == "shutdown":
            logging.info("Executing OS halt...")
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    # ----------------- Health Monitor Loop -----------------
    def health_monitor_loop(self):
        """
        A stateful worker that maintains a persistent counter of health events.
        It reads the stats file into memory, updates counts based on events from
        the health_q, and rewrites the file.
        """

        # --- In-memory state of the health statistics ---
        # Format: { "EVENT_TYPE": {"count": N, "level": "LEVEL", "last_message": "..."} }
        health_stats = {}

        # --- Read the existing stats file on startup ---
        try:
            if os.path.exists(self.health_log_out_file):
                with open(self.health_log_out_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert count back to an integer
                        health_stats[row['event_type']] = {
                            "count": int(row['count']),
                            "level": row['level'],
                            "last_message": row['last_message']
                        }
                logging.info(f"Health monitor loaded {len(health_stats)} existing event types.")
        except Exception as e:
            logging.error(f"Could not load existing health stats file: {e}")

        while self.running.is_set():
            try:
                # Wait for a new health event to arrive
                health_event = self.health_q.get(timeout=1.0)
                if health_event is None:
                    break # Sentinel for shutdown

                event_type = health_event['event_type']
                
                if event_type in health_stats:
                    # 1. If event type exists, increment the counter
                    health_stats[event_type]['count'] += 1
                    # Always update with the latest message and level
                    health_stats[event_type]['last_message'] = health_event['message']
                    health_stats[event_type]['level'] = health_event['level']
                else:
                    # 2. If it's a new event type, create a new entry
                    health_stats[event_type] = {
                        "count": 1,
                        "level": health_event['level'],
                        "last_message": health_event['message']
                    }
                
                try:
                    # Use a temporary file for an atomic write to prevent corruption
                    temp_path = self.health_log_out_file + ".tmp"
                    with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                        # Define the fieldnames in the order you want them
                        fieldnames = ['event_type', 'count', 'level', 'last_message']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        writer.writeheader()
                        # Sort by count so the file is always nicely ordered
                        sorted_events = sorted(health_stats.items(), key=lambda item: item[1]['count'], reverse=True)
                        for event_type, data in sorted_events:
                            writer.writerow({
                                'event_type': event_type,
                                'count': data['count'],
                                'level': data['level'],
                                'last_message': data['last_message']
                            })
                    # Atomically replace the old file with the new one
                    os.rename(temp_path, self.health_log_out_file)
                except Exception as e:
                    logging.error(f"Failed to write updated health stats file: {e}")
                               
            except queue.Empty:
                continue # This is normal, just loop again
            except Exception as e:
                logging.exception("Error in health_monitor_loop: %s", e)
        
        logging.info("Health monitor has exited.")

    # -----------------
    def log_health_event(self, level, event_type, message):
        """Creates a structured health event and puts it on the health queue."""
        if not self.cfg.get("health_monitor", {}).get("enabled", True):
            return
        
        event = {
            "level": level,
            "event_type": event_type,
            "message": message.replace(",", ";") # Sanitize commas for CSV
        }
        try:
            self.health_q.put_nowait(event)
        except queue.Full:
            logging.warning("Health event queue is full. A statistic was dropped.")
            self.log_health_event("WARNING", "LOG_HEALTH_ERROR", "Health event queue is full.")

    # ----------------- LED Status -----------------
    def led_status_loop(self):
        """
        A dedicated thread to control a status LED, indicating the pipeline's
        current state through different blink patterns.
        """
        if not self.led_line:
            return

        logging.info("LED Status thread started and now has control.")
        
        while self.running.is_set():
            # --- NEW DEFAULT: A slow "breathing" pulse for idle ---
            on_time, off_time = 0.1, 2.5 

            try:
                if self.emergency_stop_active.is_set() or self.camera_fatal_error.is_set():
                    on_time, off_time = 0.1, 0.1 # Rapid flicker for ERROR

                elif self.event_in_progress.is_set():
                    on_time, off_time = 0.25, 0.25 # Fast blink for EVENT

                elif self.producer_thread_active():
                    on_time, off_time = 1.0, 1.0 # Slow pulse for CAPTURING
                
                # If none of the above, it remains in the slow "breathing" IDLE state.

                self.led_line.set_value(self.led_pin, gpiod.line.Value.ACTIVE)
                expiry = time.time() + on_time
                while self.running.is_set() and time.time() < expiry:
                    time.sleep(0.1)
                    
                if not self.running.is_set():
                    break
                
                self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                expiry = time.time() + off_time
                while self.running.is_set() and time.time() < expiry:
                    time.sleep(0.1)

            except Exception as e:
                logging.error(f"Error in LED status loop: {e}")
                time.sleep(5)

    # -----------------
    def _led_startup_signal(self):
        """
        Performs a distinct blink pattern to visually signal that all threads
        have started successfully.
        """
        if not self.led_line:
            return
            
        try:
            logging.info("Signaling successful startup with LED pattern...")
            # Turn LED off first in case it was on
            self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
            time.sleep(0.5)
            # Perform a quick triple-blink
            for _ in range(3):
                self.led_line.set_value(self.led_pin, gpiod.line.Value.ACTIVE)
                time.sleep(0.15)
                self.led_line.set_value(self.led_pin, gpiod.line.Value.INACTIVE)
                time.sleep(0.15)
            # The led_status_loop will take over from here.
            logging.info("LED startup signal complete.")
        except Exception as e:
            logging.error(f"Failed to execute LED startup signal: {e}")

    # -----------------
    def _setup_logging(self):
        """Configures the root logger, ensuring all timestamps are in UTC."""
        log_dir = self.general_log_dir
        max_mb = self.cfg.get("janitor", {}).get("log_rotation_mb", 10)
        backup_count = self.cfg.get("janitor", {}).get("log_backup_count", 5)

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "pipeline.log")

        utc_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
        utc_formatter.converter = time.gmtime

        color_utc_formatter = ThreadColorLogFormatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
        color_utc_formatter.converter = time.gmtime

        file_handler = RotatingFileHandler(log_path, maxBytes=max_mb * 1024 * 1024, backupCount=backup_count)
        file_handler.setFormatter(utc_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_utc_formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logging.info("Logging started. All timestamps will be in UTC.")

    # -----------------
    def run(self, duration=0):
        """
        The main entry point to start and run the entire pipeline application.
        """
        self._setup_logging()
        
        shutdown_event = threading.Event()
        def handle_signal(sig, frame):
            if not shutdown_event.is_set():
                logging.warning("Shutdown signal received! Initiating graceful shutdown...")
                self.stop()
                shutdown_event.set()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        logging.info(f"--- Meteora Pipeline v{self.cfg['general']['version']} Starting ---")
        self.start()
        self._led_startup_signal()
        
        logging.info("All threads launched. Setting 'System Ready' signal.")
        self.system_ready.set()

        try:
             if duration > 0:
                 logging.info(f"Running pipeline for a fixed duration of {duration} seconds.")
                 shutdown_event.wait(timeout=duration)
             else:
                logging.info("Running pipeline indefinitely (Ctrl+C to stop).")
                shutdown_event.wait()
        finally:
            if not shutdown_event.is_set():
                logging.info("Main loop finished, initiating shutdown...")
                self.stop()

        logging.info("Pipeline has been stopped. Main thread exiting.")
        time.sleep(2)

    def get_health_statistics(self, filepath):
        """Reads the pre-aggregated health stats CSV file and returns it as a list of dicts."""
        if not os.path.exists(filepath):
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                # The file is already sorted by count, so we just return the rows
                stats = list(reader)
            return stats
        except Exception as e:
            # Return an error that can be displayed on the dashboard
            return [{
                "event_type": "ERROR_READING_STATS",
                "count": 1, 
                "last_message": str(e), 
                "level": "ERROR"
            }]

    def read_last_lines(self, filepath, num_lines=50):
        """
        Efficiently reads the last N lines from a file.
        Returns a list of strings, or an empty list if the file doesn't exist.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Use a deque for efficient appending to the front
                lines = deque(maxlen=num_lines)
                for line in f:
                    lines.append(line.strip())
                return list(lines)
        except FileNotFoundError:
            return ["Log file not found."]
        except Exception as e:
            return [f"Error reading log file: {e}"]

    def get_file_list(self, path):
        """Scans a directory and returns a list of file details."""
        files = []
        if not os.path.isdir(path):
            return files
            
        for entry in os.scandir(path):
            # Skip directories and hidden files
            if entry.is_file() and not entry.name.startswith('.'):
                try:
                    stat = entry.stat()
                    files.append({
                        'name': entry.name,
                        'path': entry.path,
                        'size_mb': f"{stat.st_size / (1024*1024):.2f} MB",
                        'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
                except FileNotFoundError:
                    continue # File might have been deleted between scan and stat
        
        # Sort files by modification time, newest first
        files.sort(key=lambda x: x['mtime'], reverse=True)
        return files

    def is_safe_path(self, base_dirs, path_to_check):
        """
        CRITICAL SECURITY FUNCTION: Checks if a file path is safely within one of
        the allowed base directories to prevent path traversal attacks.
        """
        try:
            # Resolve the absolute path to prevent '..' tricks
            resolved_path = os.path.abspath(path_to_check)
            
            # Check if the resolved path starts with any of the allowed base directories
            for base in base_dirs:
                if resolved_path.startswith(os.path.abspath(base)):
                    return True
        except Exception:
            return False
        return False

    def get_latest_file(self, path, extension=".png"):
        """Finds the most recently modified file with a given extension in a directory."""
        files = []
        if not os.path.isdir(path):
            return None
            
        for entry in os.scandir(path):
            if entry.is_file() and entry.name.lower().endswith(extension):
                try:
                    files.append(entry.path)
                except FileNotFoundError:
                    continue
        
        if not files:
            return None
            
        # Return the file with the latest modification time
        return max(files, key=os.path.getctime)

    def get_core_voltage(self):
        """
        Gets the core voltage for the current Raspberry Pi model, attempting multiple methods for robustness.
        Supports Raspberry Pi 5 (sysfs) and older models (vcgencmd).
        Returns the voltage in Volts, or 0.0 on error.
        """
        # --- Method 1: Try the modern sysfs interface first (for Pi 5) ---
        try:
            # Find the hwmon device for the DA9091 PMIC
            for path in glob.glob('/sys/class/hwmon/hwmon*/name'):
                with open(path, 'r') as f:
                    if f.read().strip() == 'da9091':
                        hwmon_path = os.path.dirname(path)
                        # Find the specific voltage input file labeled "VDD_CORE"
                        for label_path in glob.glob(os.path.join(hwmon_path, 'in*_label')):
                            with open(label_path, 'r') as f_label:
                                if f_label.read().strip() == 'VDD_CORE':
                                    input_path = label_path.replace('_label', '_input')
                                    with open(input_path, 'r') as f_in:
                                        # Read value (in millivolts) and convert to Volts
                                        return int(f_in.read().strip()) / 1000.0
        except (IOError, ValueError, FileNotFoundError):
            # This will catch permission errors or cases where files don't exist.
            # We don't log an error here, just silently fall back to the next method.
            pass

        # --- Method 2: Fallback to vcgencmd (for Pi 4 and as a backup for Pi 5) ---
        try:
            result = subprocess.run(['vcgencmd', 'measure_volts', 'core'], capture_output=True, text=True, check=True)
            # Output is typically "volt=1.2345V"
            voltage_str = result.stdout.split('=')[1].replace('V', '').strip()
            return float(voltage_str)
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
            # This will catch errors if vcgencmd isn't found, fails, or returns unexpected output.
            pass

        # --- Final Fallback ---
        # If both methods fail, return 0.0
        return 0.0

    def ensure_dir(self, path):
        """Ensure that a directory exists."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def load_and_validate_config(path=None, merge=True):
        """
        Loads a config from a file path, merges with defaults (unless merge=False),
        and validates using the Pydantic schema. Exits on validation failure.
        """
        # If the caller only wants the raw, unmerged file, use the legacy loader.
        # This is primarily for the dashboard to get a clean base.
        if not merge:
            return Pipeline.legacy_load_config(path, merge=False)

#        if not IS_PYDANTIC_AVAILABLE:
#            return Pipeline.legacy_load_config(path, merge=True)

        user_cfg = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    user_cfg = json.load(f)
            except json.JSONDecodeError as e:
                print(f"FATAL: Error decoding JSON from config file '{path}': {e}", file=sys.stderr)
                sys.exit(1)
        
        try:
            # Pydantic's model_validate will automatically use the defaults defined
            # in the schema for any keys missing from user_cfg. This is the merge step.
            validated_model = MainConfig.model_validate(user_cfg)
            validated_dict = validated_model.model_dump()
            
            # Only print this message on initial startup
            if path:
                print("Configuration loaded and validated successfully.")
            return validated_dict
            
        except ValueError as e:
            print("="*60, file=sys.stderr)
            print("FATAL: CONFIGURATION LOGIC VALIDATION FAILED", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Error details:\n{e}", file=sys.stderr)
            print("\nPlease correct your config file and try again.", file=sys.stderr)
            sys.exit(1)
        except Exception as e: # Catches Pydantic's built-in ValidationError
            print("="*60, file=sys.stderr)
            print("FATAL: CONFIGURATION SCHEMA VALIDATION FAILED", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Error details:\n{e}", file=sys.stderr)
            print("\nPlease correct your config file and try again.", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def legacy_load_config(path=None, merge=True):
        # This is the old load_config, kept as a fallback.
        import copy
        if merge:
            cfg = json.loads(json.dumps(DEFAULTS))    # copy.deepcopy(DEFAULTS)
        else:
            cfg = {}
        if path and os.path.exists(path):
            with open(path, "r") as f:
                user_cfg = json.load(f)
            if merge:
                for k, v in user_cfg.items():
                    if isinstance(v, dict) and k in cfg:
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
            else:
                cfg = user_cfg.copy()
        return cfg

    @staticmethod
    def save_json_atomic(path, data):
        try:
            temp_path = path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=4)
            os.rename(temp_path, path)
        except Exception as e:
            logging.error("Failed to save JSON atomically to %s: %s", path, e)

# -------------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meteora Pipeline Pi5")
    parser.add_argument("--config", type=str, help="Path to config JSON file", default="config.json")
    parser.add_argument("--create_config", action="store_true", help="Create a default config.json file and exit.")
    parser.add_argument("--duration", type=int, help="Run for a fixed duration in seconds (0 = infinite)", default=0)
    args = parser.parse_args()

    # --- Handle --create-config argument ---
    if args.create_config:
        config_path = "config.json" # Define a default name
        if os.path.exists(config_path):
            print(f"Configuration file '{config_path}' already exists. No action taken.")
        else:
            try:
                # Use the static method to ensure consistency
                Pipeline.save_json_atomic(config_path, DEFAULTS)
                print(f"Successfully created default configuration file at '{config_path}'.")
                print("Please review and edit this file before running the pipeline.")
            except IOError as e:
                print(f"ERROR: Could not write configuration file to '{config_path}': {e}", file=sys.stderr)
        sys.exit(0) # Exit after handling config creation

    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found at '{args.config}'.", file=sys.stderr)
        print("Please create one or run with --create-config to generate a default file.", file=sys.stderr)
        sys.exit(1)

    # 1. Load the configuration from the file.
    config = Pipeline.load_and_validate_config(args.config)

    # 2. Create an instance of the main application class.
    app = Pipeline(cfg=config, config_path=args.config)

    # 3. Run the application.
    app.run(duration=args.duration)
