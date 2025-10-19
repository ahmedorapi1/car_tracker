# 🚗 Car Tracker (YOLO + OpenCV)

A real-time object detection and tracking system built with **YOLOv11** and **OpenCV** to detect and track moving vehicles from a video feed or live camera stream.

---

## 🧠 Overview

This project uses **Ultralytics YOLOv11** for object detection and a simple **Euclidean distance tracker** for assigning persistent IDs to detected cars across frames.  
It demonstrates how to combine deep learning detection with classical tracking for lightweight, efficient motion analysis.

---

## ✨ Features

- 🚘 Detects and tracks multiple cars simultaneously  
- 🎯 Assigns unique IDs to each vehicle  
- 📹 Works on video files or live webcam streams  
- ⚡ Uses YOLOv11 (Ultralytics) for fast, accurate detection  
- 🧩 Easily extendable to support other object classes or trackers (e.g., SORT, ByteTrack)

---

## 🗂️ Project Structure

car_tracker/
│
├── detector.py # Handles YOLO model loading and detection
├── tracker.py # Main script for reading video, tracking cars
├── highway.mp4 # Sample video (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation
