# Project Panopticon: Intelligent Exam Proctoring

## Overview

Project Panopticon is a Machine Learning time-series classification project for intelligent online exam proctoring.

The system analyzes video telemetry and computer system events to identify potentially suspicious behavior while reducing false accusations.

## Objective

The objective is to build a classification pipeline that:

- Aligns asynchronous video and system-event data.
- Handles missing video sensor data without removing the exam timeline.
- Uses rolling-window features to reduce short-term movement noise.
- Uses a weighted classification model.
- Uses probability-based predictions and a 90% confidence threshold to prioritize precision.

## Dataset

### Video Telemetry

The video telemetry contains:

- `timestamp`
- `eye_gaze_angle`
- `audio_db`

### System Events

The system-event data contains:

- `timestamp`
- `tab_switches`
- `is_cheating`

## Data Processing

### Missing Values

Missing eye-gaze and audio values are handled using Pandas interpolation rather than deleting rows.

### Asynchronous Data Alignment

The two datasets have different event frequencies, so `pandas.merge_asof()` is used to align events according to timestamp.

### Rolling Features

10-second rolling averages are created for:

- Gaze
- Audio

These features help reduce the effect of very short movements or noise.

## Machine Learning

A `RandomForestClassifier` is used with:

```python
class_weight="balanced"