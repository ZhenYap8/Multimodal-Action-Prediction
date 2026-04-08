# Multimodal Action Prediction

## Overview

This project explores whether human action intention can be predicted from multimodal spatial cues, specifically **gaze direction** and **hand position**. Inspired by research on action understanding and predictive perception, the system uses a supervised machine learning pipeline to classify which object a subject is likely to interact with during an unfolding action.

The project frames action prediction as a binary classification problem between two possible targets:

- Tennis ball
- Orange

Rather than treating the task as simple object classification, the broader aim is to model how intention may be inferred from partial visual information over time.

---

## Motivation

Humans can often anticipate what another person is about to do before the action is complete. This ability is thought to depend on integrating multiple cues, including:

- eye gaze
- hand movement
- temporal progression of the action

This project investigates whether a computational model can approximate part of that process using engineered spatial features derived from video frames.

Potential application areas include:

- human-robot interaction
- assistive robotics
- intention-aware interfaces
- behavioural prediction systems

---

## Dataset

The dataset consists of multiple short video clips in which a subject interacts with one of two objects. Each video is decomposed into labelled frames.

Each row in the dataset represents a single frame and includes:

- `video` – source video identifier
- `frame` – frame index / action progression
- `gaze_x`, `gaze_y` – gaze position
- `hand_x`, `hand_y` – hand position
- `tennis_x`, `tennis_y` – tennis ball position
- `orange_x`, `orange_y` – orange position
- `target` – ground truth object label

Example:

```csv
video,frame,gaze_x,gaze_y,hand_x,hand_y,tennis_x,tennis_y,orange_x,orange_y,target
17.mov,0,827,299,730,996,424,992,1100,999,orange
18.mov,0,812,270,723,994,363,999,1067,1044,tennis