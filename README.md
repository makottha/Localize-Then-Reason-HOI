# Localize-Then-Reason: Zero-Shot Human–Object Interaction Detection  
### YOLO + CLIP for Cyclist Detection in Real-World Traffic Video

This repository provides the official implementation of our paper:

**Localize–Then–Reason: A Vision–Language Framework for Zero-Shot Human–Object Interaction Detection (Cyclists as a Case Study)**  
Manikanta Kotthapalli, Portland State University  
(2025)

This codebase implements a **hybrid detection pipeline** that combines:

1. **YOLOv8/YOLOv11** for spatial grounding of persons and bicycles  
2. **CLIP / OpenCLIP VLMs** for semantic reasoning to determine whether a person is *riding* a bicycle  
3. **Zero-shot HOI classification** using prompt ensembles (no labels or fine-tuning required)

Our approach replaces heavy supervised cyclist labeling with **vision–language reasoning**, achieving strong accuracy in real-world video captured from Portland-area traffic cameras.

---

## 🔍 Why This Matters

Cyclist detection is critical for:
- Urban planning  
- Transportation research  
- Multimodal traffic monitoring  
- Safety analytics  
- Infrastructure planning (bike lanes, intersections, bus stops)

Traditional detectors (YOLO, Faster R-CNN) cannot differentiate:
- A **person riding** a bicycle  
vs.  
- A **person standing near** a bicycle  
- A **person walking** a bicycle  
- A **parked bicycle near pedestrians**

Our method solves this via **VLM relational reasoning**—a new approach for HOI detection in surveillance and smart-city applications.

---

## 🧠 Key Idea: *Localize–Then–Reason*

We break HOI detection into two steps:

### **1. Detect person + bicycle (YOLO)**
YOLO localizes instances:
- person
- bicycle

Then we create person–bicycle candidate pairs using IoU fusion:

### **2. Reason about the interaction (CLIP / OpenCLIP)**  
Each candidate crop is classified using a prompt ensemble:

**Rider prompts**
- “a person riding a bicycle”
- “a cyclist biking on the street”
- “a person on a moving bicycle”

**Non-rider prompts**
- “a person walking a bicycle”
- “a bicycle without a rider”
- “a person standing next to a bicycle”

This enables **zero-shot classification** without cyclist labels.


