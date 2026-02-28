import cv2
import numpy as np
import os
import time
from PIL import Image

# =====================================================================
# 🚀 ENTERPRISE-GRADE AI TRAINING PIPELINE
# Features: CLAHE Lighting, Blur Detection, Normalization, Augmentation
# =====================================================================

path = 'dataset'
print("\n" + "="*50)
print(" 🧠 AI Core: Initializing Deep Learning Pre-Processing...")
print("="*50)

# 1. ADVANCED LBPH CONFIGURATION
# We increase the grid size (10x10 instead of 8x8) for higher spatial accuracy
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=10, grid_y=10)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Prevents extreme shadows or bright lights from corrupting the face data
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def is_blurry(image, threshold=40.0):
    """
    Computes the Laplacian variance of the image. 
    If the variance is below the threshold, the image is blurry.
    """
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]     
    faceSamples = []
    ids = []
    
    # Analytics Trackers
    total_scanned = 0
    rejected_blur = 0
    rejected_no_face = 0
    augmented_count = 0

    print(f"📁 Found {len(imagePaths)} raw files in dataset. Commencing analysis...\n")

    for imagePath in imagePaths:
        try:
            total_scanned += 1
            
            # Load image and convert to grayscale math array
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract Student ID
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            # Detect faces with strict parameters to avoid background noise
            faces = detector.detectMultiScale(
                img_numpy,
                scaleFactor=1.1,     # Slower but highly accurate
                minNeighbors=7,      # Strict false-positive filter
                minSize=(60, 60)     # Ignore tiny background artifacts
            )

            if len(faces) == 0:
                rejected_no_face += 1
                continue

            for (x, y, w, h) in faces:
                # 1. Crop the face
                face_crop = img_numpy[y:y+h, x:x+w]
                
                # 2. Check for Blur (Quality Control)
                if is_blurry(face_crop):
                    rejected_blur += 1
                    continue
                
                # 3. Apply CLAHE Lighting Correction
                face_crop = clahe.apply(face_crop)
                
                # 4. Standardize Size (Crucial for LBPH math grids)
                face_crop = cv2.resize(face_crop, (200, 200))
                
                # Add original perfect face
                faceSamples.append(face_crop)
                ids.append(id)
                
                # 5. DATA AUGMENTATION: Horizontal Flip
                # Teaches the AI to recognize the face from opposite angles
                face_flip = cv2.flip(face_crop, 1)
                faceSamples.append(face_flip)
                ids.append(id)
                augmented_count += 1
                
        except Exception as e:
            print(f"⚠️ Corrupted file skipped: {imagePath} - {e}")
            
    # Print Analytics Dashboard
    print("-" * 40)
    print("📊 DATASET PIPELINE ANALYTICS:")
    print(f"   Total Raw Images Scanned : {total_scanned}")
    print(f"   ❌ Rejected (No Face)    : {rejected_no_face}")
    print(f"   ❌ Rejected (Too Blurry) : {rejected_blur}")
    print(f"   ✨ Augmented Clones      : +{augmented_count}")
    print(f"   ✅ Final Verified Vectors: {len(faceSamples)}")
    print("-" * 40)
            
    return faceSamples, ids

start_time = time.time()
print("⚙️ Compiling neural vectors and building LBPH Histograms...")

faces, ids = getImagesAndLabels(path)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    
    elapsed = round(time.time() - start_time, 2)
    print(f"\n🏆 SUCCESS: AI Brain 'trainer.yml' compiled in {elapsed} seconds!")
    print(f"👤 System currently recognizes {len(np.unique(ids))} unique student(s).")
    print("="*50 + "\n")
else:
    print("\n❌ CRITICAL ERROR: The data pipeline returned 0 valid faces.")
    print("💡 Fix: Ensure students are well-lit, looking at the camera, and moving slowly.\n")