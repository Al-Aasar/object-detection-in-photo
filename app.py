import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# تحميل نموذج YOLOv8
@st.cache_resource
def load_model():
    return YOLO("yolov8l.pt")  # يمكن تغييرها إلى yolov8s.pt أو غيره حسب الدقة المطلوبة

model = load_model()

# واجهة التطبيق
st.title("YOLOv8 Object Detection App")
st.write("Upload an image and click **Detect** to see the objects detected.")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("Detect"):
        # التنبؤ باستخدام YOLO
        results = model.predict(source=img_array, save=False, conf=0.3)

        result = results[0]
        boxes = result.boxes
        class_names = model.names

        # نسخة من الصورة للرسم عليها
        annotated_img = img_array.copy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls]} ({conf:.2%})"

            # رسم المربع حول الكائن
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # تكبير الخط حسب حجم الصورة
            font_scale = max(0.7, annotated_img.shape[1] / 1000)
            font_thickness = 4

            # حجم النص
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # خلفية للنص
            cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 0, 0), -1)

            # كتابة النص فوق المربع
            cv2.putText(annotated_img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        # عرض الصورة النهائية
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)
