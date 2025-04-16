from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt

# 初始化 MTCNN
detector = MTCNN()

# 读取测试图像
img_path = r"C:\Users\Lzp\Desktop\DalaoPlan\SimFuPulse\Visual\ubfc-rppg\img1.png"  # 需要替换成你的视频帧截图
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

# 进行人脸检测
boxes, _ = detector.detect(image_rgb)

# 显示检测结果
if boxes is not None:
    for (x, y, x2, y2) in boxes:
        cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
