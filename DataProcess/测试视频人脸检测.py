import cv2
import numpy as np
from facenet_pytorch import MTCNN


class FaceProcessor:
    def __init__(self):
        self.detector = MTCNN(keep_all=False)  # 只保留最大人脸
        self.face_box = None  # 用于存储第一帧检测到的坐标

    import cv2
    import numpy as np

    def process_frames(self, frames):
        """ 记录所有帧的检测框，计算最终裁剪区域，并统一处理所有帧 """
        print("--------------- 开始人脸检测 ---------------")
        face_boxes_list = []  # 记录所有帧的检测框

        # 遍历所有帧，收集人脸框
        for idx, frame in enumerate(frames):
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            face_boxes, _ = self.detector.detect(frame)

            if face_boxes is not None and len(face_boxes) > 0:
                x1, y1, x2, y2 = map(int, face_boxes[0])  # 取第一张人脸
                face_boxes_list.append((x1, y1, x2, y2))
                print(f"第 {idx} 帧：检测到人脸，坐标：{x1, y1, x2, y2}")

        # 确保至少检测到一张人脸，否则直接返回缩放帧
        if not face_boxes_list:
            print("未检测到任何人脸，将缩放整帧处理！")
            return np.array([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (128, 128)) for frame in frames])

        # 计算最小外接矩形
        min_x1 = min(box[0] for box in face_boxes_list)
        min_y1 = min(box[1] for box in face_boxes_list)
        max_x2 = max(box[2] for box in face_boxes_list)
        max_y2 = max(box[3] for box in face_boxes_list)

        # 计算宽高并扩展 10%
        w, h = max_x2 - min_x1, max_y2 - min_y1
        min_x1 = max(0, int(min_x1 - 0.1 * w))
        min_y1 = max(0, int(min_y1 - 0.1 * h))
        max_x2 = min(frames[0].shape[1], int(max_x2 + 0.1 * w))
        max_y2 = min(frames[0].shape[0], int(max_y2 + 0.1 * h))

        print(f"最终裁剪区域：({min_x1}, {min_y1}, {max_x2}, {max_y2})")

        # 处理所有帧
        processed_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = rgb_frame[min_y1:max_y2, min_x1:max_x2]
            face_resized = cv2.resize(face, (128, 128))
            processed_frames.append(face_resized)

        print("--------------- 人脸检测完成 ---------------")
        return np.array(processed_frames)


def process_video(input_video_path, output_video_path):
    """ 读取视频并进行人脸检测 """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)

    cap.release()

    # 处理人脸检测
    processor = FaceProcessor()
    processed_frames = processor.process_frames(frame_list)

    # 保存视频
    height, width = 128, 128
    fps = 30  # 可根据输入视频调整
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in processed_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 转回 BGR 再写入

    # for frame in processed_frames:
    #     out.write(frame)

    out.release()
    print(f"视频已保存至 {output_video_path}")


# 测试代码
if __name__ == '__main__':


    input_video = f'C:/Users\Lzp/Desktop/dataset/p002/v01/video_RAW_YUV420.avi'  # 替换成你的输入视频路径
    output_video = f'processed_video4.avi'
    process_video(input_video, output_video)
