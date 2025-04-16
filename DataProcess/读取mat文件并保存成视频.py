# import scipy.io
# import numpy as np
# import cv2
#
# # 读取 .mat 文件
# mat_file_path = f"E:/datasets/mini_MMPD/subject1/p1_0.mat"
# mat_data = scipy.io.loadmat(mat_file_path)
#
# # 提取视频数据
# video_data = mat_data.get("video")
# if video_data is None:
#     raise ValueError("video 数据不存在！")
#
# # 确保数据类型为 uint8，并归一化到 [0, 255]
# video_uint8 = (video_data * 255).clip(0, 255).astype(np.uint8)
#
# # 统一分辨率到 128x128
# video_resized = np.array([cv2.resize(frame, (128, 128)) for frame in video_uint8])
#
# # 保存为 AVI 视频
# output_path = "output_video.avi"
# fps = 30
# height, width = 128, 128
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
# for frame in video_resized:
#     video_writer.write(frame)
#
# video_writer.release()
# print(f"视频已保存至 {output_path}")