import os
import cv2
import torch
import time
import logging
import threading
import colorsys
import numpy as np


logger = logging.getLogger(__name__)
GOLDEN_RATIO = 0.618033988749895


def draw_detections(frame, detections, classes, color=(255, 255, 255), show_conf=True):
    for det, cls in zip(detections, classes):
        text = f"{det[4]:.2f}: {cls}" if show_conf else None
        draw_bbox(frame, det[:4], color, 1, text)


def get_color(idx, s=0.8, vmin=0.7):
    h = np.fmod(idx * GOLDEN_RATIO, 1.0)
    v = 1.0 - np.fmod(idx * GOLDEN_RATIO, 1.0 - vmin)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(255 * b), int(255 * g), int(255 * r)


def draw_bbox(frame, tlbr, color, thickness, text=None):
    tlbr = np.asarray(tlbr, dtype=int)
    tlbr = tlbr.astype(int)
    tl, br = tuple(tlbr[:2]), tuple(tlbr[2:])
    cv2.rectangle(frame, tl, br, color, thickness)
    if text is not None:
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            tl,
            (tl[0] + text_width - 1, tl[1] + text_height - 1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (tl[0], tl[1] + text_height - 1),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            0,
            1,
            cv2.LINE_AA,
        )

def calculate_density(detections, class_names):
  class_weights = {
      "car": 2,
      "bus": 4,
      "truck": 4,
      "person": 0.5,
      "bicycle": 0.5,
      "motorcycle": 1
  }

  detections = np.array(detections)
  traffic_density = 0.0
  for det, cls in zip(detections, class_names):
    if cls in class_weights:
      traffic_density += class_weights[cls]
  return traffic_density


def update_density(density_score, name, lock):
    name_to_idx = {"east": 0,
                   "south": 1,
                   "west": 2,
                   "north": 3}
    with lock, open("val.txt", "r+") as file:
        line = file.read()
        densities = line.split(" ")
        densities[name_to_idx[name]] = str(density_score)
        file.seek(0)
        file.writelines(" ".join(densities))
        file.truncate()


def infer_on_video(video_path, frame_write_path, name, thread_lock):
  vidReader = cv2.VideoCapture(video_path)
  columns_to_extract = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']

  num_frames_processed = 0
  infer_time = []

  while True:
    ret, frame = vidReader.read()
    if not ret:
      logger.warning(
          f"No frame could be retrieved from provided video {video_path}\n Exiting"
          )
      return

    cv2.imwrite(f"{frame_write_path}/{name}.jpg", frame) # input image

    tick = time.time()
    detections = yolov5(frame)
    tock = time.time()
    num_frames_processed += 1
    infer_time.append(tock - tick)

    detections = detections.pandas().xyxy[0]
    detections = detections.loc[:, columns_to_extract]
    detections = np.array(detections.values.tolist())

    bbox_with_conf = np.array(detections[:, :5], dtype=float)
    class_names = detections[:, -1]

    draw_detections(frame, bbox_with_conf, class_names)
    cv2.imwrite(f"{frame_write_path}/{name}_bbox.jpg", frame) # bbox image

    # DENSITY CALCULATION AND DUMPING CODE HERE
    curr_traffic_density_score = calculate_density(bbox_with_conf, class_names)
    update_density(curr_traffic_density_score, name, thread_lock)

    if (num_frames_processed) % 100 == 0:
      logger.debug(f"Processed {num_frames_processed} of video {video_path}")
      logger.info(f"Mean FPS: {1 / np.mean(infer_time)}")
      logger.info(f"Median FPS: {1 / np.median(infer_time)}")


def infer_multiple_videos(video_path_list, img_out_dir):
  logger.info(f"Output images would be dumped at path: {img_out_dir}")
  thread_lock = threading.Lock()
  inference_threads = []
  for vid_path in video_path_list:
    logger.info(f"Starting inference on video: {vid_path}")
    out_img_name = str(os.path.basename(vid_path)).split('_')[0]
    vid_infer_thread = threading.Thread(
        target=infer_on_video, 
        args=(vid_path,
              img_out_dir,
              out_img_name, 
              thread_lock)
        )
    vid_infer_thread.start()
    inference_threads.append(vid_infer_thread)

  for infer_thread in inference_threads:
    infer_thread.join()


if __name__ == '__main__':
  
  logging.basicConfig(filename="./app.log",filemode="w",level=logging.DEBUG,
      format="%(asctime)s [%(levelname)8s] [%(filename)s:%(lineno)d]: %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S")
  

  os.system("pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt")
  yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m')

  video_path_list = [r"C:\Users\ARYAN RAWAT\minor_project\Test\test1\east_video.mp4",
                     r"C:\Users\ARYAN RAWAT\minor_project\Test\test1\north_video.mp4",
                     r"C:\Users\ARYAN RAWAT\minor_project\Test\test1\south_video.mp4",
                     r"C:\Users\ARYAN RAWAT\minor_project\Test\test1\west_video.mp4"] # Add video paths, videos should be named direction_something.vidFormat
  img_out_dir = r"C:\ Users\ARYAN RAWAT\ minor_project\ YOLOv5_results" 
  # Set image_out_dir

  #os.makedirs(img_out_dir, exist_ok=True)
  infer_multiple_videos(video_path_list, img_out_dir)
