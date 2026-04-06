import tensorflow as tf

import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
import os
import glob
import cv2
from src.utils.utils import Utils

# Not show warnings of tf:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Waymo(object):
    def __init__(self, config):
        self.config = config
        self.utils = Utils()

    def export_image_and_label_files(self):
        list_path_tfrecord = glob.glob(
            self.config["waymo"][1]["path_folder_tfrecord"] + "/*.tfrecord")
        count_segment = len(list_path_tfrecord)
        print("[INFO] Waymo Open Dataset segment count: ", count_segment)

        if count_segment == 0:
            print("Waymo Open Dataset segment count is 0. Please check the path.")
            return
        else:

            for id_tfrecord, path_tfrecord in enumerate(list_path_tfrecord):
                print("[INFO] Processing segment: ", id_tfrecord + 1, "/", count_segment)
                self.process_tfrecord(path_tfrecord)

    def process_tfrecord(self, path_tfrecord):
        segment_name = path_tfrecord.split("/")[-1].split(".")[0].split("_with")[0].split("-")[-1]

        # Read segment:
        dataset = tf.data.TFRecordDataset(path_tfrecord, compression_type='')

        # Iterate over frames in the segment:
        for frame_id, data in enumerate(dataset):
            frame = open_dataset.Frame()
            context = open_dataset.Context()
            frame.ParseFromString(bytearray(data.numpy()))

            for camera in frame.images:
                waymo_camera_id = camera.name

                # print("frame_id: ", frame_id, " camera_id: ", waymo_camera_id)

                # Take only specific camera ids written in config file:
                if waymo_camera_id not in self.config["waymo"][2]["camera_ids"]:
                    # print("Skip camera_id: ", waymo_camera_id)
                    continue
                else:
                    # print("frame_id: ", frame_id, " camera_id: ", waymo_camera_id)

                    # Iterate over frame labels to find related camera:
                    for tf_label in frame.camera_labels:
                        if tf_label.name != waymo_camera_id:
                            continue
                        else:
                            label_count = len(tf_label.labels)
                            if label_count == 0:
                                # print("No labels in frame_id: ", frame_id, " camera_id: ", waymo_camera_id)
                                continue
                            else:
                                # At this point processing image and label file:
                                image = np.array(tf.image.decode_jpeg(camera.image))
                                path_file_image = "datasets/waymo/images/" + str(segment_name) + '_frame_' + str(
                                    frame_id) + '_camera_' + str(waymo_camera_id) + ".jpg"
                                path_file_label = "datasets/waymo/yolo_labels/" + str(segment_name) + '_frame_' + str(
                                    frame_id) + '_camera_' + str(waymo_camera_id) + ".txt"
                                # Open .txt file write labels:
                                # print("Writing labels to file: ", path_file_label)
                                file_label = open(path_file_label, "w+")

                                # Iterate over labels to transform YOLO format:
                                list_single_image_labels = []
                                for label in tf_label.labels:
                                    # Image size:
                                    img_height = image.shape[0]
                                    img_width = image.shape[1]
                                    # BBox coordinates:
                                    class_id = label.type
                                    class_id_universal = None
                                    # Mapping between Waymo Open Dataset classes and universal classes:
                                    if int(class_id) == 1:
                                        class_id_universal = self.config["waymo"][0]['map_waymo_id_to_universal_id'][0][
                                            'vehicle']
                                        # print("Class id: ", int(class_id), class_id_universal)
                                    if int(class_id) == 2:
                                        class_id_universal = self.config["waymo"][0]['map_waymo_id_to_universal_id'][1][
                                            'pedestrian']
                                        # print("Class id: ", int(class_id), class_id_universal)
                                    if int(class_id) == 4:
                                        class_id_universal = self.config["waymo"][0]['map_waymo_id_to_universal_id'][2][
                                            'cyclist']
                                        # print("Class id: ", int(class_id), class_id_universal)

                                    bbox_cx = label.box.center_x
                                    bbox_cy = label.box.center_y
                                    bbox_width = label.box.length
                                    bbox_height = label.box.width

                                    yolo_label = {
                                        "class_id": class_id_universal,
                                        "x_center_norm": bbox_cx / img_width,
                                        "y_center_norm": bbox_cy / img_height,
                                        "width_norm": bbox_width / img_width,
                                        "height_norm": bbox_height / img_height
                                    }

                                    list_single_image_labels.append(yolo_label)
                                    row = str(yolo_label["class_id"]) + ' ' + \
                                          str(yolo_label["x_center_norm"]) + \
                                          ' ' + str(yolo_label["y_center_norm"]) + ' ' + \
                                          str(yolo_label["width_norm"]) \
                                          + ' ' + \
                                          str(yolo_label["height_norm"])
                                    # Write labels row by row in YOLO format:
                                    file_label.write(row + '\n')

                                # print(np.array(list_single_image_labels))
                                # Close .txt file and write image:
                                cv2.imwrite(path_file_image, image)
                                file_label.close()

                                # self.utils.show_img_with_yolo_labels(image, list_single_image_labels)

        print("[INFO] Done.")
