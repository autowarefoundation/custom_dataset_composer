import sys
import yaml
import numpy as np
import cv2

from src.utils.utils import Utils

if __name__ == '__main__':
    if len(sys.argv) == 2:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        # Path of train or val of custom dataset:
        path_txt = sys.argv[1]

        # Check whether the file exists in the path:
        if not path_txt.endswith('.txt'):
            print('The file must be a txt file!')
            exit(1)

        utils = Utils()

        # Read train or valid txt file line by line:
        with open(path_txt) as f:
            lines_train_val = f.readlines()
            if len(lines_train_val) == 0:
                print('The file is empty!')
                exit(1)

            for line_img_path in lines_train_val:
                path_img = line_img_path.strip()
                path_label = path_img.replace('.jpg', '.txt')
                img = np.array(cv2.imread(path_img))
                img_height = img.shape[0]
                img_width = img.shape[1]
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                print("[INFO] Image: ", path_img)
                print("[INFO] Label: ", path_label)
                list_yolo_labels = []

                # Read label file line by line:
                with open(path_label) as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        print('The label file is empty!')
                        exit(1)

                    for line in lines:
                        line = line.split(' ')
                        if len(line) != 5 or len(lines) == 0:
                            print('The label file is not correct!')
                            exit(1)

                        class_name = line[0]
                        x_center_norm = float(line[1])
                        y_center_norm = float(line[2])
                        width_norm = float(line[3])
                        height_norm = float(line[4])

                        yolo_label = {
                            "class_id": class_name,
                            "x_center_norm": x_center_norm,
                            "y_center_norm": y_center_norm,
                            "width_norm": width_norm,
                            "height_norm": height_norm
                        }

                        bbox_width = img_width * yolo_label["width_norm"]  # d
                        bbox_height = img_height * yolo_label["height_norm"]  # c
                        bbox_cx = img_width * yolo_label["x_center_norm"]  # a
                        bbox_cy = img_height * yolo_label["y_center_norm"]  # b

                        image = cv2.circle(image, (int(bbox_cx), int(bbox_cy)), radius=0, color=(255, 0, 0), thickness=3)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(bbox_cx), int(bbox_cy))
                        fontScale = 0.4
                        fontColor = (255, 255, 255)
                        thickness = 1
                        lineType = 2

                        cv2.putText(image, class_name,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                        top_left_corner = (int(bbox_cx - (bbox_width / 2)), int(bbox_cy - (bbox_height / 2)))
                        bottom_right_corner = (int(bbox_cx + (bbox_width / 2)), int(bbox_cy + (bbox_height / 2)))
                        color = (255, 0, 0)
                        thickness = 1
                        image = cv2.rectangle(image, bottom_right_corner, top_left_corner, color, thickness)

                cv2.imshow('Sample with YOLO Labels', image)
                cv2.waitKey(1000)
    else:
        print("[ERROR] Wrong number of arguments.")




