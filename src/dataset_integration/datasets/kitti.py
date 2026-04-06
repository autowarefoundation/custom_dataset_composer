from src.utils.utils import Utils
import glob
import concurrent.futures
import numpy as np
import cv2


class KITTI(object):
    def __init__(self, config):
        self.config = config
        self.utils = Utils()
        self.num_worker = int(config['count_max_process'])

    def process_frame(self, path_img_file, path_label_file):
        img = np.array(cv2.imread(path_img_file))
        height = float(img.shape[0])
        width = float(img.shape[1])
        name_img_path = 'datasets/kitti/images/' + path_img_file.split('/')[-1].split('.')[0] + '.jpg'
        path_yolo_label_file = 'datasets/kitti/yolo_labels/' + path_label_file.split('/')[-1].split('.')[0] + '.txt'
        cv2.imwrite(name_img_path, img)
        file_label = open(path_yolo_label_file, "a")
        with open(path_label_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                class_name = line[0]
                left = float(line[4])
                top = float(line[5])
                right = float(line[6])
                bottom = float(line[7])

                c = bottom-top
                d = right-left
                a = left+(d/2)
                b = top+(c/2)

                class_id_universal = None
                if class_name == "DontCare":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][0][
                                        "don't care"]
                elif class_name == "Person_sitting":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][1][
                    "person sitting"]
                elif class_name == "Truck":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][2][
                    "truck"]
                elif class_name == "Tram":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][3][
                    "tram"]
                elif class_name == "Cyclist":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][4][
                    "cyclist"]
                elif class_name == "Car":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][5][
                    "car"]
                elif class_name == "Pedestrian":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][6][
                    "pedestrian"]
                elif class_name == "Van":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][7][
                    "van"]
                elif class_name == "Misc":
                    class_id_universal = self.config["kitti"][0]['map_kitti_id_to_universal_id'][8][
                    "misc"]
                else:
                    print("Unknown class name: " + class_name)

                cx_norm = a/width
                cy_norm = b/height
                width_norm = d/width
                height_norm = c/height

                yolo_label = {
                "class_id": str(class_id_universal),
                "x_center_norm": str(cx_norm),
                "y_center_norm": str(cy_norm),
                "width_norm": str(width_norm),
                "height_norm": str(height_norm)
                }

                row = str(yolo_label["class_id"]) + ' ' + \
                  str(yolo_label["x_center_norm"]) + \
                  ' ' + str(yolo_label["y_center_norm"]) + ' ' + \
                  str(yolo_label["width_norm"]) \
                  + ' ' + \
                  str(yolo_label["height_norm"])

                file_label.write(row + '\n')
        file_label.close()

    def export_image_and_label_files(self):
        list_path_img_files = sorted(glob.glob(
            self.config["kitti"][1]["path_folder_training_images"] + "/*.png"))
        list_path_label_files = sorted(glob.glob(
            self.config["kitti"][2]["path_folder_training_labels"] + "/*.txt"))
        assert len(list_path_img_files) == len(list_path_label_files)

        frames = zip(list_path_img_files, list_path_label_files)

        print("[INFO] Number of training frames: ", len(list_path_label_files))
        executor = concurrent.futures.ProcessPoolExecutor(self.num_worker)
        futures = [executor.submit(self.process_frame, path_img_file, path_label_file) for (path_img_file, path_label_file) in frames]
        concurrent.futures.wait(futures)


