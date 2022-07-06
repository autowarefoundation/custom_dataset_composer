import json
from pathlib import Path
import cv2
from src.utils.utils import Utils
import concurrent.futures


class BDD100K(object):
    def __init__(self, config):
        self.config = config
        self.utils = Utils()
        self.num_worker = int(config['count_max_process'])

    def process_frame(self, frame_data):
        is_train_set, frame = frame_data
        img_name = Path(frame["name"])
        assert img_name.suffix == ".jpg"
        frame_name = str(img_name.stem)

        if is_train_set:
            path_image = self.config["bdd100k"][3]["path_folder_images_train"] + "/" + str(img_name)
        else:
            path_image = self.config["bdd100k"][4]["path_folder_images_val"] + "/" + str(img_name)

        img = cv2.imread(path_image, 1)
        path_img_copy = "datasets/bdd100k/images/" + str(img_name)
        cv2.imwrite(path_img_copy, img)

        # Read json file to generate YOLO label file:
        path_label_file_txt = "datasets/bdd100k/yolo_labels/" + frame_name + ".txt"
        list_yolo_labels = []
        with open(path_label_file_txt, "w+") as f:
            # For each sub label of each image, get the box2d variable
            # Get the relative center point compared to the image size 1280/720

            for label in frame["labels"]:
                if "box2d" not in label:
                    continue
                box2d = label["box2d"]
                if box2d["x1"] >= box2d["x2"] or box2d["y1"] >= box2d["y2"]:
                    continue
                cx, cy, width, height = self.box2d_to_yolo(box2d)
                lbl_dataset = str(label["category"])

                universal_class_id = None
                if lbl_dataset == "car":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][0]['car']
                elif lbl_dataset == "bus":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][1]['bus']
                elif lbl_dataset == "person":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][2]['person']
                elif lbl_dataset == "bike":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][3]['bike']
                elif lbl_dataset == "truck":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][4]['truck']
                elif lbl_dataset == "motor":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][5]['motor']
                elif lbl_dataset == "train":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][6]['train']
                elif lbl_dataset == "rider":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][7]['rider']
                elif lbl_dataset == "traffic sign":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][8]['traffic sign']
                elif lbl_dataset == "traffic light":
                    universal_class_id = self.config["bdd100k"][0]['map_bdd100k_id_to_universal_id'][9]['traffic light']
                elif universal_class_id is None:
                    print("Unknown label: ", lbl_dataset)
                    print(universal_class_id)
                else:
                    print("Error")

                yolo_label = \
                    {
                        "class_id": universal_class_id,
                        "x_center_norm": cx,
                        "y_center_norm": cy,
                        "width_norm": width,
                        "height_norm": height
                    }

                list_yolo_labels.append(yolo_label)
                f.write("{} {} {} {} {}\n".format(universal_class_id, cx, cy, width, height))

        f.close()

    def export_image_and_label_files(self):
        path_label_json_train = self.config["bdd100k"][1]["path_file_train_json"]
        frames_train = json.load(open(path_label_json_train, "r"))

        list_frames_train = []
        for frame in frames_train:
            sample = (True, frame)
            list_frames_train.append(sample)

        print("[INFO] Number of training frames: ", len(list_frames_train))
        executor = concurrent.futures.ProcessPoolExecutor(self.num_worker)
        futures = [executor.submit(self.process_frame, frame) for frame in list_frames_train]
        concurrent.futures.wait(futures)

        print("[INFO] Done exporting training image and label files of BDD100K Dataset ...")

        path_label_json_val = self.config["bdd100k"][2]["path_file_val_json"]
        frames_val = json.load(open(path_label_json_val, "r"))

        list_frames_val = []
        for frame in frames_val:
            sample = (False, frame)
            list_frames_val.append(sample)

        print("[INFO] Number of validation frames: ", len(list_frames_val))

        executor = concurrent.futures.ProcessPoolExecutor(self.num_worker)
        futures = [executor.submit(self.process_frame, frame) for frame in list_frames_val]
        concurrent.futures.wait(futures)

        print("[INFO] Done exporting valid image and label files of BDD100K Dataset ...")

    def box2d_to_yolo(self, box2d):
        IMG_WIDTH = 1280
        IMG_HEIGHT = 720
        x1 = box2d["x1"] / IMG_WIDTH
        x2 = box2d["x2"] / IMG_WIDTH
        y1 = box2d["y1"] / IMG_HEIGHT
        y2 = box2d["y2"] / IMG_HEIGHT

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        return cx, cy, width, height
