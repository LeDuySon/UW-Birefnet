from __future__ import annotations

import os
import random
from typing import List

import cv2
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.preproc import preproc
from src.utils import path_to_image


Image.MAX_IMAGE_PIXELS = None  # remove DecompressionBombWarning
# config = Config()

_class_labels_TR_sorted = (
    "Airplane, Ant, Antenna, Archery, Axe, BabyCarriage, Bag, BalanceBeam, Balcony, Balloon, Basket, BasketballHoop, Beatle, Bed, Bee, Bench, Bicycle, "
    "BicycleFrame, BicycleStand, Boat, Bonsai, BoomLift, Bridge, BunkBed, Butterfly, Button, Cable, CableLift, Cage, Camcorder, Cannon, Canoe, Car, "
    "CarParkDropArm, Carriage, Cart, Caterpillar, CeilingLamp, Centipede, Chair, Clip, Clock, Clothes, CoatHanger, Comb, ConcretePumpTruck, Crack, Crane, "
    "Cup, DentalChair, Desk, DeskChair, Diagram, DishRack, DoorHandle, Dragonfish, Dragonfly, Drum, Earphone, Easel, ElectricIron, Excavator, Eyeglasses, "
    "Fan, Fence, Fencing, FerrisWheel, FireExtinguisher, Fishing, Flag, FloorLamp, Forklift, GasStation, Gate, Gear, Goal, Golf, GymEquipment, Hammock, "
    "Handcart, Handcraft, Handrail, HangGlider, Harp, Harvester, Headset, Helicopter, Helmet, Hook, HorizontalBar, Hydrovalve, IroningTable, Jewelry, Key, "
    "KidsPlayground, Kitchenware, Kite, Knife, Ladder, LaundryRack, Lightning, Lobster, Locust, Machine, MachineGun, MagazineRack, Mantis, Medal, MemorialArchway, "
    "Microphone, Missile, MobileHolder, Monitor, Mosquito, Motorcycle, MovingTrolley, Mower, MusicPlayer, MusicStand, ObservationTower, Octopus, OilWell, "
    "OlympicLogo, OperatingTable, OutdoorFitnessEquipment, Parachute, Pavilion, Piano, Pipe, PlowHarrow, PoleVault, Punchbag, Rack, Racket, Rifle, Ring, Robot, "
    "RockClimbing, Rope, Sailboat, Satellite, Scaffold, Scale, Scissor, Scooter, Sculpture, Seadragon, Seahorse, Seal, SewingMachine, Ship, Shoe, ShoppingCart, "
    "ShoppingTrolley, Shower, Shrimp, Signboard, Skateboarding, Skeleton, Skiing, Spade, SpeedBoat, Spider, Spoon, Stair, Stand, Stationary, SteeringWheel, "
    "Stethoscope, Stool, Stove, StreetLamp, SweetStand, Swing, Sword, TV, Table, TableChair, TableLamp, TableTennis, Tank, Tapeline, Teapot, Telescope, Tent, "
    "TobaccoPipe, Toy, Tractor, TrafficLight, TrafficSign, Trampoline, TransmissionTower, Tree, Tricycle, TrimmerCover, Tripod, Trombone, Truck, Trumpet, Tuba, "
    "UAV, Umbrella, UnevenBars, UtilityPole, VacuumCleaner, Violin, Wakesurfing, Watch, WaterTower, WateringPot, Well, WellLid, Wheel, Wheelchair, WindTurbine, Windmill, WineGlass, WireWhisk, Yacht"
)
class_labels_TR_sorted = _class_labels_TR_sorted.split(", ")


class MyData(data.Dataset):
    def __init__(self, datasets, image_size, config, is_train=True):
        self.size_train = image_size
        self.size_test = image_size
        self.keep_size = not config.size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        self.preproc_methods = config.preproc_methods
        self.auxiliary_classification = config.auxiliary_classification
        self.device = config.device
        if self.is_train and config.auxiliary_classification:
            self.cls_name2id = {
                _name: _id for _id, _name in enumerate(class_labels_TR_sorted)
            }

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ][self.load_all or self.keep_size :]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
            ][self.load_all or self.keep_size :]
        )

        dataset_root = os.path.join(config.data_root_dir, config.task)
        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in datasets.split("+"):
            image_root = os.path.join(dataset_root, dataset, "im")
            self.image_paths += [
                os.path.join(image_root, p) for p in os.listdir(image_root)
            ]
        self.label_paths = []

        for p in self.image_paths:
            for ext in [".png", ".jpg", ".PNG", ".JPG", ".JPEG"]:
                ## 'im' and 'gt' may need modifying
                p_gt = p.replace("/im/", "/gt/")[: -(len(p.split(".")[-1]) + 1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                print("Not exists:", p_gt)

        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(
                zip(self.image_paths, self.label_paths), total=len(self.image_paths)
            ):
                _image = path_to_image(
                    image_path, size=(config.size, config.size), color_type="rgb"
                )
                _label = path_to_image(
                    label_path, size=(config.size, config.size), color_type="gray"
                )
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split("/")[-1].split("#")[3]]
                    if self.is_train and config.auxiliary_classification
                    else -1
                )

    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = (
                self.class_labels_loaded[index]
                if self.is_train and self.auxiliary_classification
                else -1
            )
        else:
            image = path_to_image(
                self.image_paths[index], size=self.data_size, color_type="rgb"
            )
            label = path_to_image(
                self.label_paths[index], size=self.data_size, color_type="gray"
            )
            class_label = (
                self.cls_name2id[self.label_paths[index].split("/")[-1].split("#")[3]]
                if self.is_train and self.auxiliary_classification
                else -1
            )

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=self.preproc_methods)
        # else:
        #     if _label.shape[0] > 2048 or _label.shape[1] > 2048:
        #         _image = cv2.resize(_image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        #         _label = cv2.resize(_label, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)


def visualize_augmentation(dataset, origin_index, augmented_index, save_path):
    # Get original image and label
    orig_image, orig_label, _ = dataset[origin_index]

    # Get augmented image and label (assuming it's the next item after all original images)
    aug_image, aug_label, _ = dataset[augmented_index]

    # Function to denormalize and convert tensor to numpy array
    def tensor_to_numpy(tensor):
        denormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            ]
        )
        return denormalize(tensor).permute(1, 2, 0).cpu().numpy()

    # Convert tensors to numpy arrays
    orig_image_np = tensor_to_numpy(orig_image)
    aug_image_np = tensor_to_numpy(aug_image)
    orig_label_np = orig_label.squeeze().cpu().numpy()
    aug_label_np = aug_label.squeeze().cpu().numpy()

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Original vs Augmented Image and Label", fontsize=16)

    # Plot original image
    axs[0, 0].imshow(orig_image_np)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # Plot original label
    axs[0, 1].imshow(orig_label_np, cmap="gray")
    axs[0, 1].set_title("Original Label")
    axs[0, 1].axis("off")

    # Plot augmented image
    axs[1, 0].imshow(aug_image_np)
    axs[1, 0].set_title("Augmented Image")
    axs[1, 0].axis("off")

    # Plot augmented label
    axs[1, 1].imshow(aug_label_np, cmap="gray")
    axs[1, 1].set_title("Augmented Label")
    axs[1, 1].axis("off")

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Visualization saved to {save_path}")


# Dataset customize for andre segmentation dataset
class CarSegmentationData(data.Dataset):
    def __init__(self, datasets, config, is_train=True):
        self.size_train = config.size
        self.size_test = config.size
        self.keep_size = not config.size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        self.preproc_methods = config.preproc_methods
        self.auxiliary_classification = config.auxiliary_classification
        self.device = config.device

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ][self.load_all or self.keep_size :]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
            ][self.load_all or self.keep_size :]
        )

        dataset_root = config.data_root_dir
        dataset_list = datasets.split("+")
        print(f"Dataset root: {dataset_root}")
        print(f"Loading datasets: {dataset_list}")

        # datasets can be a list of different datasets for training on combined sets.
        self.image_paths = []
        for dataset in dataset_list:
            image_root = os.path.join(dataset_root, dataset, "images")
            self.image_paths += self.get_paths(image_root)

        # load mask paths
        self.label_paths = []
        for dataset in dataset_list:
            label_root = os.path.join(dataset_root, dataset, "gt", "masks")
            self.label_paths += self.get_paths(label_root)

        carvana_augmentation = (
            config.carvana_augmentation if hasattr(config, "carvana_augmentation") else None
        )
        print(f"Carvana augmentation path: {carvana_augmentation}")

        # sorted image and label paths
        self.image_paths = sorted(self.image_paths, key=lambda x: os.path.basename(x))
        self.label_paths = sorted(self.label_paths, key=lambda x: os.path.basename(x))
        self.validate_data_pair(self.image_paths, self.label_paths)

        if is_train and carvana_augmentation is not None:
            carvana_images = self.get_paths(carvana_augmentation + "/train")
            carvana_masks = self.get_paths(carvana_augmentation + "/train_masks")
            self.image_paths.extend(carvana_images)
            self.label_paths.extend(carvana_masks)


        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            self.class_labels_loaded = []
            # for image_path, label_path in zip(self.image_paths, self.label_paths):
            for image_path, label_path in tqdm(
                zip(self.image_paths, self.label_paths), total=len(self.image_paths)
            ):
                _image = path_to_image(
                    image_path, size=self.data_size, color_type="rgb"
                )
                _label = path_to_image(
                    label_path, size=self.data_size, color_type="gray"
                )
                self.images_loaded.append(_image)
                self.labels_loaded.append(_label)
                self.class_labels_loaded.append(
                    self.cls_name2id[label_path.split("/")[-1].split("#")[3]]
                    if self.is_train and config.auxiliary_classification
                    else -1
                )

        if is_train:
            self.background_paths = self.get_paths(
                config.augmentation_path + "/testval"
            )
            self.background_paths.extend(
                self.get_paths(config.augmentation_path + "/train")
            )

    #     if config.augmentation_path is not None and is_train:
    #         subfolder = "/train" if self.is_train else "/test"
    #         self._gen_data(config.augmentation_path + subfolder)

    # def _gen_data(self, background_path: str):
    #     print(f"Initializing augmentation with backgrounds from: {background_path}")
    #     augmented_images = []
    #     augmented_labels = []
    #     augmented_class_labels = []

    #     backgrounds = self.get_paths(background_path)

    #     for i in tqdm(range(len(self.image_paths)), desc="Augmenting images"):
    #         for bg_file in backgrounds:
    #             bg_path = bg_file
    #             background = Image.open(bg_path).convert('RGB')

    #             # Resize background to match the data size
    #             background = background.resize(self.data_size)

    #             # Load original image and label
    #             original_image = self.images_loaded[i] if self.load_all else Image.open(self.image_paths[i]).convert('RGB')
    #             original_label = self.labels_loaded[i] if self.load_all else Image.open(self.label_paths[i]).convert('L')

    #             # Resize original image and label if necessary
    #             if not self.keep_size:
    #                 original_image = original_image.resize(self.data_size)
    #                 original_label = original_label.resize(self.data_size)

    #             # Create a new image by pasting the original onto the background
    #             augmented_image = background.copy()
    #             augmented_image.paste(original_image, (0, 0), original_label)

    #             # Add augmented image and corresponding label to lists
    #             augmented_images.append(augmented_image)
    #             augmented_labels.append(original_label)

    #             if self.is_train and self.auxiliary_classification:
    #                 augmented_class_labels.append(self.class_labels_loaded[i] if self.load_all else
    #                                               self.cls_name2id[self.label_paths[i].split("/")[-1].split("#")[3]])

    #     self.images_loaded.extend(augmented_images)
    #     self.labels_loaded.extend(augmented_labels)
    #     if self.is_train and self.auxiliary_classification:
    #         self.class_labels_loaded.extend(augmented_class_labels)

    #     print(f"Augmentation complete. Total dataset size: {len(self.images_loaded)}")

    def _background_augmentation(self, image, label):
        # 50% chance to apply background augmentation
        if random.random() < 0.5:
            return image, label

        # Randomly select a background image
        bg_path = random.choice(self.background_paths)
        background = Image.open(bg_path).convert("RGB")

        # Resize background to match the data size
        background = background.resize(self.data_size)

        # Resize original image and label if necessary
        if not self.keep_size:
            image = image.resize(self.data_size)
            label = label.resize(self.data_size)

        # Create a new image by pasting the original onto the background
        augmented_image = background.copy()
        augmented_image.paste(image, (0, 0), label)

        return augmented_image, label

    def get_paths(self, data_path: str, exts = (".png", ".jpg", ".PNG", ".JPG", ".JPEG")):
        # get image paths recursively
        paths = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(exts):
                    paths.append(os.path.join(root, file))
        return paths

    def validate_data_pair(self, image_paths: List[str], label_paths: List[str]):
        print("Validating image and label paths")
        for img_path, lbl_path in zip(image_paths, label_paths):
            img_name = os.path.basename(img_path).split(".")[0]
            lbl_name = os.path.basename(lbl_path).split(".")[0]
            assert (
                img_name == lbl_name
            ), f"Image and label names do not match: {img_name} != {lbl_name}"
        print("Validation complete")

    def __getitem__(self, index):
        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
            class_label = (
                self.class_labels_loaded[index]
                if self.is_train and self.auxiliary_classification
                else -1
            )
        else:
            image = path_to_image(
                self.image_paths[index], size=self.data_size, color_type="rgb"
            )
            label = path_to_image(
                self.label_paths[index], size=self.data_size, color_type="gray"
            )
            class_label = (
                self.cls_name2id[self.label_paths[index].split("/")[-1].split("#")[3]]
                if self.is_train and self.auxiliary_classification
                else -1
            )

        # loading image and label
        if self.is_train:
            image, label = self._background_augmentation(image, label)
            image, label = preproc(image, label, preproc_methods=self.preproc_methods)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label, class_label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="carvana_augmentation.yaml",
)
def main(cfg: DictConfig):
    config = hydra.utils.instantiate(cfg.get("config"))
    if config.task != "":
        dataset = CarSegmentationData(
            datasets=config.training_set, config=config, is_train=True
        )
        for i in range(1):
            visualize_augmentation(
                dataset, i, len(dataset) - i - 1, f"augmentation_{i}.png"
            )


if __name__ == "__main__":
    main()
