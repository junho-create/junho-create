import json
from pathlib import Path
from typing import List


def read_file(path: str, supported_formats: str = ".json") -> dict:
    """Read a file and return its content as a string

    Args:
        path (str): the path to the file to read
        supported_formats (str, optional): the supported file formats. Defaults to ".json".

    Returns:
        dict: the json content of the file

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if the file format is not supported
    """

    path = Path(path)

    # check if the file exists and is a file
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File {path} not found")

    # check if the file format is supported
    if path.suffix not in supported_formats:
        raise ValueError(f"File format {path.suffix} not supported")

    with path.open("r") as file:
        file_content = json.load(file)

    return file_content


def create_directory(path: str) -> None:
    """Create a directory if it does not exist

    Args:
        path (str): the path to the directory to create
    """

    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def read_file_paths(path: str, supported_formats: List[str] = [".jpg"]) -> List[str]:
    """Read files in a directory and return their content as a list of strings

    Args:
        path (str): the path to the directory containing the file paths to read
        supported_formats (List[str], optional): the supported file formats. Defaults to [".jpg"].

    Returns:
        list: list of valid file paths

    Raises:
        FileNotFoundError: if the directory does not exist
    """

    path = Path(path)

    # check if the directory exists and is a directory
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory {path} not found")

    # get the list of files in the directory
    file_paths = [file for file in path.iterdir() if file.is_file()]

    # filter file paths based on the supported formats
    if supported_formats:
        file_paths = [file for file in file_paths if file.suffix in supported_formats]
    else:
        file_paths = []

    return file_paths


def check_dataset_format(data: dict, image_key: str) -> None:
    """Check the format of the dataset

    Args:
        data (dict): the gt/prediction dataset to check
        image_key (str): the image name acting as the key in the dataset

    Raises:
        ValueError: if a key is missing in the dataset
    """
    if data[image_key].get("elements") is None:
        raise ValueError(
            f"{image_key} does not have 'elements' key in the json file. "
            "Check if you are passing the correct data."
        )

    elements = data[image_key]["elements"]
    for elem in elements:
        if elem.get("category") is None:
            raise ValueError(
                f"{image_key} does not have 'category' key in the ground truth file. "
                "Check if you are passing the correct data."
            )

        if elem.get("content") is None:
            raise ValueError(
                f"{image_key} does not have 'content' key in the ground truth file. "
                "Check if you are passing the correct data."
            )
        else:
            content = elem["content"]
            if content.get("text") is None:
                raise ValueError(
                    f"{image_key} does not have 'text' key in the ground truth file. "
                    "Check if you are passing the correct data."
                )


def check_data_validity(gt_data: dict, pred_data: dict) -> None:
    """Check the validity of the ground truth and prediction data

    Args:
        gt_data (dict): the ground truth data
        pred_data (dict): the prediction data

    Raises:
        ValueError: if the ground truth or prediction data is invalid
    """

    if not gt_data:
        raise ValueError("Ground truth data is empty")

    if not pred_data:
        raise ValueError("Prediction data is empty")

    for image_key in gt_data.keys():
        pred_elem = pred_data.get(image_key)
        if pred_data is None:
            raise ValueError(
                f"{image_key} not found in prediction. "
                "Check if you are passing the correct data."
            )

    for image_key in gt_data.keys():
        check_dataset_format(gt_data, image_key)

    for image_key in pred_data.keys():
        check_dataset_format(pred_data, image_key)
