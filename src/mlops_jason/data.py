import torch
import typer

DATA_ROOT = "C:/Users/Notandi/mlops/testformlops"  # path to project


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images.

    Subtracts the mean and divides by the standard deviation for the entire
    batch, producing images with mean 0 and std 1.

    Args:
        images (torch.Tensor): The images to be normalized.

    Returns:
        torch.Tensor: A new tensor with normalized images.
    """
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """
    Preprocess raw data and save the resulting tensors.

    Loads training and testing images/targets from the raw directory, reshapes
    them, converts to the correct data types, normalizes, and saves them to
    the processed directory.

    Args:
        raw_dir (str): Directory containing raw data files.
        processed_dir (str): Directory to save the processed data files.

    Returns:
        None
    """
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load the processed MNIST dataset and return training and testing datasets.

    Loads the previously saved processed data (images and targets) and returns
    them as TensorDatasets.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            A tuple of (train_set, test_set).
    """
    train_images = torch.load(f"{DATA_ROOT}/data/processed/train_images.pt")
    train_target = torch.load(f"{DATA_ROOT}/data/processed/train_target.pt")
    test_images = torch.load(f"{DATA_ROOT}/data/processed/test_images.pt")
    test_target = torch.load(f"{DATA_ROOT}/data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
