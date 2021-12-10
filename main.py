import torchvision.datasets as dset
import torch
import torchvision.transforms as transforms
from dcgan import DCGAN 
from wgan import WGAN


if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 999

    # Root directory for dataset
    dataroot = "data/mnist"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 64

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of training epochs
    num_epochs = 100

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    cuda = (ngpu > 0)

    # Clamp parameters to a range [-c, c], c=weight_cliping_limit
    weight_cliping_limit = 0.01 

    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)


    model_name = "wgan"
    train = True
    if model_name == "dcgan":
        model = DCGAN(workers, batch_size, image_size, num_epochs, lr, beta1, ngpu)
    elif model_name == "wgan":
        model = WGAN(cuda, batch_size, lr, weight_cliping_limit, num_epochs)
    else:
        print("model not found")
        exit(1)
    if train: 
        model.train(dataloader)
    else:
        model.evaluate("./weight/G.pt")
