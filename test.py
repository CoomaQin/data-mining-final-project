import torch
from model import *
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    ngpu = 1
    device = device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # input to the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    generated_imgs = []

    # load the trained generator
    modelG = Generator(ngpu).to(device)
    modelG.load_state_dict(torch.load("./weight/G.pt"))
    modelG.eval()
    with torch.no_grad():
        fake = modelG(fixed_noise).detach().cpu()
        generated_imgs = vutils.make_grid(fake, padding=2, normalize=True)
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(generated_imgs, (1, 2, 0)))
    plt.show()