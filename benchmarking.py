import torch
from QBI_radon import Radon

# import libraries
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
plt.rcParams["figure.figsize"] = [7, 7]
import numpy as np
import time

img = shepp_logan_phantom()

n_angles = 256
# create a batch of 32 images
batch_size = 128
images = np.zeros((batch_size, 1, img.shape[0], img.shape[0]))

for i in range(batch_size):
    images[i, 0, :, :] = img

thetas_skimage = np.linspace(0, 180, n_angles)
thetas_qbi = np.linspace(0, np.pi, n_angles)
#calculate time for radon forward
sinograms = []
start_time = time.time()# apply radon forward by radon skimage
for i in range(batch_size):
    sinograms.append(radon(images[i, 0, :, :], theta=thetas_skimage, circle=False))
sinograms_skimage = np.array(sinograms)
time_radon_forward_skimage = time.time() - start_time
print(f"Time for radon forward: {time_radon_forward_skimage} seconds")
# calculate time for radon forward by QBI_radon
device = torch.device("cuda")
images_torch = torch.FloatTensor(images).to(device)
start_time = time.time()
radon_op = Radon(thetas=thetas_qbi, circle=False, device=device, filter_name="ramp")
sinograms_qbi = radon_op.forward(images_torch)
time_radon_forward_qbi = time.time() - start_time
print(f"Time for radon forward: {time_radon_forward_qbi} seconds")

# apply radon backward by radon skimage
reconstructed_images = []
start_time = time.time()
for i in range(batch_size):
    reconstructed_images.append(iradon(sinograms_skimage[i, :, :], theta=thetas_skimage, circle=False))
reconstructed_images_skimage = np.array(reconstructed_images)
time_radon_backward_skimage = time.time() - start_time
print(f"Time for radon backward: {time_radon_backward_skimage} seconds")
# apply radon backward by QBI_radon
start_time = time.time()
reconstructed_images_qbi = radon_op.filter_backprojection(sinograms_qbi)
time_radon_backward_qbi = time.time() - start_time
print(f"Time for radon backward: {time_radon_backward_qbi} seconds")
# calculate the difference between the reconstructed images and the original images
# difference_skimage = np.linalg.norm(images - reconstructed_images_skimage) / np.linalg.norm(images)
# difference_qbi = np.linalg.norm(images - reconstructed_images_qbi.cpu().numpy()) / np.linalg.norm(images)
# print(f"Difference between the reconstructed images and the original images: {difference_skimage}")
# print(f"Difference between the reconstructed images and the original images: {difference_qbi}")

# create a figure to show the time comparison forward and backward put the time in a same table


# Build a grouped bar chart similar to the provided example, using throughput (images/second)
def _format_int(x):
    try:
        return f"{int(round(x))}"
    except Exception:
        return "0"

forward_ips_skimage = batch_size / time_radon_forward_skimage if time_radon_forward_skimage > 0 else 0.0
backward_ips_skimage = batch_size / time_radon_backward_skimage if time_radon_backward_skimage > 0 else 0.0
forward_ips_qbi = batch_size / time_radon_forward_qbi if time_radon_forward_qbi > 0 else 0.0
backward_ips_qbi = batch_size / time_radon_backward_qbi if time_radon_backward_qbi > 0 else 0.0

categories = ['forward', 'backward']
astra = [0, 0]  # placeholder (not available in this benchmark)
skimage_vals = [forward_ips_skimage, backward_ips_skimage]
qbi_vals = [forward_ips_qbi, backward_ips_qbi]

x = np.arange(len(categories))
width = 0.28

fig, ax = plt.subplots(figsize=(9, 6))

rects1 = ax.bar(x - width, skimage_vals, width, label='radon skimage', color='#1f77b4')
rects2 = ax.bar(x, qbi_vals, width, label='QBI_radon', color='#ff7f0e')

# Optional third series could be half precision if available; omitted here

ax.set_ylabel('Images/second')
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
ax.set_title(f"Image size {img.shape[0]}x{img.shape[1]}, {n_angles} angles and batch size {batch_size} on a {device_name}")
ax.set_xticks(x, categories)
ax.legend()

def autolabel(rects):
    for r in rects:
        height = r.get_height()
        ax.annotate(_format_int(height),
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('benchmarking.png')


