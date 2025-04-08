import nibabel as nib
import matplotlib.pyplot as plt

nii = nib.load('.....')
data = nii.get_fdata()

# Display a middle slice:
plt.imshow(data[:, :, data.shape[2]//2], cmap='gray')
plt.show()
