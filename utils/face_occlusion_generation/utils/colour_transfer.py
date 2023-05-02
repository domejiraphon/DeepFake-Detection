import numpy as np
import torch


# import cupy as cp

def gaussian(x_square, sigma):
    # return cp.exp(-0.5*x_square/sigma**2)
    # return np.exp(-0.5*x_square/sigma**2)
    return torch.exp(-0.5 * x_square / sigma ** 2).cuda()


# https://github.com/avivelka/Bilateral-Filter/blob/0127ff4f30c3108af3a5fc8780eb30a40817da02/src/vectorized_bilateral_filter.py
def bilateral_filter2(image, sigma_space, sigma_intensity):
    # kernel_size should be twice the sigma space to avoid calculating negligible values
    kernel_size = int(2 * sigma_space + 1)
    half_kernel_size = int(kernel_size / 2)
    # result = cp.zeros(image.shape)
    result = np.zeros(image.shape)
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size + 1):
        for y in range(-half_kernel_size, half_kernel_size + 1):
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            # shifted_image = cp.roll(image, [x, y], [1, 0])
            shifted_image = np.roll(image, [x, y], [1, 0])
            intensity_difference_image = image - shifted_image
            Gintenisity = gaussian(
                intensity_difference_image ** 2, sigma_intensity)
            result += Gspace * Gintenisity * shifted_image
            W += Gspace * Gintenisity

    return result / W


def bilateral_filter(image, sigma_space, sigma_intensity):
    # kernel_size should be twice the sigma space to avoid calculating negligible values
    kernel_size = int(2 * sigma_space + 1)
    half_kernel_size = int(kernel_size / 2)
    # result = cp.zeros(image.shape)
    image = torch.from_numpy(image).float().cuda()
    result = torch.zeros_like(image)
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size + 1):
        for y in range(-half_kernel_size, half_kernel_size + 1):
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            # shifted_image = cp.roll(image, [x, y], [1, 0])
            # shifted_image = np.roll(image, [x, y], [1, 0])
            shifted_image = torch.roll(image, [x, y], [1, 0])
            intensity_difference_image = image - shifted_image
            Gintenisity = gaussian(
                intensity_difference_image ** 2, sigma_intensity)
            result += Gspace * Gintenisity * shifted_image
            W += Gspace * Gintenisity

    return result / W


# https://github.com/iperov/DeepFaceLab/blob/master/core/imagelib/color_transfer.py
def color_transfer_sot2(src, trg, steps=15, batch_size=5, reg_sigmaXY=16,
                        reg_sigmaV=30):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer
    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter
    return value - clip it manually
    """
    # if not np.issubdtype(src.dtype, np.floating):
    #     raise ValueError("src value must be float")
    # if not np.issubdtype(trg.dtype, np.floating):
    #     raise ValueError("trg value must be float")
    # src= cp.array(src)
    # trg= cp.array(trg)
    src = np.array(src)
    trg = np.array(trg)

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h, w, c = src.shape
    new_src = src.copy()

    # advect = cp.empty ( (h*w,c), dtype=src_dtype )
    advect = np.empty((h * w, c), dtype=src_dtype)
    for step in range(steps):
        advect.fill(0)
        for batch in range(batch_size):
            """
            dir = cp.random.normal(size=c).astype(src_dtype)
            dir /= cp.linalg.norm(dir)

            projsource = cp.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = cp.sum( trg*dir, axis=-1).reshape ((h*w))

            idSource = cp.argsort (projsource)
            idTarget = cp.argsort (projtarget)
            """
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= np.linalg.norm(dir)

            projsource = np.sum(new_src * dir, axis=-1).reshape((h * w))
            projtarget = np.sum(trg * dir, axis=-1).reshape((h * w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)
            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_src += advect.reshape((h, w, c)) / batch_size

    if reg_sigmaXY != 0.0:
        # src_diff = (new_src-src).astype(cp.float32)
        src_diff = (new_src - src).astype(np.float32)
        # try:
        try:

            # src_diff_filt = bilateral_filter(src_diff, reg_sigmaV, reg_sigmaXY )
            R_bf = bilateral_filter(src_diff[:, :, 0], reg_sigmaV, reg_sigmaXY)
            G_bf = bilateral_filter(src_diff[:, :, 1], reg_sigmaV, reg_sigmaXY)
            B_bf = bilateral_filter(src_diff[:, :, 2], reg_sigmaV, reg_sigmaXY)
            # src_diff_filt = cp.stack([R_bf, G_bf, B_bf], axis=2)
            src_diff_filt = np.stack([R_bf, G_bf, B_bf], axis=2)
        except Exception as e:
            print(e)

        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
    # return cp.asnumpy(new_src)
    return new_src


def color_transfer_sot(src, trg, steps=15, batch_size=5, reg_sigmaXY=16,
                       reg_sigmaV=30):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer
    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter
    return value - clip it manually
    """
    # if not np.issubdtype(src.dtype, np.floating):
    #     raise ValueError("src value must be float")
    # if not np.issubdtype(trg.dtype, np.floating):
    #     raise ValueError("trg value must be float")
    # src= cp.array(src)
    # trg= cp.array(trg)
    src = torch.from_numpy(np.array(src)).cuda()
    trg = torch.from_numpy(np.array(trg)).cuda()

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    # src_dtype = src.dtype
    h, w, c = src.shape
    new_src = src

    # advect = cp.empty ( (h*w,c), dtype=src_dtype )
    advect = torch.zeros((h * w, c)).float().cuda()
    for step in range(steps):
        for batch in range(batch_size):
            """
            dir = cp.random.normal(size=c).astype(src_dtype)
            dir /= cp.linalg.norm(dir)

            projsource = cp.sum( new_src*dir, axis=-1).reshape ((h*w))
            projtarget = cp.sum( trg*dir, axis=-1).reshape ((h*w))

            idSource = cp.argsort (projsource)
            idTarget = cp.argsort (projtarget)
            """
            # dir = np.random.normal(size=c).astype(src_dtype)
            # dir /= np.linalg.norm(dir)
            dir = torch.randn(c, dtype=torch.float32).cuda()
            dir /= torch.linalg.norm(dir)

            projsource = torch.sum(new_src * dir, dim=-1).view((h * w))
            projtarget = torch.sum(trg * dir, dim=-1).view((h * w))

            idSource = torch.argsort(projsource)
            idTarget = torch.argsort(projtarget)
            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_src += advect.view((h, w, c)) / batch_size

    if reg_sigmaXY != 0.0:
        # src_diff = (new_src-src).astype(cp.float32)
        src_diff = (new_src - src)  # .astype(np.float32)
        # try:
        try:

            # src_diff_filt = bilateral_filter(src_diff, reg_sigmaV, reg_sigmaXY )
            R_bf = bilateral_filter(src_diff[:, :, 0], reg_sigmaV, reg_sigmaXY)
            G_bf = bilateral_filter(src_diff[:, :, 1], reg_sigmaV, reg_sigmaXY)
            B_bf = bilateral_filter(src_diff[:, :, 2], reg_sigmaV, reg_sigmaXY)
            # src_diff_filt = cp.stack([R_bf, G_bf, B_bf], axis=2)
            src_diff_filt = torch.stack([R_bf, G_bf, B_bf], dim=2)
        except Exception as e:
            print(e)

        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
    # return cp.asnumpy(new_src)
    return new_src
