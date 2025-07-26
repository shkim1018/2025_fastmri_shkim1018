import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice

class DataTransform_aug:
    def __init__(self, isforward, max_key, augmentor = None, mask_augmentor = None):
        self.isforward = isforward
        self.max_key = max_key
        if augmentor is not None:
            self.augmentor = augmentor
            self.use_augment = augmentor.aug_on
        else :
            self.use_augment = False

        if mask_augmentor is not None:
            self.mask_augmentor = mask_augmentor
            self.use_mask_augment = mask_augmentor.mask_aug_on
        else :
            self.use_mask_augment = False
            
    def __call__(self, mask, input, target, attrs, fname, slice):


        
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # print(f'input: {input.shape}')
        # print(f'target: {target.shape}')
        # # print(f'mask: {mask.shape}')


        #transform input's final dimension to 2dim and apply mask.
        if self.use_mask_augment and self.mask_augmentor.schedule_p() > 0.0:
            kspace, mask = self.mask_augmentor(input, mask) #mask : original mask. it is used for the case not augmented.
        else:
            kspace = to_tensor(input * mask)
            kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
            #transform nd.array to tensor
            mask = to_tensor(mask)
            
        # kspace = torch.stack((kspace.real, kspace.imag), dim=-1)

        # print(f'kspace: {kspace.shape}')
        
        # set_mask_dim = kspace.shape[-2]
        
        # print(f"augmetor_schedule :, {self.augmentor.schedule_p()}")
        # Apply augmentations if needed
        if self.use_augment: 
            if self.augmentor.schedule_p() > 0.0:             
                # kspace, target = self.augmentor(kspace, target.shape)
                kspace, target = self.augmentor(kspace, target, target.shape)

        # print(f'aug_kspace: {kspace.shape}')
        # print(f'aug_target: {target.shape}')
        # print(mask.shape)
        
        # #if shape is not augmented:
        # if set_mask_dim == kspace.shape[-2]:
        #     mask = torch.from_numpy(mask.reshape(1, 1, set_mask_dim, 1).astype(np.float32)).byte()
        # #if shape is reversed:
        # else:
        #     mask = torch.from_numpy(mask.reshape(1, set_mask_dim, 1, 1).astype(np.float32)).byte()

        # mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()


        mask = mask.reshape(1, 1, kspace.shape[-2], 1).float().byte()
        
        return mask, kspace, target, maximum, fname, slice

# #원본 code
# def __call__(self, mask, input, target, attrs, fname, slice):
#         if not self.isforward:
#             target = to_tensor(target)
#             maximum = attrs[self.max_key]
#         else:
#             target = -1
#             maximum = -1

#         # print(f'input: {input.shape}')
#         # print(f'target: {target.shape}')
#         # # print(f'mask: {mask.shape}')
        
#         kspace = to_tensor(input * mask)
#         kspace = torch.stack((kspace.real, kspace.imag), dim=-1)

#         # print(f'kspace: {kspace.shape}')
        
#         # set_mask_dim = kspace.shape[-2]
        
#         # print(f"augmetor_schedule :, {self.augmentor.schedule_p()}")
#         # Apply augmentations if needed
#         if self.use_augment: 
#             if self.augmentor.schedule_p() > 0.0:             
#                 # kspace, target = self.augmentor(kspace, target.shape)
#                 kspace, target = self.augmentor(kspace, target, target.shape)

#         # print(f'aug_kspace: {kspace.shape}')
#         # print(f'aug_target: {target.shape}')
#         # print(mask.shape)
        
#         # #if shape is not augmented:
#         # if set_mask_dim == kspace.shape[-2]:
#         #     mask = torch.from_numpy(mask.reshape(1, 1, set_mask_dim, 1).astype(np.float32)).byte()
#         # #if shape is reversed:
#         # else:
#         #     mask = torch.from_numpy(mask.reshape(1, set_mask_dim, 1, 1).astype(np.float32)).byte()

#         mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
            

#         return mask, kspace, target, maximum, fname, slice