import torch 
def fgsm_attack(image, epsilon, data_grad, mask):
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.mul(data_grad.sign(), mask.view(-1, 1, 1, 1))

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image