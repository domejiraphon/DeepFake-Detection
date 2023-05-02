import torch
import torch.nn.functional as F
from torch.nn import ReLU


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model, target_layer, devices):
        self.model = model
        # self.gradients = None
        self.target_layer = target_layer

        self.device = devices
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        self.resnet_layer = list(dict(self.model._modules.items()).keys())
        layer = dict(self.model._modules.items())[
            self.resnet_layer[self.target_layer]]

        layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs

        list_conv = self.model._modules.items()
        for pos, module in list_conv:
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass

        input_image.requires_grad = True
        target_class = target_class
        model_output = self.model(input_image)

        # Zero gradients

        self.model.zero_grad()
        # Target for backprop

        one_hot_output = F.one_hot(
            target_class, num_classes=model_output.size()[-1]
        ).to(self.device)

        # Backward pass
        model_output.backward(gradient=one_hot_output)

        return self.gradients
