{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import math\n",
    "\n",
    "# Constraints\n",
    "# Input: [batch_size, in_channels, height, width]\n",
    "\n",
    "# Scaled weight - He initialization\n",
    "# \"explicitly scale the weights at runtime\"\n",
    "class ScaleW:\n",
    "    '''\n",
    "    Constructor: name - name of attribute to be scaled\n",
    "    '''\n",
    "    def __init__(self, name):\n",
    "        self.name = name\n",
    "    \n",
    "    def scale(self, module):\n",
    "        weight = getattr(module, self.name + '_orig')\n",
    "        fan_in = weight.data.size(1) * weight.data[0][0].numel()\n",
    "        \n",
    "        return weight * math.sqrt(2 / fan_in)\n",
    "    \n",
    "    @staticmethod\n",
    "    def apply(module, name):\n",
    "        '''\n",
    "        Apply runtime scaling to specific module\n",
    "        '''\n",
    "        hook = ScaleW(name)\n",
    "        weight = getattr(module, name)\n",
    "        module.register_parameter(name + '_orig', nn.Parameter(weight.data))\n",
    "        del module._parameters[name]\n",
    "        module.register_forward_pre_hook(hook)\n",
    "    \n",
    "    def __call__(self, module, whatever):\n",
    "        weight = self.scale(module)\n",
    "        setattr(module, self.name, weight)\n",
    "\n",
    "# Quick apply for scaled weight\n",
    "def quick_scale(module, name='weight'):\n",
    "    ScaleW.apply(module, name)\n",
    "    return module\n",
    "\n",
    "# Uniformly set the hyperparameters of Linears\n",
    "# \"We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)\"\n",
    "# 5/13: Apply scaled weights\n",
    "class SLinear(nn.Module):\n",
    "    def __init__(self, dim_in, dim_out):\n",
    "        super().__init__()\n",
    "\n",
    "        linear = nn.Linear(dim_in, dim_out)\n",
    "        linear.weight.data.normal_()\n",
    "        linear.bias.data.zero_()\n",
    "        \n",
    "        self.linear = quick_scale(linear)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.linear(x)\n",
    "\n",
    "# Uniformly set the hyperparameters of Conv2d\n",
    "# \"We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)\"\n",
    "# 5/13: Apply scaled weights\n",
    "class SConv2d(nn.Module):\n",
    "    def __init__(self, *args, **kwargs):\n",
    "        super().__init__()\n",
    "\n",
    "        conv = nn.Conv2d(*args, **kwargs)\n",
    "        conv.weight.data.normal_()\n",
    "        conv.bias.data.zero_()\n",
    "        \n",
    "        self.conv = quick_scale(conv)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.conv(x)\n",
    "\n",
    "# Normalization on every element of input vector\n",
    "class PixelNorm(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "\n",
    "    def forward(self, x):\n",
    "        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)\n",
    "\n",
    "# \"learned affine transform\" A\n",
    "class FC_A(nn.Module):\n",
    "    '''\n",
    "    Learned affine transform A, this module is used to transform\n",
    "    midiate vector w into a style vector\n",
    "    '''\n",
    "    def __init__(self, dim_latent, n_channel):\n",
    "        super().__init__()\n",
    "        self.transform = SLinear(dim_latent, n_channel * 2)\n",
    "        # \"the biases associated with ys that we initialize to one\"\n",
    "        self.transform.linear.bias.data[:n_channel] = 1\n",
    "        self.transform.linear.bias.data[n_channel:] = 0\n",
    "\n",
    "    def forward(self, w):\n",
    "        # Gain scale factor and bias with:\n",
    "        style = self.transform(w).unsqueeze(2).unsqueeze(3)\n",
    "        return style\n",
    "    \n",
    "# AdaIn (AdaptiveInstanceNorm)\n",
    "class AdaIn(nn.Module):\n",
    "    '''\n",
    "    adaptive instance normalization\n",
    "    '''\n",
    "    def __init__(self, n_channel):\n",
    "        super().__init__()\n",
    "        self.norm = nn.InstanceNorm2d(n_channel)\n",
    "        \n",
    "    def forward(self, image, style):\n",
    "        factor, bias = style.chunk(2, 1)\n",
    "        result = self.norm(image)\n",
    "        result = result * factor + bias  \n",
    "        return result\n",
    "\n",
    "# \"learned per-channel scaling factors\" B\n",
    "# 5/13: Debug - tensor -> nn.Parameter\n",
    "class Scale_B(nn.Module):\n",
    "    '''\n",
    "    Learned per-channel scale factor, used to scale the noise\n",
    "    '''\n",
    "    def __init__(self, n_channel):\n",
    "        super().__init__()\n",
    "        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))\n",
    "    \n",
    "    def forward(self, noise):\n",
    "        result = noise * self.weight\n",
    "        return result \n",
    "\n",
    "# Early convolutional block\n",
    "# 5/13: Debug - tensor -> nn.Parameter\n",
    "# 5/13: Remove noise generating module\n",
    "class Early_StyleConv_Block(nn.Module):\n",
    "    '''\n",
    "    This is the very first block of generator that get the constant value as input\n",
    "    '''\n",
    "    def __init__ (self, n_channel, dim_latent, dim_input):\n",
    "        super().__init__()\n",
    "        # Constant input\n",
    "        self.constant = nn.Parameter(torch.randn(1, n_channel, dim_input, dim_input))\n",
    "        # Style generators\n",
    "        self.style1   = FC_A(dim_latent, n_channel)\n",
    "        self.style2   = FC_A(dim_latent, n_channel)\n",
    "        # Noise processing modules\n",
    "        self.noise1   = quick_scale(Scale_B(n_channel))\n",
    "        self.noise2   = quick_scale(Scale_B(n_channel))\n",
    "        # AdaIn\n",
    "        self.adain    = AdaIn(n_channel)\n",
    "        self.lrelu    = nn.LeakyReLU(0.2)\n",
    "        # Convolutional layer\n",
    "        self.conv     = SConv2d(n_channel, n_channel, 3, padding=1)\n",
    "    \n",
    "    def forward(self, latent_w, noise):\n",
    "        # Gaussian Noise: Proxyed by generator\n",
    "        # noise1 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()\n",
    "        # noise2 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()\n",
    "        result = self.constant.repeat(noise.shape[0], 1, 1, 1)\n",
    "        result = result + self.noise1(noise)\n",
    "        result = self.adain(result, self.style1(latent_w))\n",
    "        result = self.lrelu(result)\n",
    "        result = self.conv(result)\n",
    "        result = result + self.noise2(noise)\n",
    "        result = self.adain(result, self.style2(latent_w))\n",
    "        result = self.lrelu(result)\n",
    "        \n",
    "        return result\n",
    "    \n",
    "# General convolutional blocks\n",
    "# 5/13: Remove upsampling\n",
    "# 5/13: Remove noise generating\n",
    "class StyleConv_Block(nn.Module):\n",
    "    '''\n",
    "    This is the general class of style-based convolutional blocks\n",
    "    '''\n",
    "    def __init__ (self, in_channel, out_channel, dim_latent):\n",
    "        super().__init__()\n",
    "        # Style generators\n",
    "        self.style1   = FC_A(dim_latent, out_channel)\n",
    "        self.style2   = FC_A(dim_latent, out_channel)\n",
    "        # Noise processing modules\n",
    "        self.noise1   = quick_scale(Scale_B(out_channel))\n",
    "        self.noise2   = quick_scale(Scale_B(out_channel))\n",
    "        # AdaIn\n",
    "        self.adain    = AdaIn(out_channel)\n",
    "        self.lrelu    = nn.LeakyReLU(0.2)\n",
    "        # Convolutional layers\n",
    "        self.conv1    = SConv2d(in_channel, out_channel, 3, padding=1)\n",
    "        self.conv2    = SConv2d(out_channel, out_channel, 3, padding=1)\n",
    "    \n",
    "    def forward(self, previous_result, latent_w, noise):\n",
    "        # Upsample: Proxyed by generator\n",
    "        # result = nn.functional.interpolate(previous_result, scale_factor=2, mode='bilinear',\n",
    "        #                                           align_corners=False)\n",
    "        # Conv 3*3\n",
    "        result = self.conv1(previous_result)\n",
    "        # Gaussian Noise: Proxyed by generator\n",
    "        # noise1 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()\n",
    "        # noise2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()\n",
    "        # Conv & Norm\n",
    "        result = result + self.noise1(noise)\n",
    "        result = self.adain(result, self.style1(latent_w))\n",
    "        result = self.lrelu(result)\n",
    "        result = self.conv2(result)\n",
    "        result = result + self.noise2(noise)\n",
    "        result = self.adain(result, self.style2(latent_w))\n",
    "        result = self.lrelu(result)\n",
    "        \n",
    "        return result    \n",
    "\n",
    "# Very First Convolutional Block\n",
    "# 5/13: No more downsample, this block is the same sa general ones\n",
    "# class Early_ConvBlock(nn.Module):\n",
    "#     '''\n",
    "#     Used to construct progressive discriminator\n",
    "#     '''\n",
    "#     def __init__(self, in_channel, out_channel, size_kernel, padding):\n",
    "#         super().__init__()\n",
    "#         self.conv = nn.Sequential(\n",
    "#             SConv2d(in_channel, out_channel, size_kernel, padding=padding),\n",
    "#             nn.LeakyReLU(0.2),\n",
    "#             SConv2d(out_channel, out_channel, size_kernel, padding=padding),\n",
    "#             nn.LeakyReLU(0.2)\n",
    "#         )\n",
    "    \n",
    "#     def forward(self, image):\n",
    "#         result = self.conv(image)\n",
    "#         return result\n",
    "    \n",
    "# General Convolutional Block\n",
    "# 5/13: Downsample is now removed from block module\n",
    "class ConvBlock(nn.Module):\n",
    "    '''\n",
    "    Used to construct progressive discriminator\n",
    "    '''\n",
    "    def __init__(self, in_channel, out_channel, size_kernel1, padding1, \n",
    "                 size_kernel2 = None, padding2 = None):\n",
    "        super().__init__()\n",
    "        \n",
    "        if size_kernel2 == None:\n",
    "            size_kernel2 = size_kernel1\n",
    "        if padding2 == None:\n",
    "            padding2 = padding1\n",
    "        \n",
    "        self.conv = nn.Sequential(\n",
    "            SConv2d(in_channel, out_channel, size_kernel1, padding=padding1),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            SConv2d(out_channel, out_channel, size_kernel2, padding=padding2),\n",
    "            nn.LeakyReLU(0.2)\n",
    "        )\n",
    "    \n",
    "    def forward(self, image):\n",
    "        # Downsample now proxyed by discriminator\n",
    "        # result = nn.functional.interpolate(image, scale_factor=0.5, mode=\"bilinear\", align_corners=False)\n",
    "        # Conv\n",
    "        result = self.conv(image)\n",
    "        return result\n",
    "        \n",
    "    \n",
    "# Main components\n",
    "class Intermediate_Generator(nn.Module):\n",
    "    '''\n",
    "    A mapping consists of multiple fully connected layers.\n",
    "    Used to map the input to an intermediate latent space W.\n",
    "    '''\n",
    "    def __init__(self, n_fc, dim_latent):\n",
    "        super().__init__()\n",
    "        layers = [PixelNorm()]\n",
    "        for i in range(n_fc):\n",
    "            layers.append(SLinear(dim_latent, dim_latent))\n",
    "            layers.append(nn.LeakyReLU(0.2))\n",
    "            \n",
    "        self.mapping = nn.Sequential(*layers)\n",
    "    \n",
    "    def forward(self, latent_z):\n",
    "        latent_w = self.mapping(latent_z)\n",
    "        return latent_w    \n",
    "\n",
    "# Generator\n",
    "# 5/13: Support progressive training\n",
    "# 5/13: Proxy noise generating\n",
    "# 5/13: Proxy upsampling\n",
    "class StyleBased_Generator(nn.Module):\n",
    "    '''\n",
    "    Main Module\n",
    "    '''\n",
    "    def __init__(self, n_fc, dim_latent, dim_input):\n",
    "        super().__init__()\n",
    "        # Waiting to adjust the size\n",
    "        self.fcs    = Intermediate_Generator(n_fc, dim_latent)\n",
    "        self.convs  = nn.ModuleList([\n",
    "            Early_StyleConv_Block(512, dim_latent, dim_input),\n",
    "            StyleConv_Block(512, 512, dim_latent),\n",
    "            StyleConv_Block(512, 512, dim_latent),\n",
    "            StyleConv_Block(512, 512, dim_latent),\n",
    "            StyleConv_Block(512, 256, dim_latent),\n",
    "            StyleConv_Block(256, 128, dim_latent),\n",
    "            StyleConv_Block(128, 64, dim_latent),\n",
    "            StyleConv_Block(64, 32, dim_latent),\n",
    "            StyleConv_Block(32, 16, dim_latent)\n",
    "        ])\n",
    "        self.to_rgbs = nn.ModuleList([\n",
    "            SConv2d(512, 3, 1),\n",
    "            SConv2d(512, 3, 1),\n",
    "            SConv2d(512, 3, 1),\n",
    "            SConv2d(512, 3, 1),\n",
    "            SConv2d(256, 3, 1),\n",
    "            SConv2d(128, 3, 1),\n",
    "            SConv2d(64, 3, 1),\n",
    "            SConv2d(32, 3, 1),\n",
    "            SConv2d(16, 3, 1)\n",
    "        ])\n",
    "    def forward(self, latent_z, \n",
    "                step = 0,       # Step means how many layers (count from 4 x 4) are used to train\n",
    "                alpha=-1,       # Alpha is the parameter of smooth conversion of resolution):\n",
    "                noise=None,     # TODO: support none noise\n",
    "                mix_steps=[],   # steps inside will use latent_z[1], else latent_z[0]\n",
    "                latent_w_center=None, # Truncation trick in W    \n",
    "                psi=0):               # parameter of truncation\n",
    "        if type(latent_z) != type([]):\n",
    "            print('You should use list to package your latent_z')\n",
    "            latent_z = [latent_z]\n",
    "        if (len(latent_z) != 2 and len(mix_steps) > 0) or type(mix_steps) != type([]):\n",
    "            print('Warning: Style mixing disabled, possible reasons:')\n",
    "            print('- Invalid number of latent vectors')\n",
    "            print('- Invalid parameter type: mix_steps')\n",
    "            mix_steps = []\n",
    "        \n",
    "        latent_w = [self.fcs(latent) for latent in latent_z]\n",
    "        batch_size = latent_w[0].size(0)\n",
    "\n",
    "        # Truncation trick in W    \n",
    "        # Only usable in estimation\n",
    "        if latent_w_center is not None:\n",
    "            latent_w = [latent_w_center + psi * (unscaled_latent_w - latent_w_center) \n",
    "                for unscaled_latent_w in latent_w]\n",
    "        \n",
    "        # Generate needed Gaussian noise\n",
    "        # 5/22: Noise is now generated by outer module\n",
    "        # noise = []\n",
    "        result = 0\n",
    "        current_latent = 0\n",
    "        # for i in range(step + 1):\n",
    "        #     size = 4 * 2 ** i # Due to the upsampling, size of noise will grow\n",
    "        #     noise.append(torch.randn((batch_size, 1, size, size), device=torch.device('cuda:0')))\n",
    "        \n",
    "        for i, conv in enumerate(self.convs):\n",
    "            # Choose current latent_w\n",
    "            if i in mix_steps:\n",
    "                current_latent = latent_w[1]\n",
    "            else:\n",
    "                current_latent = latent_w[0]\n",
    "                \n",
    "            # Not the first layer, need to upsample\n",
    "            if i > 0 and step > 0:\n",
    "                result_upsample = nn.functional.interpolate(result, scale_factor=2, mode='bilinear',\n",
    "                                                  align_corners=False)\n",
    "                result = conv(result_upsample, current_latent, noise[i])\n",
    "            else:\n",
    "                result = conv(current_latent, noise[i])\n",
    "            \n",
    "            # Final layer, output rgb image\n",
    "            if i == step:\n",
    "                result = self.to_rgbs[i](result)\n",
    "                \n",
    "                if i > 0 and 0 <= alpha < 1:\n",
    "                    result_prev = self.to_rgbs[i - 1](result_upsample)\n",
    "                    result = alpha * result + (1 - alpha) * result_prev\n",
    "                    \n",
    "                # Finish and break\n",
    "                break\n",
    "        \n",
    "        return result\n",
    "\n",
    "    def center_w(self, zs):\n",
    "        '''\n",
    "        To begin, we compute the center of mass of W\n",
    "        '''\n",
    "        latent_w_center = self.fcs(zs).mean(0, keepdim=True)\n",
    "        return latent_w_center\n",
    "        \n",
    "\n",
    "# Discriminator\n",
    "# 5/13: Support progressive training\n",
    "# 5/13: Add downsample module\n",
    "# Component of Progressive GAN\n",
    "# Reference: Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017).\n",
    "# Progressive Growing of GANs for Improved Quality, Stability, and Variation, 1–26.\n",
    "# Retrieved from http://arxiv.org/abs/1710.10196\n",
    "class Discriminator(nn.Module):\n",
    "    '''\n",
    "    Main Module\n",
    "    '''\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        # Waiting to adjust the size\n",
    "        self.from_rgbs = nn.ModuleList([\n",
    "            SConv2d(3, 16, 1),\n",
    "            SConv2d(3, 32, 1),\n",
    "            SConv2d(3, 64, 1),\n",
    "            SConv2d(3, 128, 1),\n",
    "            SConv2d(3, 256, 1),\n",
    "            SConv2d(3, 512, 1),\n",
    "            SConv2d(3, 512, 1),\n",
    "            SConv2d(3, 512, 1),\n",
    "            SConv2d(3, 512, 1)\n",
    "       ])\n",
    "        self.convs  = nn.ModuleList([\n",
    "            ConvBlock(16, 32, 3, 1),\n",
    "            ConvBlock(32, 64, 3, 1),\n",
    "            ConvBlock(64, 128, 3, 1),\n",
    "            ConvBlock(128, 256, 3, 1),\n",
    "            ConvBlock(256, 512, 3, 1),\n",
    "            ConvBlock(512, 512, 3, 1),\n",
    "            ConvBlock(512, 512, 3, 1),\n",
    "            ConvBlock(512, 512, 3, 1),\n",
    "            ConvBlock(513, 512, 3, 1, 4, 0)\n",
    "        ])\n",
    "        self.fc = SLinear(512, 1)\n",
    "        \n",
    "        self.n_layer = 9 # 9 layers network\n",
    "    \n",
    "    def forward(self, image, \n",
    "                step = 0,  # Step means how many layers (count from 4 x 4) are used to train\n",
    "                alpha=-1):  # Alpha is the parameter of smooth conversion of resolution):\n",
    "        for i in range(step, -1, -1):\n",
    "            # Get the index of current layer\n",
    "            # Count from the bottom layer (4 * 4)\n",
    "            layer_index = self.n_layer - i - 1 \n",
    "            \n",
    "            # First layer, need to use from_rgb to convert to n_channel data\n",
    "            if i == step: \n",
    "                result = self.from_rgbs[layer_index](image)\n",
    "            \n",
    "            # Before final layer, do minibatch stddev\n",
    "            if i == 0:\n",
    "                # In dim: [batch, channel(512), 4, 4]\n",
    "                res_var = result.var(0, unbiased=False) + 1e-8 # Avoid zero\n",
    "                # Out dim: [channel(512), 4, 4]\n",
    "                res_std = torch.sqrt(res_var)\n",
    "                # Out dim: [channel(512), 4, 4]\n",
    "                mean_std = res_std.mean().expand(result.size(0), 1, 4, 4)\n",
    "                # Out dim: [1] -> [batch, 1, 4, 4]\n",
    "                result = torch.cat([result, mean_std], 1)\n",
    "                # Out dim: [batch, 512 + 1, 4, 4]\n",
    "            \n",
    "            # Conv\n",
    "            result = self.convs[layer_index](result)\n",
    "            \n",
    "            # Not the final layer\n",
    "            if i > 0:\n",
    "                # Downsample for further usage\n",
    "                result = nn.functional.interpolate(result, scale_factor=0.5, mode='bilinear',\n",
    "                                                  align_corners=False)\n",
    "                # Alpha set, combine the result of different layers when input\n",
    "                if i == step and 0 <= alpha < 1:\n",
    "                    result_next = self.from_rgbs[layer_index + 1](image)\n",
    "                    result_next = nn.functional.interpolate(result_next, scale_factor=0.5,\n",
    "                                                           mode = 'bilinear', align_corners=False)\n",
    "                \n",
    "                    result = alpha * result + (1 - alpha) * result_next\n",
    "                    \n",
    "        # Now, result is [batch, channel(512), 1, 1]\n",
    "        # Convert it into [batch, channel(512)], so the fully-connetced layer \n",
    "        # could process it.\n",
    "        result = result.squeeze(2).squeeze(2)\n",
    "        result = self.fc(result)\n",
    "        return result"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
