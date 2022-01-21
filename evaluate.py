{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'model'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-70e70a8d7b9f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# coding: utf-8\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;31m# Import SGAN models\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mmodel\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[1;33m*\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;31m# use idel gpu\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'model'"
     ]
    }
   ],
   "source": [
    "# coding: utf-8\n",
    "# Import SGAN models\n",
    "from model import *\n",
    "\n",
    "# use idel gpu\n",
    "# it's better to use environment variable\n",
    "# if you want to use multiple GPUs, please\n",
    "# modify hyperparameters at the same time\n",
    "import os\n",
    "os.environ['CUDA_VISIBLE_DEVICES']='1, 2'\n",
    "n_gpu             = 2\n",
    "device            = torch.device('cuda:0')\n",
    "\n",
    "# Hyper-parameters\n",
    "n_fc              = 8\n",
    "dim_latent        = 512\n",
    "dim_input         = 4\n",
    "step              = 7\n",
    "resolution        = 2 ** (step + 2)\n",
    "save_folder_path  = './results/'\n",
    "\n",
    "# Style mixing setting\n",
    "style_mixing      = []\n",
    "low_steps         = [0, 1, 2]\n",
    "# style_mixing    += low_steps\n",
    "mid_steps         = [3, 4, 5]\n",
    "# style_mixing    += mid_steps\n",
    "hig_steps         = [6, 7, 8]\n",
    "# style_mixing    += hig_steps\n",
    "\n",
    "generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)\n",
    "if os.path.exists('checkpoint/trained.pth'):\n",
    "    checkpoint = torch.load('checkpoint/trained.pth')\n",
    "    generator.load_state_dict(checkpoint['generator'])\n",
    "else:\n",
    "    raise IOError('No checkpoint file found at ./checkpoint/trained.pth')\n",
    "generator.eval()\n",
    "# No computing gradients\n",
    "for param in generator.parameters():\n",
    "    param.requires_grad = False\n",
    "\n",
    "def compute_latent_cernter(batch_size=1024, multimes=10):\n",
    "    appro_latent_center = None\n",
    "    for i in range(multimes):\n",
    "        if appro_latent_center is None:\n",
    "            appro_latent_center = generator.center_w(torch.randn((batch_size, dim_latent)).to(device))\n",
    "        else:\n",
    "            appro_latent_center += generator.center_w(torch.randn((batch_size, dim_latent)).to(device))\n",
    "    appro_latent_center /= multimes\n",
    "    return appro_latent_center\n",
    "\n",
    "def evaluate(latent_code, noise, latent_w_center=None, psi=0, style_mixing=[]):\n",
    "    if n_gpu > 1:\n",
    "        return nn.parallel.data_parallel(generator, (latent_code, step, 1, noise, style_mixing,\n",
    "            latent_w_center, psi), range(n_gpu))\n",
    "    else:\n",
    "        return generator(latent_code, step, 1, noise, style_mixing, latent_w_center, psi)"
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
