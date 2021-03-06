# pt-deep-image-colorization

PyTorch implementation of a deep neural network that utilizes a pre-trained VGG19 classification network together with a modified cascaded refinement network to (re-)colorize grayscale input images using only a small dataset of 10.000 images for training (scraped from [unsplash.com](https://unsplash.com/)). The network operates on images in the LAB color space.
 
 ![custom_datatset](unsplash-custom-dataset-10K.png)
 
 ### Setup
 In order to install all dependencies, you can simply create a new virtual environment with anaconda by using `conda env create -f deep-image-colorization.yml`. Afterwards, you can execute the main script and specify your own data paths or hyperparameters using `python main.py --data <PATH> --test_data <PATH> --g_lr <G_LR> --d_lr >D_LR> --batch_size 1 --epochs <E> --n_val <N>`.
 
 ### Results
 Below you can find some example results from the validation (L+AB, AB, L, AB_fake, L+AB_fake) and the testing sets (L, AB_fake, L+AB_fake).
 
![example_0](logs/exp_layernorm/validation/images/25/3.png)
![example_1](logs/exp_layernorm/validation/images/25/20.png)
![example_2](logs/exp_layernorm/validation/images/25/21.png)
![example_3](logs/exp_layernorm/validation/images/25/25.png)
![example_4](logs/exp_layernorm/validation/images/25/10.png)

![example_5](examples/1.png)
![example_6](examples/18.png)
![example_7](examples/73.png)
![example_8](examples/23.png)


