import argparse

def Train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--data_dir', type=str, default='images', help='the directory to get data')
    parser.add_argument('--check_dir', type=str, default='save', help='the directory to save checkpoints')
    parser.add_argument('--image_dir', type=str, default='gen_images', help='the directory to save images')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--img_height', type=int, default=100, help='height of each image')
    parser.add_argument('--img_width', type=int, default=128, help='width of each image')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=2, help='interval betwen image samples')
    parser.add_argument('--resume', type=str, default=None, help='resume from a given checkpoint')
    opt = parser.parse_args()
    # manually set attribute
    setattr(opt, 'img_shape', (opt.channels, opt.img_height, opt.img_width))
    return opt

if __name__ == '__main__':
    A = Train_options()
    print(A.img_shape)