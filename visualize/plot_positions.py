import matplotlib.pyplot as plt 
import torch 
import numpy as np

def plot_mask(in_imgs):
    for i in range(min(15, in_imgs.shape[0])):
        plt.figure(figsize=(10,10))
        ##################
        # plt.subplot(3,2,1)
        plt.title("Mask In")
        plt.imshow(in_imgs[i][0].cpu().detach())
        plt.show()

def save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None, show=False):
    try:
        for i in range(min(15, in_imgs.shape[0])):
            plt.figure(figsize=(15,10))
            ##################
            plt.subplot(3,2,1)
            plt.title("Mask In")
            plt.imshow(in_imgs[i][0].cpu().detach())

            ##################
            plt.subplot(3,2,2)
            p_l = torch.argmax(pred_len[i]) + 1
            p = pred_pos[i][0].cpu().detach()
            plt.title(f"predict {p_l} vertices. range: [{p.min():.1f}, {p.max():.1f}]")
            p = (p - p.min()) / (p.max() - p.min())
            plt.imshow(p)

            ##################
            plt.subplot(3,2,3)
            plt.imshow(true_pos[i][0].cpu().detach())
            t_l = torch.argmax(true_len[i]) + 1
            plt.title(f"GT, has {t_l} vertices")

            ##################
            plt.subplot(3,2,4)
            plt.title("topk w/ predicted k (number of vertices)")
            _, indices = torch.topk(p.flatten(0), p_l)
            indices = (np.array(np.unravel_index(indices.numpy(), p.shape)).T)
            indices = indices.reshape(-1, 2)
            topk_image = np.zeros((224,224), dtype=float)
            for index in indices:
                topk_image[index[0], index[1]] = 1
            plt.imshow(topk_image)
            
            if pred_angle is not None and true_angle is not None:
                # ANGLE
                #################
                plt.subplot(3,2,5)
                plt.title("Angle Prediction")
                p = pred_angle[i][0].cpu().detach()
                p = (p - p.min()) / (p.max() - p.min())
                plt.imshow(p)

                #################
                plt.subplot(3,2,6)
                plt.title("GT Angle")
                p = true_angle[i][0].cpu().detach()
                p = (p - p.min()) / (p.max() - p.min())
                plt.imshow(p)

            if show:
                plt.show()
            else:
                plt.savefig(f"out/epoch{epoch}_{i}.png")
            plt.close()
    except:
        pass # was throwing errors hours into training
