import torch 
import numpy as np 
from tqdm import trange, tqdm
import matplotlib.pyplot as plt 
import cv2
from scipy.spatial import ConvexHull
import alphashape
from descartes import PolygonPatch

from datasets.coco_pts import COCO_PTS
from modules.PolyTransform import PolyTransform
# from helpers import angle_between, euclid_dis, unit_vector

def calc_area(mask):
    if type(mask) == torch.Tensor:
        mask = mask.numpy()
    
    return np.sum((mask > 0).reshape(-1), axis=0)

def draw_shapely(shape):
    img = np.zeros((224, 224))

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = np.array([int_coords(shape.exterior.coords)])

    for i in range(exterior.shape[1]):
        x, y = exterior[0, i]
        exterior[0, i] = [np.clip(y, 0, 223), np.clip(x, 0, 223)]

    img = cv2.fillPoly(img, exterior, color=(255,255,255))

    for interior in shape.interiors:
        for i in range(interior.shape[1]):
            x, y = interior[0, i]
            interior[0, i] = [np.clip(y, 0, 223), np.clip(x, 0, 223)]

        img = cv2.fillPoly(img, interior, color=(0,0,0))

    return img

def _get_area(points, alphas):
    alpha_shape = alphashape.alphashape(
        points,
        lambda ind, r: alphas[ind])

    return calc_area(draw_shapely(alpha_shape))

def find_best_alphas(idx, alphas, points, true_area):
    lr = 1
    h = 0.01
    eps = 10 

    # function to minimize
    # f = lambda alpha: _get_area(points, [alphas[i] if i != idx else alpha for i in range(len(alphas))])
    f = lambda alpha: _get_area(points, [alphas[i] if i != idx else alpha for i in range(len(alphas))])
    
    

    return alphas

def regenerate_polygon(true_mask, nodes_list, node_coords, pred_pos):
    # path = findLongestPath(nodes_list, node_coords, true_mask)
    # polygon = ConvexHull(node_coords.reshape(-1, 2)).vertices

    # print("mask area", calc_area(true_mask))

    points = node_coords.reshape(-1, 2)
    alpha = 1 * alphashape.optimizealpha(points)
    # alpha = 0
    hull = alphashape.alphashape(points, alpha)
    # hull = draw_shapely(find_best_alphas(0, [alpha for i in range(points.shape[0])], points, calc_area(true_mask))[1])
    # hull = draw_shapely(find_best_alphas(0, [alpha for i in range(points.shape[0])], points, calc_area(true_mask))[1])


    return draw_shapely(hull)

def main():
    device = torch.device("cpu")

    # model = torch.load("PolyTransform.pt").to(device)
    model = PolyTransform().to(device)

    val_dataset = COCO_PTS('key_pts/key_pts_instances_val2017.json')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = 2,
                                        num_workers = 0,
                                        shuffle = True)

    for in_imgs, true_pos, length, params, contours, _ in tqdm(val_loader, unit='batch'):
        in_imgs = in_imgs.to(device=device, dtype=torch.float32)
        true_pos = true_pos.to(device, dtype=torch.float32).squeeze(1)

        pred_pos = model(in_imgs, contours)

        print("pred_pos", pred_pos.shape)

        for i in range(in_imgs.shape[0]):
            plt.figure(figsize=(15,15))
            ##################
            plt.subplot(2,2,1)
            plt.title("Mask In")
            mask_in = in_imgs[i][0].cpu().detach()
            plt.imshow(mask_in)

            ##################
            plt.subplot(2,2,2)
            plt.title("predicted vertices")
            pred = pred_pos[i].cpu().detach().numpy()
            # pred = pred + (np.array([112, 112]) - np.mean(pred, axis=0))
            # print("pred", pred, pred_pos.shape)
            # p_img = np.zeros((224,224, 3))
            # for j in range(pred.shape[0]):
            #     # print(pred[j].shape, pred[j][0], pred[j][1])
            #     p_img[int(pred[j][1]), int(pred[j][0])] = [255, 0, 0]
            #     # p_img = cv2.circle(p_img, (int(pred[j][0]), int(pred[j][1])), 5, (255,255,255), 1)

            # plt.imshow(p_img)

            p_img = np.zeros((224, 224, 3))
            pts = pred.reshape(-1, 1, 2).astype(np.int32)
            p_img = cv2.polylines(p_img, [pts], True, (255, 0, 0), 1)
            plt.imshow(p_img)


            #################
            plt.subplot(2,2,3)
            # img = np.zeros((224,224))
            # plt.title("Sampling Area")
            # img = cv2.circle(img, (int(params[i][0]), int(params[i][1])), int(params[i][2]), (255,255,255), 1)
            # plt.imshow(img)

            img = np.zeros((224,224,3))
            contours_ = contours[i].reshape(-1, 2).numpy()
            print("res contours", contours_.shape)
            # img = cv2.drawContours(img, contours, -1, (255,0,255), 1)
            for j in range(pred.shape[0]):
                img[int(contours_[j][1]), int(contours_[j][0])] = 255
                # img = cv2.circle(img, (int(contours_[j][0]), int(contours_[j][1])), 2, (255,255,255), 0)

            plt.imshow(p_img)
            plt.title("Restricted Contours")
            plt.imshow(img)

            ################## 
            plt.subplot(2,2,4)

            mask_in = np.array(mask_in, dtype=np.uint8)
            # mask_in = cv2.cvtColor(mask_in, cv2.COLOR_)
            ret, thresh = cv2.threshold(mask_in, 127, 255, 0)
            contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            out = np.zeros((224,224,3))
            out = cv2.drawContours(out, contours_, -1, (255,0,255), 1)

            print("contours", contours_[0].shape)

            plt.title("contours")
            plt.imshow(out)

            plt.show()

if __name__ == "__main__":
    main()