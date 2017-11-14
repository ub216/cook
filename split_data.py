import numpy as np
import os, os.path
import argparse

if __name__ == '__main__':
    root_dir = os.getcwd()
    img_dir = "sushi_or_sandwich"
    img_dir = "/".join([root_dir,img_dir])
    classes = [clsNm for clsNm in os.listdir(img_dir) if os.path.isdir("/".join([img_dir,clsNm]))]
    split_ratio= 0.9
    np.random.seed(0)

    os.makedirs("/".join([img_dir,'train']))
    os.makedirs("/".join([img_dir,'val']))
    for cls in classes:
        imgs = [name for name in os.listdir("/".join([img_dir,cls])) if name.endswith(".jpg")]
        no_imgs = len(imgs)
        no_tr_imgs = np.ceil(no_imgs*split_ratio).astype('int')
        no_te_imgs = no_imgs - no_tr_imgs
        print('Folder %s with %d images to be split into %d training and %d testing images' %(cls,no_imgs,no_tr_imgs,no_te_imgs))

        os.makedirs("/".join([img_dir,'train',cls]))
        shuffled_ids = np.random.permutation(range(no_imgs))
        for ids in shuffled_ids[:no_tr_imgs]:
            old_loc = "/".join([img_dir,cls,imgs[ids]])
            new_loc = "/".join([img_dir,'train',cls,imgs[ids]])
            os.rename(old_loc, new_loc)

        os.makedirs("/".join([img_dir,'val',cls]))
        for ids in shuffled_ids[no_tr_imgs:]:
            old_loc = "/".join([img_dir,cls,imgs[ids]])
            new_loc = "/".join([img_dir,'val',cls,imgs[ids]])
            os.rename(old_loc, new_loc)

        os.removedirs("/".join([img_dir,cls]))

