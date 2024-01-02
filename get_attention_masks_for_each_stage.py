
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from model import segformer_mit_b3
from torchvision import transforms as pth_transforms
import os
import matplotlib

def preprocess_image(image_path, tf, patch_size):
    # read image -> convert to RGB -> torch Tensor
    rgb_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = tf(rgb_img)
    _, image_height, image_width = img.shape

    # make the image divisible by the patch size
    w, h = image_width - image_width % patch_size, image_height - image_height % patch_size
    img = img[:, :h, :w].unsqueeze(0)

    w_featmap = img.shape[-1] // patch_size
    h_featmap = img.shape[-2] // patch_size
    return rgb_img, img, w_featmap, h_featmap


def calculate_attentions(img, w_featmap, h_featmap, patch_size, mode = 'bilinear', stage_num=-1):

    attentions = model.get_selfattention_for_any_stage(img.to(device), stage_num=stage_num)
    nh = attentions.shape[1]
    print("attentions shape for last stage: ", attentions.shape)
    print(f"Attentions shape for {stage_num+1}th stage :", attentions[0, :, :, 0].shape, nh, h_featmap, w_featmap )


    # For last stage
    attentions = model.get_last_selfattention(img.to(device))
    print("attentions shape for last stage: ", attentions.shape)
    nh = attentions.shape[1]

    # # For ith stage
    
    # we keep only the output patch attention
    # reshape to image size
    print("Attentions Shape: ", attentions[0, :, :, 0].shape, nh, h_featmap, w_featmap )
    attentions = attentions[0, :, :, 0].reshape(nh, h_featmap, w_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode=mode)[0].detach().cpu().numpy()
    return attentions


def get_attention_masks(image_path, model, transform, patch_size, mode = 'bilinear', stage_num=-1):
    rgb_img, img, w_featmap, h_featmap = preprocess_image(image_path, transform, patch_size)
    attentions = calculate_attentions(img, w_featmap, h_featmap, patch_size, mode = mode, stage_num=stage_num)
    return rgb_img, attentions


def calculate_segformer_stage_attentions(img, num_stages, mode = 'bilinear'):
    stages_data = model.get_attention_outputs(img.to(device))
    stage_attn_output = []

    for i, data in enumerate(stages_data[0:num_stages]):
        stage_attn = data['attn']
        stage_nh = stage_attn.shape[1]

        # we keep only the output patch attention
        # reshape to image size
        stage_attn = stage_attn[0, :, :, 0]
        stage_h, stage_w = int(targetHeight / stage_scale[i]), int(targetWidth / stage_scale[i])
        stage_attn = stage_attn.reshape(stage_nh, stage_h, stage_w)
        stage_attn = F.interpolate(stage_attn.unsqueeze(0), size=(targetHeight, targetWidth), mode=mode)[0].detach().cpu().numpy()
        stage_attn_output.append(stage_attn)

    stage_attn_output = np.concatenate(stage_attn_output, axis=0)
    return stage_attn_output


def get_stage_attention_masks(image_path, model, transform, patch_size, num_stages, mode = 'bilinear'):
    rgb_img, img, w_featmap, h_featmap = preprocess_image(image_path, transform, patch_size)
    attentions = calculate_segformer_stage_attentions(img, num_stages = num_stages, mode = mode)
    return rgb_img, attentions



def create_frames_for_each_head(images_path, model, transform, patch_size, stage_num):
    fig, axes = plt.subplots(3,3, figsize=(15.5,8))
    axes = axes.flatten()
    fig.tight_layout()

    print( type(images_path), len(images_path) )

    for image_path in tqdm.tqdm(images_path):
        # pass
        image_name = image_path.split(os.sep)[-1].split('.')[0]

        rgb_img, attentions = get_attention_masks(image_path, model, transform, patch_size, mode = 'bilinear', stage_num=stage_num)
    #     rgb_img, attentions = get_stage_attention_masks(image_path, model, transform, patch_size, num_stages=3, mode = 'bilinear')

    #     for i in range(len(axes)):
    #         axes[i].clear()
    #         axes[i].imshow(rgb_img)
    #         axes[i].imshow(attentions[i], cmap='inferno', alpha=0.5)
    #         axes[i].axis('off')
    #         axes[i].set_title(titles[i], x= 0.22, y=0.9, va="top")


    ###########################################
        for i in range(len(axes)):
            axes[i].clear()
            if (i < 4):
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i+8], x= 0.20, y=0.9, va="top")

            elif(i==4):
                axes[i].imshow(np.zeros_like(rgb_img))
            else:
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i-1], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i-1+8], x= 0.20, y=0.9, va="top")

            axes[i].axis('off')

    ###########################################

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f'./attention_images/{stage_num}/{image_name}_stage_{stage_num}.png')
    return



def convert_images_to_video(images_dir, output_video_path, fps : int = 20):

    input_images = [os.path.join(images_dir, *[x]) for x in sorted(os.listdir(images_dir)) if x.endswith('png')]

    if(len(input_images) > 0):
        sample_image = cv2.imread(input_images[0])
        height, width, _ = sample_image.shape

        # handles for input output videos
        output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        # create progress bar
        num_frames = int(len(input_images))
        pbar = tqdm.tqdm(total = num_frames, position=0, leave=True)

        for i in tqdm.tqdm(range(num_frames), position=0, leave=True):
            frame = cv2.imread(input_images[i])
            output_handle.write(frame)
            pbar.update(1)

        # release the output video handler
        output_handle.release()

    else:
        pass
    return



def createDir(dirPath):
    if(not os.path.isdir(dirPath)):
        os.mkdir(dirPath)
    else:
        print("present: ", dirPath)
    return



if __name__ == '__main__':
    targetWidth = 1024
    targetHeight = 512

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage_num = 2 # 0-based

    NUM_CLASSES = 19
    MODEL_NAME = f'segformer_mit_b3_stage_{stage_num+1}'

    model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.eval();
    print("Before loading: ")
    checkpoint = torch.load('segformer_mit_b3_cs_pretrain_19CLS_512_1024_CE_loss.pt')
    model.load_state_dict(checkpoint)

    print( f"Num of params: (in MBs): {sum([ p.nelement()*p.element_size() for p in model.parameters() ])/(1024 * 1024):.3f}"  )

    output_dir = '/home/prateek/ThinkAuto/SegFormer-with-AttentionMaps-and-Quantisation/AttentionMasks/'
    patch_size = 32
    stage_scale = [4, 8, 16, 32]
    stage_heads = [1, 2, 5, 8]
    titles = []
    for stage_index, stage_nh in enumerate(stage_heads):
        titles.extend([f"STAGE_{stage_index+1}_HEAD_{x+1}" for x in range(stage_nh)])

    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    input_dir = 'demoVideo/stuttgart_00'
    image_list = sorted(os.listdir(input_dir))
    images_path = [os.path.join(input_dir, x) for x in image_list]
    # print( len(image_list) )


    # matplotlib.font_manager._rebuild()
    font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 4}
    plt.rc('font', **font)
    plt.rcParams['text.color'] = 'white'



    create_frames_for_each_head(images_path, model, transform, patch_size, stage_num)

    # First Save images for each frame for each stage and head
    video_output_dir = os.path.join(output_dir, *['videos'])
    createDir(video_output_dir)
    output_video_path = os.path.join(video_output_dir, *[f"{MODEL_NAME}_stage_{stage_num}_demoVideo.mp4"])
    print(output_video_path)


    convert_images_to_video(f'./attention_images/{stage_num}/', output_video_path)
