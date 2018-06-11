import numpy as np
from tqdm import tqdm
import imageio
import cv2

def smoothen_frame_list(list_frames,conv_weights= [1.0]):
    conv_window = len(conv_weights)
    conv_weights_array = np.array(conv_weights)/np.sum(conv_weights)
    full_probs_array = np.zeros_like(list_frames[0])
    full_probs_array = full_probs_array.reshape(full_probs_array.shape+(1,))
    full_probs_array = np.repeat(full_probs_array,conv_window,axis = 3)

    for i,image in zip(range(conv_window-1),list_frames):
        full_probs_array[:,:,:,i+1] = image
    list_final_frames = list()
    for image in tqdm(list_frames[(conv_window-1):]):
        full_probs_array = np.roll(full_probs_array,shift = -1,axis = 3)
        full_probs_array[:, :, :, -1] = image
        list_final_frames.append(np.tensordot(full_probs_array,conv_weights_array,axes = ([3],[0])))
    return list_final_frames


def export_frame_GIF(image_paths, output_path, duration=.5):
    with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
        for filename in image_paths:
            image = imageio.imread(filename)
            writer.append_data(image)

            
def export_frame_MJPG(image_paths, output_path, duration=.5):
    frame_width, frame_height, _ = imageio.imread(image_paths[0]).shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 1 / duration, (frame_width, frame_height))
    for filename in image_paths:
        image = imageio.imread(filename)
        out.write(image)
    out.release()
