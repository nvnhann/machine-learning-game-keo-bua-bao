import pygame
import button
import random
import time
pygame.init()

#create game window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("GAME KÉO BÚA BAO")

#game variables
game_paused = False
menu_state = "main"
select_icon = False

computer_select = 0
user_select = 0
#ROCK, PAPER OR SCISSORS
list_select = {
    "bao" : 0 ,
    "bua" : 1,
    "keo" : 2
}
user_cmd = None
who_win = None
#define fonts
font = pygame.font.SysFont("arialblack", 30)

#define colours
TEXT_COL = (255, 255, 255)

#load button images
resume_img = pygame.image.load("images/button_resume.png").convert_alpha()
quit_img = pygame.image.load("images/button_quit.png").convert_alpha()
bao_img = pygame.image.load("images/button_bao.png").convert_alpha()
bua_img = pygame.image.load("images/button_bua.png").convert_alpha()
keo_img = pygame.image.load("images/button_keo.png").convert_alpha()
cam_img = pygame.image.load("images/button_camera.png").convert_alpha()


#create button instances    
resume_button = button.Button(304, 125, resume_img, 1)
quit_button = button.Button(336, 250, quit_img, 1)
bao_button = button.Button(30,200,bao_img,1)
bua_button = button.Button(300,200,bua_img,1)
keo_button = button.Button(570,200,keo_img,1)
cam_button = button.Button(700,100,cam_img,1)

def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  screen.blit(img, (x, y))

music = pygame.mixer.music.load('nhacnen.mp3')
pygame.mixer.music.play(-1)
#faster -rcnn

# import the opencv library
import cv2
import os
import six
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
num_threads = 5
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["TF_NUM_INTRAOP_THREADS"] = "5"
os.environ["TF_NUM_INTEROP_THREADS"] = "5"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
import importlib
importlib.reload(vis_util)

PATH_TO_CKPT = 'inference_graph' + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('inference_graph','faster_rcnn_inception_v2_custom_dataset.pbtxt')

NUM_CLASSES = 3
frcnn_graph = tf.Graph()
with frcnn_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



def get_classes_name_and_scores(
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.7): # returns bigger than 90% precision
    display_str = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            if classes[i] in six.viewkeys(category_index):
                display_str['name'] = category_index[classes[i]]['name']
                display_str['score'] = '{}%'.format(int(100 * scores[i]))

    return display_str


def PIL_to_numpy(image):
  (w, h) = image.size

  return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess):
  
    # Get handles to input and output tensors
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)
        
    if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
                                    'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
IMAGE_PATHS = [ os.path.join('', 'image.jpg')]
#end
#game loop
run = True
while run:

  screen.fill((52, 78, 91))

  #check if game is paused
  if game_paused == False:
    #check menu state
    if menu_state == "main":
      #draw pause screen buttons
      if resume_button.draw(screen):
        game_paused = True
      if quit_button.draw(screen):
        run = False
    #check if the options menu is open
  else:
    if select_icon == False:
        draw_text("BAN HAY CHON KEO, BUA HOAC BAO", font, TEXT_COL, 80, 500)
        if cam_button.draw(screen):
            print('bat cam')
            vid = cv2.VideoCapture(0)
            x_ke = 0
            x_ba = 0
            x_bu = 0
            with frcnn_graph.as_default():
                with tf.compat.v1.Session() as sess:
                    while True:
                        ret, frame  = vid.read()
                        cv2.imwrite(filename='image.jpg',img=frame)
                        cv2.imshow('cap',frame)
                        for image_path in IMAGE_PATHS:
                            image = Image.open(image_path)
                            image_np = PIL_to_numpy(image)

                        # image_np_expanded = np.expand_dims(image_np, axis=0)
                        
                            output_dict = run_inference_for_single_image(image_np, sess)
                        
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                output_dict['detection_boxes'],
                                output_dict['detection_classes'],
                                output_dict['detection_scores'],
                                category_index,
                                instance_masks=output_dict.get('detection_masks'),
                                use_normalized_coordinates=True,
                                line_thickness=8)

                            plt.imshow(image_np)
                            plt.savefig("test.jpg")
                            img_new = cv2.imread('test.jpg',cv2.IMREAD_COLOR)
                            cv2.imshow('frame',cv2.resize(img_new,(1000,800)))
                            str_obj = get_classes_name_and_scores(
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index)
                            
                            print(str_obj)
                
                            
                            print(str_obj)
                            try:
                                if str_obj["name"] == "bao":
                                    x_ba +=1
                                    
                                    if x_ba >=5 :
                                        user_select = list_select["bao"]
                                        user_cmd = "BAN DA CHON BAO"
                                        select_icon = True
                                        break
                                elif str_obj["name"] == "bua":
                                    x_bu +=1
                                    
                                    if x_bu >=5 :
                                        user_select = list_select["bua"]
                                        user_cmd = "BAN DA CHON BUA"
                                        select_icon = True
                                        break
                                else:
                                    x_ke +=1
                                    
                                    if x_ke >=5:
                                        user_select = list_select["keo"]
                                        user_cmd = "BAN DA CHON KEO"
                                        select_icon = True
                                        break
                            except:
                               pass
                        cv2.imshow('image', cv2.resize(image_np,(1000,800)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
        if bao_button.draw(screen):
            user_select = list_select["bao"]
            user_cmd = "BAN DA CHON BAO"
            select_icon = True
        if bua_button.draw(screen):
            user_select = list_select["bua"]
            user_cmd = "BAN DA CHON BUA"
            select_icon = True
        if keo_button.draw(screen):
            user_select = list_select["keo"]
            user_cmd = "BAN DA CHON KEO"
            select_icon = True
    else:
        if who_win == None:
            draw_text(user_cmd + ", VUI LONG CHO MAY CHON", font, TEXT_COL, 20, 200)
            pygame.display.update()
            time.sleep(1)
            computer_select = random.randint(0,2)
            if user_select == list_select["bao"] and computer_select == list_select["bao"]:
                who_win = "hoa"
            elif user_select == list_select["bao"] and computer_select == list_select["keo"]:
                who_win = "computer"
            elif user_select == list_select["bao"] and computer_select == list_select["bua"]:
                who_win = "user"
            elif user_select == list_select["bua"] and computer_select == list_select["bao"]:
                who_win = "computer"
            elif user_select == list_select["bua"] and computer_select == list_select["keo"]:
                who_win = "user"
            elif user_select == list_select["bua"] and computer_select == list_select["bua"]:
                who_win = "hoa"
            elif user_select == list_select["keo"] and computer_select == list_select["bao"]:
                who_win = "user"
            elif user_select == list_select["keo"] and computer_select == list_select["keo"]:
                who_win = "hoa"
            elif user_select == list_select["keo"] and computer_select == list_select["bua"]:
                who_win = "computer"
        else:
            if who_win == "hoa":
                user_cmd = "BAN VA MAY DA HOA"
            elif who_win == "computer":
                user_cmd = "MAY DA THANG BAN"
            elif who_win == "user" :
                user_cmd = "BAN DA THANG MAY"
            if computer_select == list_select["bao"]:
                button.Button(550,350,bao_img,1).draw(screen)
            elif computer_select == list_select["bua"]:
                button.Button(550,350,bua_img,1).draw(screen)
            elif computer_select == list_select["keo"]:
                button.Button(550,350,keo_img,1).draw(screen)
            
            if user_select == list_select["bao"] :
               button.Button(50,350,bao_img,1).draw(screen)
            elif user_select == list_select["bua"]:
                button.Button(50,350,bua_img,1).draw(screen)
            elif user_select == list_select["keo"]:
                button.Button(50,350,keo_img,1).draw(screen)
            draw_text(user_cmd, font, TEXT_COL, 230, 50)
            user_cmd = "VS"
            draw_text(user_cmd, font, TEXT_COL, 380, 400)
            if resume_button.draw(screen):
                game_paused = True
                select_icon = False
                user_cmd = None
                who_win = None
            if quit_button.draw(screen):
                run = False
  #event handler
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_SPACE:
        game_paused = False
    if event.type == pygame.QUIT:
      run = False

  pygame.display.update()

pygame.quit()