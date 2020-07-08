from flask import Flask
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
import glob
from skimage.transform import resize
import matplotlib.pyplot as plt
import urllib
from bs4 import BeautifulSoup
import cv2
import os
from os import listdir
from os.path import isfile, join
from random import randrange
from flask import Flask, render_template
from flask import request

# the function scrap frames from an image

def get_frames_from_video(video_path , delay):
    
    """
        video_path is the path to the video with video formate
        e.g '/content/gdrive/My Drive/videos/video.mp4'


        delay is the parameter to fetch a frame from video after a certain delay
        like you want to fetch each frame after 5 seconds


        it will return each frame fetched from the video given by video_path after 
        each delay given by 'delay' parameter in numpy
    
    """
        

    
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    frames = list()
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    i = 0
    while success: 
      
        frames.append(image)
        success,image = vidcap.read()

        print(f'Read a new frame{i}: ', success)
        count += delay*fps
        i+=1
        vidcap.set(1, count)
    return  np.array(frames)


def scrap_faces_from_image(image, cascade_path):

    """
          image: is the numpy multidimensional array having pixel values of images at location

          cascade_path: it is a complete path to an xml document. 
          this xml document contains real values of pretrained model.
          for scraping of faces we can use cascade_frontalface_default or any other varient.
          the related pre trained values can be found at 'https://github.com/opencv/opencv/tree/master/data/haarcascades'

          it will return boxes e.i coordinates of faces detected in the image
          there could be more than one face
          each face is coordinated with x,y,w,h (x and y are the left top corner of the box whereas 'w' is the width of the box and 'h' is the height of the box)
    
    """

    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def crop_faces_from_image(faces, img):

    """
          faces: is the numpy multidimensional array containing coordinates of each faces the image contained
          each face box can be expressed as the following tuple (x,y,w,h)

          img: is the image containing some faces


          return: the function return a multidiemnsional numpy array containg cropped faces according to given coordinates in 'faces'
    """

    face_images = list()
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_image = img[y:y+h, x:x+w]
        face_images.append(face_image)

    return np.array(face_images)



def fetch_files_from_directory(directory):
    """
    
        directory: the source directory which contains files
        the function fetch all files within the directory and return a list of file names
    
    """
  
    return [f for f in listdir(directory) if isfile(join(directory, f))]



def scrap_faces_from_videos(src_dir, des_dir, cascade_path, gray_scale = True, delay=5):

    """
          src_dir: it is a source directory path where video files are placed. this function fetch all video files from that source directory.
          

          des_dir: the destination directory where all scaped faces would be placed


          cascade_path: it is a complete path to an xml document. 
          this xml document contains real values of pretrained model.
          for scraping of faces we can use cascade_frontalface_default or any other varient.
          the related pre trained values can be found at 'https://github.com/opencv/opencv/tree/master/data/haarcascades'

          gray_scale: a boolean variable which determins either you want to save faces in a gray scale or note. Default is True.


          delay: is the parameter to fetch a frame from video after a certain delay
          like you want to fetch each frame after 5 seconds

    
    """

    files = fetch_files_from_directory(src_dir)
    i = 0
    for f in files:
        video_path = f'{src_dir}/{f}'
        frames = get_frames_from_video(video_path, delay)
        croped_faces = list()
        for frame in frames:
            faces = scrap_faces_from_image(frame, cascade_path)
            croped_faces.extend(crop_faces_from_image(faces, frame))

        croped_faces = np.array(croped_faces)

        for face in croped_faces:
            if gray_scale:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f'{des_dir}/face{i}.jpg', face)
        i += 1



def anotate_frame(frame, faces, label):

    """
        frame: a multidimensional numpy array of pixel values representing a frame or an image
        faces: coordinates of the faces in that image that has to be annotated
        label: the text that should appear on boxes 


        return: the anotated image with a box to face and a label on it
        
    
    """
    img = None
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img, label, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 0, 0)   , 2)
    return img



class Grey_32_64_128_gp (nn.Module):
  def __init__(self, n_classes):
    super(Grey_32_64_128_gp, self).__init__()

    self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
    self.conv1_1_bn = nn.BatchNorm2d(32)
    self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
    self.conv1_2_bn = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d((2, 2))

    self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
    self.conv2_1_bn = nn.BatchNorm2d(64)
    self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
    self.conv2_2_bn = nn.BatchNorm2d(64)
    self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
    self.conv2_3_bn = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d((2, 2))

    self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
    self.conv3_1_bn = nn.BatchNorm2d(128)
    self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
    self.conv3_2_bn = nn.BatchNorm2d(128)
    self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
    self.conv3_3_bn = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d((2, 2))

    self.drop1 = nn.Dropout()

    self.fc4 = nn.Linear(128, 128)
    self.fc5 = nn.Linear(128, n_classes)

  def forward(self, x):
    x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
    x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

    x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
    x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
    x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

    x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
    x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
    x = self.pool3(F.relu(self.conv3_3_bn(self.conv3_3(x))))

    x = F.avg_pool2d(x, 4)
    x = x.view(-1, 128)
    x = self.drop1(x)

    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return x

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def predictEmotion(img, net):
  im = resize(img, (48, 48))
  im = rgb2gray(im)  
  expressions = ["Angry","Neutral" , "Fear", "Happy", "Sad"]
  im = im.reshape(1,1,48,48)
  out = net(torch.Tensor(im))
  _, predicted = torch.max(out, 1)
  plt.imshow(im[0,0,:,:])
  label = expressions[predicted]


  # directory path where I've placed some mp4 video files for scraping
  src_dir = 'dramas'
  # directory path for which I want to save scrapped faces from video files
  des_dir = 'drama_faces'
  # cascade xml file of pre trained model to extract faces
  cascade_path = 'haarcascade_frontalface_default.xml'
  faces = scrap_faces_from_image(img, cascade_path)
  return faces, label


# def randomImage(ls, net):
#   if len(faces) == 0:
#     while len(faces) == 0:
#       print("while...")
#       img_path = ls[randrange(len(ls))]
#       print(img_path)
#       img = cv2.imread(img_path)
#       faces, label = predictEmotion(img, net)
#       plt.axis('off')
#       plt.imshow(cv2.cvtColor(anotate_frame(img, faces, label), cv2.COLOR_BGR2RGB))
#       plt.savefig('static/images/prediction.png')
#       return
#   else:
#     print("else...")
#     plt.axis('off')
#     plt.imshow(cv2.cvtColor(anotate_frame(img, faces, label), cv2.COLOR_BGR2RGB))
#     plt.savefig('static/images/prediction.png')

def imagePrediction(img_path, net, fileName):
  img = cv2.imread(img_path)
  faces, label = predictEmotion(img, net)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(anotate_frame(img, faces, label), cv2.COLOR_BGR2RGB))
  plt.savefig('static/images/'+ fileName)
  return len(faces), label
  # plt.show()


PEOPLE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static/images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net_class = Grey_32_64_128_gp
        n_classes = 5
        student_net = net_class(n_classes).to(torch_device)
        teacher_net = net_class(n_classes).to(torch_device)
        student_net.load_state_dict(torch.load('final_PDTS_Fer2013_teacherModelpoint001.pth', map_location=torch_device))
        teacher_net.load_state_dict(torch.load('final_PDTS_Fer2013_teacherModelpoint001.pth', map_location=torch_device))
        teacher_net.eval()
        emotionRealClass = randrange(n_classes)
        lst = glob.glob('drama_frames/*.' + 'jpg')
        facesCount, label = imagePrediction(path, teacher_net, file1.filename)
        # randomImage(lst, teacher_net)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        return render_template("index.html", user_image = full_filename, label = label)
        return 'ok'
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''

# @app.route('/test')
# def hello():
# 	torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 	net_class = Grey_32_64_128_gp
# 	n_classes = 5
# 	student_net = net_class(n_classes).to(torch_device)
# 	teacher_net = net_class(n_classes).to(torch_device)
	# student_net.load_state_dict(torch.load('final_PDTS_Fer2013_teacherModelpoint001.pth', map_location=torch_device))
	# teacher_net.load_state_dict(torch.load('final_PDTS_Fer2013_teacherModelpoint001.pth', map_location=torch_device))
# 	teacher_net.eval()
# 	emotionRealClass = randrange(n_classes)
# 	lst = glob.glob('drama_frames/*.' + 'jpg')
# 	print(imagePrediction(lst[85], teacher_net))
# 	# randomImage(lst, teacher_net)
# 	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '1.png')
# 	return render_template("index.html", user_image = full_filename)

if __name__ == '__main__':
	app.run(debug=True)