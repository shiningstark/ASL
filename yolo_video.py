import sys
import argparse
from yolo import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# from flask import Flask, render_template, jsonify
import time

# app = Flask(__name__)


# @app.route('/')
# def index():
#     return render_template('settings.html')
#
#
# @app.route('/test', methods=['GET', 'POST'])
# def test():
#     t = time.time()
#     res = jsonify(t)
#     res.headers['Access-Control-Allow-Origin'] = '*'
#     return res


def detect_img(yolo):
    # img = input('Input image filename:')
    img = r"C:\Users\Asimov\Desktop\AR-ASL\jpg\\0003.jpg"
    image = cv2.imread(img)
    image = Image.fromarray(image)
    data = yolo.detect_image(image)
    r_image = yolo.draw(image, None, data)
    plt.subplot(1,2,1)
    plt.imshow(r_image)
    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.show()
    yolo.close_session()




# @app.route('/start', methods=['GET', "POST"])
def start():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./invideo/multi_obj.MOV',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="./outvideo/1.MOV",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    # if FLAGS.image:
    #     """
    #     Image detection mode, disregard any remaining command line arguments
    #     """
    #     print("Image detection mode")
    #     if "input" in FLAGS:
    #         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    #     detect_img(YOLO(**vars(FLAGS)))
    # elif "input" in FLAGS:
    #     detect_video(YOLO(**vars(FLAGS)), FLAGS.input)
    # else:
    #     print("Must specify at least video_input_path.  See usage with --help.")
    # # detect_video(YOLO(**vars(FLAGS)), FLAGS.input)
    detect_img(YOLO(**vars(FLAGS)))

if __name__ == '__main__':
    # app.debug = True
    # app.run()
    start()