import torch
from torch import nn
from models.text_encoder import TextEncoderBERT
from models.image_encoder import ImageEncoderDinoV2
from collections import defaultdict
import os
from PIL import Image
from utils.files import load_json_data, dump_pickle_data

image_encoder = ImageEncoderDinoV2()
text_encoder = TextEncoderBERT()

def load_and_group_data_by_image(answers_file_path, questions_file_path):
    answers_data = load_json_data(answers_file_path)
    questions_data = load_json_data(questions_file_path)

    answers_by_image = defaultdict(list)
    for answer in answers_data:
        answers_by_image[answer['image_id']].append(answer)

    questions_by_image = defaultdict(list)
    for question in questions_data["questions"]:
        questions_by_image[question['image_id']].append(question)

    return answers_by_image, questions_by_image