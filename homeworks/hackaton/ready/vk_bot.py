from gensim.models import Word2Vec, KeyedVectors
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
from gensim.similarities.index import AnnoyIndexer
from sklearn.neighbors import KNeighborsClassifier
import json
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import random



def vectorise(question, w2v):
    res_vector = np.zeros(300, dtype=np.float32)
    sentence = re.split('\W+', question)
    for word in sentence:
        try:
            res_vector += np.array(w2v[word])
        except KeyError:
            continue
    return res_vector / len(sentence)
    

def find_nearest_answer(vector):
    # idxs = np.random.randint(0, len(vectorised_data), size=150)
    closest = 1
    best = 0
    for i in vectorised_data.keys():
        curr = cosine(vector, np.array(vectorised_data[str(i)][0]))
        if curr < closest:
            closest = curr
            best = i
    return vectorised_data[str(best)][1]


def create_response(question, w2v):
    question_vec = vectorise(question, w2v)
    response = find_nearest_answer(question_vec)
    return response

def get_random_id():
    return random.getrandbits(31) * random.choice([-1, 1])


def write_msg(user_id, message):
    vk.method('messages.send', {'user_id': user_id, 'message': message, 'random_id':get_random_id()})


my_model = KeyedVectors.load('my_model')
with open('vec_data.json', 'r', encoding='utf8') as f:
    vectorised_data = json.load(f)


# API-ключ созданный ранее
token = "5d1d08c4b4b739b6718b5a9a09aa71332ba4f687aa58321344f706b4351c2a470d3a9011ad64e312dad30"

# Авторизуемся как сообщество
vk = vk_api.VkApi(token=token)

# Работа с сообщениями
longpoll = VkLongPoll(vk)

print('Server started')
# Основной цикл
for event in longpoll.listen():

    # Если пришло новое сообщение
    if event.type == VkEventType.MESSAGE_NEW:
        print(event.text)
        # Если оно имеет метку для меня( то есть бота)
        if event.to_me:
        
            # Сообщение от пользователя
            request = event.text
            
            response = create_response(request, my_model)
            write_msg(event.user_id, response)
                                       