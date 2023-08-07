# HyperClova: LK-D2 모델 테스트

import time
import numpy as np
import json
import random
import argparse
import requests
from dotenv import load_dotenv
import os
load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument('--length_limit', type=int, default=8, help='')
parser.add_argument('--num_cand', type=int, default=19, help='')
parser.add_argument('--random_seed', type=int, default=2023, help='')

hyperclova_api_key = os.getenv("HYPER_CLOVA_KEY")
hyperclova_api_gateway = os.getenv("HYPER_CLOVA_GATEWAY")

args = parser.parse_args()

rseed = args.random_seed
random.seed(rseed)

def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def Hyper_Clova(prompt):
        url = "https://clovastudio.apigw.ntruss.com/testapp/v1/completions/LK-D2"

        request_data = {
            'text': prompt,
            'maxTokens': 512,
            'temperature': 0.2,
            'topK': 0,
            'topP': 1,
            'repeatPenalty': 5.0,
            'start': '###답변:',
            'restart': '',
            'stopBefore': ['영화 목록', '후보 영화'],
            'includeTokens': True,
            'includeAiFilters': True,
            'includeProbs': False
        }

        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            "X-NCP-CLOVASTUDIO-API-KEY": hyperclova_api_key,
            "X-NCP-APIGW-API-KEY": hyperclova_api_gateway,
        }
        response = requests.post(url, data=json.dumps(request_data), headers=headers)
        return response.json()['result']['text']

data_ml_100k = read_json("./ml_100k.json")

u_item_dict = {}
u_item_p = 0
for elem in data_ml_100k:
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in u_item_dict:
            u_item_dict[movie] = u_item_p
            u_item_p +=1
print (len(u_item_dict))
u_item_len = len(u_item_dict)

user_list = []
for i, elem in  enumerate(data_ml_100k):
    item_hot_list = [0 for ii in range(u_item_len)]
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        item_pos = u_item_dict[movie]
        item_hot_list[item_pos] = 1
    user_list.append(item_hot_list)
user_matrix = np.array(user_list)
user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())


pop_dict = {}
for elem in data_ml_100k:
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in pop_dict:
              pop_dict[movie] = 0
        pop_dict[movie] += 1
        
        
i_item_dict = {}
i_item_id_list = []
i_item_user_dict = {}
i_item_p = 0
for i, elem in enumerate(data_ml_100k):
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in i_item_user_dict:
            item_hot_list = [0. for ii in range(len(data_ml_100k))]
            i_item_user_dict[movie] = item_hot_list
            i_item_dict[movie] = i_item_p
            i_item_id_list.append(movie)
            i_item_p+=1
        i_item_user_dict[movie][i] += 1
i_item_s_list = []
for item in i_item_id_list:
    i_item_s_list.append(i_item_user_dict[item])
item_matrix = np.array(i_item_s_list)
item_matrix_sim = np.dot(item_matrix, item_matrix.transpose())

id_list =list(range(0,len(data_ml_100k)))

### user filtering
def sort_uf_items(target_seq, us, num_u, num_i):

    candidate_movies_dict = {} 
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0/dvd
        us_elem = data_ml_100k[us_i]
        us_seq_list = us_elem[0].split(' | ')

        for us_m in us_seq_list:
            if us_m not in target_seq:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.
                candidate_movies_dict[us_m]+=us_w
                
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x:x[-1], reverse=True))
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items


### item filtering
def soft_if_items(target_seq, num_i, total_i, item_matrix_sim, item_dict):
    candidate_movies_dict = {} 
    for movie in target_seq:
        sorted_is = sorted(list(enumerate(item_matrix_sim[item_dict[movie]])), key=lambda x: x[-1], reverse=True)[:num_i]
        for is_i, is_v in sorted_is:
            s_item = i_item_id_list[is_i]
            
            if s_item not in target_seq:
                if s_item not in candidate_movies_dict:
                    candidate_movies_dict[s_item] = 0.
                candidate_movies_dict[s_item] += is_v
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x:x[-1], reverse=True))
    candidate_items = [e[0] for e in candidate_pairs][:total_i]
    return candidate_items


'''
효율성을 높이기 위해, 우리의 첫 번째 단계는 각각의 후보를 기반으로 HyperClova에서 정확한 예측을 얻을 확률이 높은 사용자 시퀀스를 식별하는 것입니다. 
이후에는 이러한 유망한 사용자 시퀀스에 대해 HyperClova API를 활용하여 예측을 생성합니다.
'''
results_data_15 = []
length_limit = args.length_limit
num_u= 12
total_i = args.num_cand
count = 0
total = 0
cand_ids = []
for i in id_list[:1000]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')
    
    candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
    
#     print (elem[-1], '-',seq_list[-1])

    if elem[-1] in candidate_items:
#         print ('HIT: 1')
        count += 1
        cand_ids.append(i)
    else:
        pass
#         print ('HIT: 0')
    total +=1
print (f'count/total:{count}/{total}={count*1.0/total}')
print ('-----------------\n')

# few-shot promting
temp_1_kr = """
후보 영화 목록과 내가 본 영화들을 기반으로, 아래의 질문에 답변해주세요.

후보 영화 목록: The Crow, First Knight, GoldenEye, Die Hard 2, In the Line of Fire, Batman Forever, Young Guns, Terminal Velocity, Clear and Present Danger, Independence Day (ID4), Stargate, The Shadow, Waterworld, Under Siege 2: Dark Territory, Natural Born Killers, Highlander, Money Train, Days of Thunder, Star Trek IV: The Voyage Home.
내가 본 영화들: Cliffhanger, True Lies, Star Trek: The Motion Picture, Speed, Last Man Standing, Demolition Man, Eraser, Last Action Hero.
###질문: 영화를 선택할 때 가장 중요하게 생각하는 특징은 무엇인가요? (내 선호도를 간단하게 요약해주세요)
###답변: 저는 흥미진진한 순간과 긴장감 넘치는 액션 영화를 선호합니다. 또한 과학 판타지 요소가 있는 영화나 강한 모험적 요소를 가진 영화도 좋아합니다. 또한 좋은 스토리와 강한 캐릭터를 가진 영화도 즐깁니다.
###질문: 내 선호도에 따라 내가 본 영화 중 가장 주요한 영화를 5개 선택하십시오 (형식: [내가 본 영화 번호.]).
###답변: [1. Cliffhanger], [2. True Lies], [3. Speed], [4. Last Man Standing], [5. Demolition Man]"
###질문: 내가 본 영화와 유사한 영화 10편을 영어로 추천해주세요 (형식: [본 영화 번호 - 추천 영화]).
###답변: [1. Cliffhanger - Die Hard 2], \n[2. True Lies - Under Siege 2: Dark Territory], \n[3. Speed - Terminal Velocity], \n[4. Last Man Standing - Young Guns], \n[5. Demolition Man - The Shadow], \n[6. Eraser - Clear and Present Danger], \n[7. Last Action Hero - Batman Forever], \n[8. The Crow - Highlander], \n[9. First Knight - Money Train], \n[10. GoldenEye - Days of Thunder]

영화 목록: {}.
내가 본 영화들: {}.
###질문: 영화를 선택할 때 가장 중요하게 생각하는 특징은 무엇인가요? (내 선호도를 간단하게 요약해주세요)
"""

count = 0
total = 0
results_data = []
for i in cand_ids[1:21]:#[:10] + cand_ids[49:57] + cand_ids[75:81]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')[::-1]
    
    candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
    random.shuffle(candidate_items)

    input_1 = temp_1_kr.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]))

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = Hyper_Clova(input_1)
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            print("Error: ", e)
            print("Retrying...")
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5)
        response = Hyper_Clova(input_1)

    split_text = response.rsplit("###답변:")
    
    predictions_1 = split_text[-3].strip().split("###질문:")[0].strip()
    predictions_2 = split_text[-2].strip().split("###질문:")[0].strip()
    predictions = split_text[-1].strip().split("\n\n")[0].strip()

    hit_=0
    if elem[1] in predictions:
        count += 1
        hit_ = 1
    else:
        pass
    total +=1
    
    print (f"GT:{elem[1]}")
    print (f"predictions:{predictions}")
    print (f'PID:{i}; count/total:{count}/{total}={count*1.0/total}\n')
    result_json = {"PID": i,
                   "candidate_items": candidate_items,
                   "seq_list": seq_list[-length_limit:],
                   "Input_1": "영화를 선택할 때 가장 중요하게 생각하는 특징은 무엇인가요? (내 선호도를 간단하게 요약해주세요)",
                   "Input_2": "내 선호도에 따라 본 영화 중 가장 주요한 영화를 선택하십시오 (형식: [본 영화 번호.]).",
                   "Input_3": "내가 본 영화와 유사한 영화 10편을 추천해주세요 (형식: [본 영화 번호 - 추천 영화]).",
                   "GT": elem[1],
                   "Predictions_1": predictions_1,
                   "Predictions_2": predictions_2,
                   "Predictions": predictions,
                   'Hit': hit_,
                   'Count': count,
                   'Current_total':total,
                   'Hit@10':count*1.0/total}
    results_data.append(result_json)

# 테스트 결과 저장
file_dir = f"./results_multi_prompting_len{length_limit}_numcand_{total_i}_seed{rseed}_kr.json"
write_json(results_data, file_dir)