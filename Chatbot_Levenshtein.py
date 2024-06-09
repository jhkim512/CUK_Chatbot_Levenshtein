
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#레벤슈타인 거리 구하는 함수
def GetLevenshteinDistance(str1, str2):
    # str1의 문자열 길이 + 1 (반복문 편의성을 위해 1을 더함)
    str1_Len = len(str1) + 1
    # str2의 문자열 길이 + 1 (반복문 편의성을 위해 1을 더함)
    str2_Len = len(str2) + 1

    # 2차원 배열 생성, str1_Len x str2_Len 크기
    matrix = [[] for a in range(str1_Len)]
    for i in range(str1_Len):
        matrix[i] = [0 for a in range(str2_Len)]
    
    # 첫 번째 행을 0부터 str2_Len-1까지의 값으로 초기화
    for i in range(str2_Len):
        matrix[0][i] = i

    # 첫 번째 열을 0부터 str1_Len-1까지의 값으로 초기화
    for i in range(str1_Len):
        matrix[i][0] = i


    # 각 문자의 비교 비용을 저장할 변수 초기화
    cost = 0

    # 배열을 채우기 위한 중첩 반복문
    for i in range(1, str1_Len):
        for j in range(1, str2_Len):
            # 문자가 다르면 비용(cost)을 1로 설정, 같으면 0으로 설정
            if str1[i - 1] != str2[j - 1]:
                cost = 1
            else:
                cost = 0
            
            # 추가(add) 연산의 비용 계산
            add_cost = matrix[i - 1][j] + 1
            # 삭제(delete) 연산의 비용 계산
            delete_cost = matrix[i][j - 1] + 1
            # 수정(modify) 연산의 비용 계산
            modify_cost = matrix[i - 1][j - 1] + cost
            # 세 가지 연산 중 최소 비용을 선택하여 배열에 저장
            min_cost = min([add_cost, delete_cost, modify_cost])
            matrix[i][j] = min_cost

    # 최종적으로 배열의 마지막 요소가 Levenshtein 거리
    return matrix[str1_Len - 1][str2_Len - 1]

class SimpleChatBot:
    def __init__(self, filepath): #질문 백터라이징
        self.questions, self.answers = self.load_data(filepath) #질문 TF-IDF변환
        self.vectorizer = TfidfVectorizer() 
        self.question_vectors = self.vectorizer.fit_transform(self.questions) 

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist() #질문열 리스트 저장
        answers = data['A'].tolist()  #답변열 리스트 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        distances = [GetLevenshteinDistance(input_sentence, question) for question in self.questions] #질문과 답변의 Levenshtein 거리 계산
        best_match_index = distances.index(min(distances)) #Levenshtein 거리 가 작은 값의 인덱스 구하기
        return self.answers[best_match_index] #답변 반환

# CSV 파일 경로
filepath = 'ChatbotData.csv'

# 챗봇 인스턴스 생성
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇의 대화반복
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)