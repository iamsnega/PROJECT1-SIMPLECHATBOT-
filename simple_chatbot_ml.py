import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load dataset
with open('simple_chatbot_dataset.json') as f:
    data = json.load(f)

inputs = [conv['input'] for conv in data['conversations']]
responses = [conv['response'] for conv in data['conversations']]

# Create and train the model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(inputs, responses)
# Inference function
def get_response(user_input):
    return model.predict([user_input])[0]

# Chatbot function
def chat():
    print("Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Goodbye! Have a great day!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
