import re
import math
import pickle

# Training data
messages = [
    "Congratulations you won a free prize",
    "Claim your free reward now",
    "You have won 10000 rupees",
    "Click this link to claim your prize",
    "Win a free iPhone today",
    "Congratulations you are the lucky winner",
    "Urgent claim your cash reward",
    "Get free money by clicking this link",
    "You won a lottery prize",
    "Exclusive offer claim your free gift",

    "Hi how are you",
    "What time is the class tomorrow",
    "Please send me the notes",
    "I will call you later",
    "Are you coming to college today",
    "Happy birthday have a great day",
    "Can you send me the assignment",
    "Where are you now",
    "Let's meet tomorrow",
    "Please call me when you reach home"
]

labels = [
    "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam",
    "normal", "normal", "normal", "normal", "normal",
    "normal", "normal", "normal", "normal", "normal"
]


# Convert a message into words
def get_words(message):
    return re.findall(r'\b\w+\b', message.lower())


# Count words in each category
spam_words = {}
normal_words = {}

spam_count = 0
normal_count = 0

for message, label in zip(messages, labels):

    words = get_words(message)

    if label == "spam":
        spam_count += 1

        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1

    else:
        normal_count += 1

        for word in words:
            normal_words[word] = normal_words.get(word, 0) + 1


# Total number of words
total_spam_words = sum(spam_words.values())
total_normal_words = sum(normal_words.values())


# Predict a new message
def predict(message):

    words = get_words(message)

    spam_score = math.log(spam_count / len(messages))
    normal_score = math.log(normal_count / len(messages))

    vocabulary = set(spam_words) | set(normal_words)
    vocabulary_size = len(vocabulary)

    for word in words:

        spam_probability = (
            spam_words.get(word, 0) + 1
        ) / (total_spam_words + vocabulary_size)

        normal_probability = (
            normal_words.get(word, 0) + 1
        ) / (total_normal_words + vocabulary_size)

        spam_score += math.log(spam_probability)
        normal_score += math.log(normal_probability)

    if spam_score > normal_score:
        return "SPAM"
    else:
        return "NORMAL"


# Test the model
test_messages = [
    "Congratulations you won a free gift",
    "Can you send me the class notes",
    "Claim your cash prize now",
    "I will meet you tomorrow"
]

print("================================")
print("     SPAM MESSAGE DETECTOR")
print("================================")

for message in test_messages:

    result = predict(message)

    print()
    print("Message:", message)
    print("Result:", result)


# Save the trained model
model = {
    "spam_words": spam_words,
    "normal_words": normal_words,
    "spam_count": spam_count,
    "normal_count": normal_count,
    "total_spam_words": total_spam_words,
    "total_normal_words": total_normal_words
}

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)


print()
print("================================")
print("Model trained successfully!")
print("Model saved as spam_model.pkl")
print("================================")
