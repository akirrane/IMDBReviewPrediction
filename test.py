import json
from predict import predict_response

# Load the JSON file
with open('movieReviewsTest.json', 'r') as f:
    data = json.load(f)

# Initialize counters
total_reviews = 0
correct_predictions = 0

# Iterate over data
for item in data['data']:
    tag = item['tag']  # 'pos' or 'neg'
    patterns = item['patterns']  # list of reviews
    for review_text in patterns:
        total_reviews += 1
        output = predict_response(review_text)
        if output == tag:
            correct_predictions += 1
        else:
            print(f"Incorrect prediction for review: {review_text[:60]}...")
            print(f"Expected: {tag}, Got: {output}\n")

# Calculate accuracy
accuracy = correct_predictions / total_reviews * 100
print(f"Total reviews: {total_reviews}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
