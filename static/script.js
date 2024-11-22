document.getElementById('submit-btn').addEventListener('click', function() {
    const reviewText = document.getElementById('review-input').value.trim();
    const responseDiv = document.getElementById('response');

    // Clear previous response
    responseDiv.classList.remove('show');
    responseDiv.innerHTML = '';

    if (reviewText === '') {
        responseDiv.innerHTML = 'Please enter a review before submitting.';
        responseDiv.classList.add('show');
        return;
    }

    fetch('/answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: reviewText })
    })
    .then(response => response.json())
    .then(data => {
        responseDiv.innerHTML = data.answer;
        responseDiv.classList.add('show');
    })
    .catch(error => {
        console.error('Error:', error);
        responseDiv.innerHTML = 'An error occurred. Please try again.';
        responseDiv.classList.add('show');
    });
});
