



// Function to fetch models and populate checkboxes
async function fetchModels() {
    const inputValue = document.getElementById('ticker').value;

    if (!inputValue) {
        alert('Please enter a ticker symbol.');
        return;
    }

    try {
        const response = await fetch('/fetchmodels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "symbol": inputValue })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }

        const modelList = await response.json();
        console.log(modelList);

        // Clear previous checkboxes
        const checkboxContainer = document.getElementById('checkboxContainer');
        checkboxContainer.innerHTML = '';

        // Populate checkboxes
        modelList.forEach(model => {
            const checkboxWrapper = document.createElement('div');
            checkboxWrapper.classList.add('checkbox-item');

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = model;
            checkbox.name = 'model';
            checkbox.value = model;

            const label = document.createElement('label');
            label.htmlFor = model;
            label.textContent = model;

            checkboxWrapper.appendChild(checkbox);
            checkboxWrapper.appendChild(label);
            checkboxContainer.appendChild(checkboxWrapper);
        });
    } catch (error) {
        console.error('Error fetching models:', error);
        alert('An error occurred while fetching models.');
    }
}

// Function to refresh the image based on selected models
async function refreshImage() {
    const inputValue = document.getElementById('ticker').value;

    if (!inputValue) {
        alert('Please enter a ticker symbol.');
        return;
    }

    // Get all checked model names
    const checkedModels = Array.from(document.querySelectorAll('input[name="model"]:checked'))
                               .map(checkbox => checkbox.value);

    console.log(checkedModels);

    try {
        const response = await fetch('/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "ticker": inputValue,
                "checked_models": checkedModels
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const img = document.createElement('img');
        img.src = url;

        const imageContainer = document.getElementById('image-container');
        imageContainer.innerHTML = ''; // Clear previous image
        imageContainer.appendChild(img);
    } catch (error) {
        console.error('Error refreshing image:', error);
        alert('An error occurred while refreshing the image.');
    }
}

// Add event listener to the Show Prediction button
document.getElementById('showPredictionButton').addEventListener('click', refreshImage);

// Handle form submission
document.getElementById('modelForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission
    refreshImage();
});

// Refresh the image every 30 seconds
setInterval(refreshImage, 30000);


// Get modal elements
const modal = document.getElementById("trainModelModal");
const btn = document.getElementById("trainNewModelButton");
const span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal
btn.onclick = function() {
    const tickerInput = document.getElementById('ticker').value;
    if (!tickerInput) {
        alert('Please enter a ticker symbol.');
        return;
    }
    document.getElementById('train_ticker').value = tickerInput; // Set the ticker in the modal form
    modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
    if (event.target === modal) {
        modal.style.display = "none";
    }
}

document.getElementById('trainModelForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission
    const ticker = document.getElementById('train_ticker').value;
    const num_epochs = document.getElementById('num_epochs').value;
    const forecast_len = document.getElementById('forecast_len').value;

    fetch('/train_new_model/', {
          method: 'POST',
            headers: {
             'Content-Type': 'application/json',
            },
         body: JSON.stringify({ "ticker":ticker, "num_epochs":num_epochs, "forecast_len":forecast_len }),
         }).then(response => response.json())
     .then(data => {
            console.log('Success:', data);
         }).catch((error) => {
            console.error('Error:', error);
         });
    

    modal.style.display = "none";
});