// Predict Button Event Listener
const btn_predict = document.getElementById('predictButton');
let modelsData = {'ticker':'AAPL', 'model_list':[] , 'mae_dict': []}

if (btn_predict) {
    btn_predict.onclick = async function() {
        const tickerInput = document.getElementById('ticker').value;
        if (!tickerInput) {
            alert('Please enter a ticker symbol.');
            return;
        }
        
        
        modelsData = await fetchModels(tickerInput);

        if (modelsData) {
            console.log('modelsData:', modelsData);
            await displayModels(modelsData);
            await refreshImage(modelsData);
        }

    };
}

// Function to fetch models and return the data
async function fetchModels(tickerInput) {
    console.log('Fetching models for ticker:', tickerInput);
    
    try {
        const response = await fetch('/fetchmodels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"symbol": tickerInput})
        });

        console.log('Fetch response status:', response.status, response.statusText);
        // Check if the response status is not 200 (OK)
        if (response.status !== 200) {
            alert('Please enter a valid ticker symbol');
            return;
            }

        const data = await response.json();
        console.log('Data:', data);


        // Log data if not empty
        console.log('Data received:', data.model_list);
        console.log('Data received:', data.ticker);
        console.log('Data received:', data.mae_dict);

        return {
            ticker: data.ticker,
            model_list: data.model_list,
            mae_dict: data.mae_dict
        };

    } catch (error) {
        console.error('Error fetching models:', error);
        alert('An error occurred while fetching the models. Please try again later.');
    }
}


function displayModels(modelsData) {
    console.log('Model List:', modelsData.model_list);
    console.log('MAE Dictionary:', modelsData.mae_dict);

    const modelList = modelsData.model_list;
    const maeDict = modelsData.mae_dict;

    const modelListContainer = document.getElementById('modelListContainer');
    if (!modelListContainer) {
        console.error('Error: modelListContainer not found.');
        return;
    }
    modelListContainer.innerHTML = '';

    modelList.forEach((model) => {
        const modelItem = document.createElement('div');
        modelItem.classList.add('model-item');

        const modelName = document.createElement('span');
        modelName.classList.add('model-name');
        modelName.innerText = model;
        modelName.style.marginRight = '10px';

        const modelMAE = document.createElement('span');
        modelMAE.classList.add('model-mae');
        
        // Ensure maeDict is defined and has the property model
        let mae = 'N/A'; // Default value if the key is not found
        if (maeDict && model in maeDict) {
            mae = maeDict[model];
        }
        
        // Display MAE or 'N/A'
        modelMAE.innerText = mae !== undefined && mae !== null ? `MAE: ${mae.toFixed(2)}` : 'MAE: N/A';

        const deleteButton = document.createElement('button');
        deleteButton.classList.add('delete-button');
        deleteButton.innerText = 'Delete';
        deleteButton.style.marginLeft = '10px';
        deleteButton.type = 'button'; 
        deleteButton.onclick = () => deleteModel(model, modelsData);

        modelItem.appendChild(modelName);
        modelItem.appendChild(modelMAE);
        modelItem.appendChild(deleteButton);
        modelListContainer.appendChild(modelItem);
    });
}

let isRefreshing = false;

// Function to refresh the plot based on selected models
async function refreshImage(modelsData) {
    console.log(modelsData)
    if (isRefreshing) return;
    isRefreshing = true;

    try {
        const inputValue = modelsData.ticker;
        console.log(modelsData.ticker)
        console.log(modelsData.model_list)
        const response = await fetch('/generate_plot/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "ticker": inputValue,
                "model_list": modelsData.model_list
            })
        });

        if (response.status === 429) {
            console.error('Rate limit exceeded:', error);
            alert('API requst limit excided for the next 24h ');

            throw new Error('Rate limit exceeded. Please try again later.');

        } else if (!response.ok) {
            console.error('Rate limit exceeded:', error);
            throw new Error('Network response was not ok.');
        }

        const plotData = await response.json();
        console.log(plotData)
        const plotObject = JSON.parse(plotData.figure);
        const updatedMaeDict = plotData.MAE;

        Plotly.react('plotly-div', plotObject.data, plotObject.layout);
        console.log(updatedMaeDict)
        modelsData.mae_dict = updatedMaeDict;
        displayModels(modelsData)
        
    } catch (error) {
        console.error('Error refreshing plot:', error);
    }
    isRefreshing = false;
}

// Refresh every 10s
let refreshCount = 0;
const maxRefreshes = 10;  // Set the maximum number of refreshes

const intervalId = setInterval(async () => {
    const tickerInput = document.getElementById('ticker').value;
    const modelData = await fetchModels(tickerInput);
    
    if (modelData) {
        await refreshImage(modelData);
    }
    
    refreshCount++;  // Increment the counter after each refresh
    
    if (refreshCount >= maxRefreshes) {
        clearInterval(intervalId);  // Stop the interval after 10 refreshes
    }
}, 60000);  // 10 seconds (10000 milliseconds) interval

// Handling new model training modal
const btn_train = document.getElementById("trainNewModelButton");
const span = document.getElementsByClassName("close")[0];
const modal = document.getElementById("trainModelModal");

if (btn_train ) {
    btn_train .onclick = function() {
        const tickerInput = document.getElementById('ticker').value;
        if (!tickerInput) {
            alert('Please enter a ticker symbol.');
            return;
        }
        document.getElementById('train_ticker').value = tickerInput;
        modal.style.display = "block";
    };
}

if (span) {
    span.onclick = function() {
        modal.style.display = "none";
    };
}

window.onclick = function(event) {
    if (event.target === modal) {
        modal.style.display = "none";
    }
}

// Handle form submission for training a new model
const trainForm = document.getElementById('trainModelForm');

if (trainForm) {
    trainForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const ticker = document.getElementById('train_ticker').value;
        const num_epochs = document.getElementById('num_epochs').value;
        const forecast_len = document.getElementById('forecast_len').value;
        const model_type = document.getElementById('model_type').value;
        
        modal.style.display = "none";

        // Show waiting screen
        if (waitingScreen) {
            waitingScreen.style.display = "block";
        }
        console.log(model_type)
        
        try {
            const response = await fetch('/train_new_model/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "ticker": ticker,
                    "num_epochs": num_epochs,
                    "forecast_len": forecast_len,
                    "model_type":model_type
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }

            const data = await response.json();
            console.log('Success:', data);

            // Update the UI
            modelsData = await fetchModels(ticker);
            if (modelsData) {
                await displayModels(modelsData);
                await refreshImage(modelsData);
            }
            modal.style.display = "none";

        } catch (error) {
            console.error('Error:', error);
        } finally {
            // Hide waiting screen
            if (waitingScreen) {
                waitingScreen.style.display = "none";
            }
        }
    });
}

// Handle deleting a model
async function deleteModel(model, modelsData) {
    const ticker = modelsData.ticker;
    console.log('ticker',modelsData.ticker)
    console.log('model list',modelsData.model_list)
    try {
        const response = await fetch('/delete_model/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "ticker": ticker,
                "model": model
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }

        modelsData = await fetchModels(ticker);
        if (modelsData) {
            await displayModels(modelsData);
            await refreshImage(modelsData);
        }

    } catch (error) {
        console.error('Error deleting model:', error);
        alert('An error occurred while deleting the model.');
    }
}