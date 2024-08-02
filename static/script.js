// Predict Button Event Listener
const btn_predict = document.getElementById('predictButton');
let modelsData = {'ticker':'', 'model_list':[] , 'mse_dict': []}

if (btn_predict) {
    btn_predict.onclick = async function() {
        const tickerInput = document.getElementById('ticker').value;
        if (!tickerInput) {
            alert('Please enter a ticker symbol.');
            return;
        }
        console.log("The ticker at Button is:",tickerInput)
        
        modelsData = await fetchModels(tickerInput);

        if (modelsData) {
            await displayModels(modelsData);
            await refreshImage(modelsData);
        }

    };
}

// Function to fetch models and return the data
async function fetchModels(tickerInput) {
    console.log('Fetching models for ticker:', tickerInput);
        const response = await fetch('/fetchmodels', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"symbol": tickerInput})
        });

        console.log('Fetch response status:', response.status, response.statusText);

        
        //const data = await response.json();
        const data = await response.json();
        console.log('Data received:', data.model_list);
        console.log('Data received:', data.ticker);
        console.log('Data received:', data.mse_dict);
        return {
            ticker: data.ticker,
            model_list: data.model_list,
            mse_dict: data.mse_dict
        };

}


// Function to display models using provided modelsData
function displayModels(modelsData) {
    console.log(modelsData.model_list)

    const modelList = modelsData.model_list;
    const mseDict = modelsData.mse_dict;

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

        const modelMSE = document.createElement('span');
        modelMSE.classList.add('model-mse');
        const mse = mseDict[model];
        modelMSE.innerText = mse !== undefined && mse !== null ? `MSE: ${mse.toFixed(2)}` : 'MSE: N/A';

        const deleteButton = document.createElement('button');
        deleteButton.classList.add('delete-button');
        deleteButton.innerText = 'Delete';
        deleteButton.style.marginLeft = '10px';
        deleteButton.type = 'button'; 
        deleteButton.onclick = () => deleteModel(model, modelsData);

        modelItem.appendChild(modelName);
        modelItem.appendChild(modelMSE);
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

        if (!response.ok) {
            throw new Error('Network response was not ok.');
        }

        const plotData = await response.json();
        console.log(plotData)
        const plotObject = JSON.parse(plotData.figure);
        const updatedMseDict = plotData.MSE;

        Plotly.react('plotly-div', plotObject.data, plotObject.layout);

        modelsData.mse_dict = updatedMseDict;

        modelsData.model_list.forEach((model) => {
            const mse = updatedMseDict[model];
            const modelItem = Array.from(document.getElementsByClassName('model-item'))
                .find(item => item.querySelector('.model-name').innerText === model);
            
            if (modelItem) {
                const modelMSE = modelItem.querySelector('.model-mse');
                if (modelMSE) {
                    modelMSE.innerText = mse !== null ? `MSE: ${mse.toFixed(2)}` : 'MSE: N/A';
                }
            }
        });

    } catch (error) {
        console.error('Error refreshing plot:', error);
        alert('An error occurred while refreshing the plot.');
    }
    isRefreshing = false;
}

// Refresh every 10s
const intervalId = setInterval(async () => {
    const tickerInput = document.getElementById('ticker').value;
    const modelData = await fetchModels( tickerInput );
    if (modelData) {
        await displayModels(modelData);
        await refreshImage(modelData);
    }
}, 10000);

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

        try {
            const response = await fetch('/train_new_model/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "ticker": ticker,
                    "num_epochs": num_epochs,
                    "forecast_len": forecast_len
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