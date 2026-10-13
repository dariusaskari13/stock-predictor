async function predict() {
  const symbolInput = document.getElementById("symbol");
  const resultDiv = document.getElementById("result");

  const symbol = symbolInput.value.trim();
  if (!symbol) {
    resultDiv.innerText = "Please enter a stock symbol.";
    return;
  }

  resultDiv.innerText = "Predicting next-day price for " + symbol + "...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol })
    });

    if (!response.ok) {
      resultDiv.innerText = "Error: " + (await response.text());
      return;
    }

    const data = await response.json();
    const modelNames = {
      linear_regression: "Linear regression",
      moving_average: "Moving average",
      random_forest: "Random forest",
      knn: "K-nearest neighbors",
      svr: "Support vector regression"
    };
    resultDiv.innerText = "Predictions for " + symbol.toUpperCase() + ":\n" +
      Object.entries(modelNames)
        .map(([key, name]) => name + ": $" + data[key].toFixed(2))
        .join("\n");
  } catch (err) {
    resultDiv.innerText = "Failed to connect to server. Is app.py running?";
  }
}
