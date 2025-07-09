document.getElementById('recommendationForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const formObject = {};
    formData.forEach((value, key) => formObject[key] = value);

    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formObject)
    });

    const data = await response.json();
    document.getElementById('result').innerHTML = `
        <strong>✅ Recommended Crop:</strong> ${data.crop} <br/>
        <strong>✅ Recommended Fertilizer:</strong> ${data.fertilizer} <br/><br/>
        <strong>🌾 Alternative Recommendations:</strong><br/>
        ${data.alternatives.map(item => `- ${item.crop}: ${item.fertilizer}`).join('<br/>')}
    `;
});
