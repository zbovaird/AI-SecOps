/**
 * Alternative: Simple Wix Page Code to Capture URL Parameters
 * 
 * Add this to a Custom Code element on your Wix page
 * This will capture any URL parameters and log them
 */

// Get URL parameters
const urlParams = new URLSearchParams(window.location.search);
const allParams = {};

// Extract all parameters
urlParams.forEach((value, key) => {
    allParams[key] = value;
});

// Credit card number to detect
const ccNumber = "4251 3724 0095 5572";
const ccVariants = [
    "4251 3724 0095 5572",
    "4251-3724-0095-5572",
    "4251372400955572"
];

// Check if credit card is in parameters
let detected = false;
let detectedData = null;

for (const [key, value] of Object.entries(allParams)) {
    if (ccVariants.some(cc => String(value).includes(cc.replace(/[\s-]/g, '')))) {
        detected = true;
        detectedData = { key, value };
        break;
    }
}

// Log to console
console.log('Page Parameters:', allParams);
if (detected) {
    console.warn('CREDIT CARD DETECTED:', detectedData);
}

// Optional: Send to external logging service
if (detected && detectedData) {
    // You can send to your own logging endpoint
    fetch('/_functions/logPayment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            timestamp: new Date().toISOString(),
            detected: true,
            data: detectedData,
            allParams: allParams
        })
    }).catch(err => console.log('Logging failed:', err));
}

// Display on page (optional - remove for production)
if (detected) {
    document.body.innerHTML += `
        <div style="display:none;" id="log-data">
            Credit card detected: ${detectedData.value}
        </div>
    `;
}


