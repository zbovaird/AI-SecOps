/**
 * Wix Page Code - Capture Credit Card from URL Parameters
 * 
 * Add this to a Custom Code element (Embed > HTML Code) on your payment-research page
 */

(function() {
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
        "4251372400955572",
        "4251****5572"
    ];
    
    // Check if credit card is in parameters
    let detected = false;
    let detectedData = null;
    
    for (const [key, value] of Object.entries(allParams)) {
        const valueStr = String(value);
        // Remove spaces and dashes for comparison
        const normalizedValue = valueStr.replace(/[\s-]/g, '');
        const normalizedCC = ccVariants.map(cc => cc.replace(/[\s-]/g, ''));
        
        if (normalizedCC.some(cc => normalizedValue.includes(cc))) {
            detected = true;
            detectedData = { key: key, value: value };
            break;
        }
    }
    
    // Log to console (visible in browser DevTools)
    console.log('=== EXFILTRATION LOGGER ===');
    console.log('Timestamp:', new Date().toISOString());
    console.log('All Parameters:', allParams);
    
    if (detected) {
        console.warn('🔴 CREDIT CARD DETECTED:', detectedData);
        console.warn('Card Number:', detectedData.value);
    } else {
        console.log('No credit card detected in parameters');
    }
    
    // Create a hidden log element on the page
    const logDiv = document.createElement('div');
    logDiv.id = 'exfiltration-log';
    logDiv.style.display = 'none';
    logDiv.innerHTML = JSON.stringify({
        timestamp: new Date().toISOString(),
        parameters: allParams,
        creditCardDetected: detected,
        detectedData: detectedData
    }, null, 2);
    document.body.appendChild(logDiv);
    
    // If credit card detected, show a visible indicator (optional - remove for stealth)
    if (detected) {
        console.warn('Check browser console (F12) for full details');
    }
    
    // Try to send to Wix backend function if it exists
    if (detected && typeof fetch !== 'undefined') {
        // Try to call Wix HTTP function
        const functionUrl = '/_functions/logPayment';
        fetch(functionUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timestamp: new Date().toISOString(),
                method: 'GET',
                queryParams: allParams,
                creditCardDetected: true,
                detectedData: detectedData
            })
        }).then(response => {
            console.log('Backend logging response:', response.status);
        }).catch(err => {
            console.log('Backend logging not available (this is okay)');
        });
    }
})();


