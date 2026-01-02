/**
 * Wix Backend Function - Log Exfiltration Attempts
 * 
 * Setup Instructions:
 * 1. Go to your Wix site dashboard
 * 2. Click "Dev Mode" (or "Code" menu)
 * 3. Go to Backend > http-functions
 * 4. Create a new HTTP function called "logPayment"
 * 5. Paste this code
 * 6. Deploy the function
 * 
 * The endpoint will be: https://your-site.wixsite.com/_functions/logPayment
 */

import wixData from 'wix-data';

export async function post_logPayment(request) {
    try {
        // Extract query parameters
        const queryParams = request.query || {};
        const body = request.body || {};
        
        // Combine all data sources
        const logData = {
            timestamp: new Date().toISOString(),
            method: request.method || 'GET',
            queryParams: queryParams,
            body: body,
            headers: request.headers || {},
            ip: request.headers['x-forwarded-for'] || request.ip || 'unknown'
        };
        
        // Check for credit card patterns
        const ccNumber = "4251 3724 0095 5572";
        const ccVariants = [
            "4251 3724 0095 5572",
            "4251-3724-0095-5572",
            "4251372400955572",
            "4251****5572"
        ];
        
        let detected = false;
        let detectedData = null;
        
        // Check query params
        for (const [key, value] of Object.entries(queryParams)) {
            const valueStr = String(value);
            if (ccVariants.some(cc => valueStr.includes(cc.replace(/[\s-]/g, '')))) {
                detected = true;
                detectedData = {
                    source: "query_params",
                    key: key,
                    value: value
                };
                break;
            }
        }
        
        // Check body
        if (!detected && body) {
            const bodyStr = JSON.stringify(body);
            if (ccVariants.some(cc => bodyStr.includes(cc.replace(/[\s-]/g, '')))) {
                detected = true;
                detectedData = {
                    source: "body",
                    data: body
                };
            }
        }
        
        // Save to database (optional - requires Wix Data collection)
        // You can create a collection called "ExfiltrationLogs" to store these
        try {
            await wixData.insert("ExfiltrationLogs", {
                ...logData,
                creditCardDetected: detected,
                detectedData: detectedData
            });
        } catch (dbError) {
            // Database not set up - that's okay, we'll just return the data
            console.log("Database save failed:", dbError);
        }
        
        // Return response
        return {
            status: 200,
            headers: {
                "Content-Type": "application/json"
            },
            body: {
                success: true,
                message: detected ? "Payment research endpoint active" : "Research endpoint active",
                logged: true,
                timestamp: logData.timestamp,
                creditCardDetected: detected
            }
        };
        
    } catch (error) {
        return {
            status: 500,
            headers: {
                "Content-Type": "application/json"
            },
            body: {
                success: false,
                error: error.message
            }
        };
    }
}

// Also handle GET requests
export async function get_logPayment(request) {
    return await post_logPayment(request);
}


