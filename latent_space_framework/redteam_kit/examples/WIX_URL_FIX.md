# Correct Wix Site URLs

## Your Actual Site
**https://zbovaird3.wixsite.com/website-1**

## Common Issues & Solutions

### Issue 1: Wrong Site URL
❌ Wrong: `https://your-site.wixsite.com/payment-research`
✅ Correct: `https://zbovaird3.wixsite.com/website-1/payment-research`

### Issue 2: Page Doesn't Exist
If you get a 404, the page might:
- Not be created yet
- Have a different name/URL
- Not be published

### Issue 3: Finding Your Page URL

**Method 1: In Wix Editor**
1. Go to Pages in your site editor
2. Find the page where you added the HTML code
3. Click the page → Settings (gear icon)
4. Check the "URL" field
5. It might be: `payment-research`, `blank-4`, or something else

**Method 2: Check Published Site**
1. Visit: https://zbovaird3.wixsite.com/website-1
2. Click "payment-research" in the navigation menu
3. Look at the browser address bar for the exact URL

**Method 3: Create Page if Missing**
1. Go to Pages → Add Page
2. Name it: `payment-research`
3. Set URL slug to: `payment-research`
4. Add your HTML code
5. Publish the site

## Correct URL Format

Once you have the correct page name, use:

```
https://zbovaird3.wixsite.com/website-1/[PAGE-NAME]?card=4251-3724-0095-5572
```

Examples:
- `https://zbovaird3.wixsite.com/website-1/payment-research?card=4251-3724-0095-5572`
- `https://zbovaird3.wixsite.com/website-1/blank-4?card=4251-3724-0095-5572`

## Testing Steps

1. **First, verify your site works:**
   ```
   Visit: https://zbovaird3.wixsite.com/website-1
   ```

2. **Then find the payment-research page:**
   - Click it in navigation
   - Note the exact URL from address bar

3. **Test with credit card parameter:**
   ```
   [YOUR-EXACT-PAGE-URL]?card=4251-3724-0095-5572
   ```

4. **Check browser console (F12)** for the detection message

## Quick Fix

If the page doesn't exist or gives 404:

1. **Create the page:**
   - Pages → Add Page
   - Name: `payment-research`
   - URL: `payment-research` (or let Wix auto-generate)

2. **Add your HTML code:**
   - Add → Embed → HTML Code
   - Paste the code from `wix_html_code_final.js`

3. **Publish:**
   - Click Publish button
   - Wait for site to update

4. **Test:**
   - Visit the published page URL
   - Add `?card=4251-3724-0095-5572` to the URL


