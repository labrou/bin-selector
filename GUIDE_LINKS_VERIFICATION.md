# Guide Links Verification

## How Guide Links Work

1. **User clicks "User guide" button** → Opens user guide dialog via `_show_user_guide()`
2. **User guide markdown is loaded** → `user_guide.md` with placeholders like `%%METHOD_GUIDE_URL%%`
3. **URL substitution** → `_guide_subs()` replaces placeholders with full URLs
   - Base URL constructed from HTTP host header: `http://localhost:PORT`
   - Final URLs: `http://localhost:PORT/app/static/method_explainer.html`
4. **Links are rendered as markdown** → `st.markdown()` creates clickable links
5. **User clicks guide link** → Browser navigates to external URL

## URL Construction

```python
# In _guide_subs() function (app.py line 565-583)
_host  = st.context.headers.get("host", "localhost:8502")  # Gets host from HTTP request
_proto = "https" if not _host.startswith("localhost") else "http"
_base  = f"{_proto}://{_host}"

# Results in URLs like:
# http://localhost:8501/app/static/sort_modes_explainer.html
# http://localhost:8501/app/static/method_explainer.html
# http://localhost:8501/app/static/visualization_explainer.html
```

## Verification Steps

To test if guide links work:

1. **Start the app**: `streamlit run app.py`
2. **Click "User guide" button** in top-right
3. **In the dialog, click guide links**:
   - "How the heatmap, bar, and drill-down connect →" (VIZ_GUIDE_URL)
   - "Visual guide to all methods →" (METHOD_GUIDE_URL)
   - "Visual guide to all sort modes →" (SORT_GUIDE_URL)
4. **Verify**:
   - Links open correctly (same tab or new tab)
   - HTML guide pages load without 404 errors
   - All content displays properly

## Configuration

- **Static file serving**: Enabled via `.streamlit/config.toml`
  ```toml
  [server]
  enableStaticServing = true
  ```

- **Static files location**: `static/` directory at project root
  - All guide HTML files exist and are readable
  - Files are served at `/app/static/` path

- **User guide markdown**: `static/user_guide.md`
  - Contains placeholders for guide URLs
  - Placeholders are substituted at render time

## Potential Issues & Solutions

| Issue | Check |
|-------|-------|
| Links return 404 | Verify `enableStaticServing = true` in config.toml |
| URLs have wrong host | Check browser's host header matches app instance |
| Links don't work on deployed version | Ensure Streamlit Cloud supports `/app/static/` serving |
| Port number wrong in URLs | Verify PORT in the URL matches running instance |

## Related Code

- **Guide setup**: `app.py` lines 563-635
- **Guide markdown**: `static/user_guide.md` lines 12, 52, 77
- **Guide URLs**: `core.py` lines 349-351
- **Config**: `.streamlit/config.toml`
