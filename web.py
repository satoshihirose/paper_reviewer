"""Entry point for the Gradio web app. Run with: python web.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response

import paper_reviewer.app as _app  # noqa: E402
from paper_reviewer.app import demo  # noqa: E402

_JS_SETUP = """
() => {
  const setup = () => {
    // Replace Gradio's built-in upload text
    document.querySelectorAll('span, button').forEach(el => {
      if (el.childElementCount === 0 && el.textContent.includes('アップロード')) {
        el.textContent = el.textContent.replace('アップロード', '開く');
      }
    });

    // Make header clickable: navigate to top screen
    const header = document.getElementById('app-header');
    if (header && !header.dataset.clickBound) {
      header.style.cursor = 'pointer';
      header.title = 'トップに戻る';
      header.dataset.clickBound = '1';
      header.addEventListener('click', () => {
        const btn = document.querySelector('#home-btn button');
        if (btn) btn.click();
      });
    }
  };
  new MutationObserver(setup).observe(document.body, { childList: true, subtree: true });
  setup();
}
"""

app = FastAPI()


@app.get("/pdf/{token}")
async def serve_pdf(token: str):
    path = _app._pdf_sessions.get(token)
    if not path or not Path(path).exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="application/pdf")


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    js=_JS_SETUP,
    css="#home-btn { display:none !important; }",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
